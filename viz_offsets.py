#!/usr/bin/env python3
"""
viz_offsets.py
Full-suite visualization + per-model DCN/flow offset diagrams for DUALQUANTA.

Runs every checkpoint in checkpoints/table1_mono/ (same weights as the
optical_flow_ablation project's runs/ablation_v17/*/best.pth — verified by
matching epoch/best_psnr_gam metadata), reconstructs each model from its
paired configs/mono/<name>.yaml (these checkpoints carry no embedded
architecture metadata), and produces:

  1. Combined panels (predictions row + offset row) per scene, with a
     direction color-wheel legend.
  2. Individual full-resolution offset diagrams per model per scene.

Offset extraction replicates the internal alignment computation for each
mode (PairwiseDCNAlign / FlowGuidedDCNAlign / CascadeLevel), since
SPADNet.forward() only returns the final reconstruction, not the
intermediate per-neighbour DCN offsets.

Output:
  runs/viz_offsets/viz_full/<scene>_s####.png          (combined panels)
  runs/viz_offsets/individual/<scene>_s####/*.png       (per-model diagrams)
"""
import argparse, os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import SPADNet, SPADDataset, build_first_window
from src.utils import load_config, load_checkpoint
from evaluate import load_quiver_model, _is_quiver_ckpt

ROOT      = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR  = os.path.join(ROOT, 'checkpoints', 'table1_mono')
CONFIG_DIR = os.path.join(ROOT, 'configs', 'mono')
TEST_ROOT_DEFAULT = '/home/samuel/dataset/i2-2kfps_v1_png/test'

p = argparse.ArgumentParser()
p.add_argument('--test_root', default=TEST_ROOT_DEFAULT)
p.add_argument('--device', default='cuda:0')
p.add_argument('--out_dir', default=os.path.join(ROOT, 'runs', 'viz_offsets'))
args = p.parse_args()

DEVICE   = torch.device(args.device)
VIZ_OUT  = os.path.join(args.out_dir, 'viz_full')
IND_OUT  = os.path.join(args.out_dir, 'individual')
os.makedirs(VIZ_OUT, exist_ok=True)
os.makedirs(IND_OUT, exist_ok=True)

GAMMA = 2.2

# ── Model list ─────────────────────────────────────────────────────────────────
# (checkpoint stem, short label for panels)
RUNS = [
    ('single_frame_nafnet', 'Single'),
    ('no_align',            'NoAlign'),
    ('dcn_h2',              'K_h2'),
    ('dcn_h4',              'K_h4'),
    ('dcn_h8',              'K_h8'),
    ('dcn_h16',             'K_h16'),
    ('cascading_dcn',       'Cascade'),
    ('oracle_flow',         'Oracle'),
    ('spynet_dcn',          'SpyNet'),
    ('quiver_retrained',    'QUIVER'),
]
# Modes with an alignment/cascade module worth visualizing an offset for.
OFFSET_STEMS = {'dcn_h2', 'dcn_h4', 'dcn_h8', 'dcn_h16',
                'cascading_dcn', 'oracle_flow', 'spynet_dcn'}


def build_model(stem, device):
    ckpt_path = os.path.join(CKPT_DIR, f'{stem}.pth')
    if not os.path.isfile(ckpt_path):
        return None, None
    ck = load_checkpoint(ckpt_path, device)
    if _is_quiver_ckpt(ck):
        model, mode, _ = load_quiver_model(ckpt_path, device)
        return model, mode

    cfg = load_config(os.path.join(CONFIG_DIR, f'{stem}.yaml'))['model']
    model = SPADNet(T=cfg.get('T', 11), bc=cfg.get('base_channels', 32),
                    nb=cfg.get('n_blocks', 2), nfpm=cfg.get('n_fpm', 2),
                    mode=cfg['mode'], raft_ckpt=cfg.get('raft_ckpt', ''))
    model.load_state_dict(ck['model'], strict=True)
    model.to(device).eval()
    ep = ck.get('epoch', '?')
    best = ck.get('best_psnr_gam', float('nan'))
    print(f"  Loaded {stem:<22} mode={cfg['mode']:<14} ep={ep}  best={best:.4f}  "
          f"params={sum(x.numel() for x in model.parameters())/1e6:.2f}M")
    return model, cfg['mode']


print("Loading models…")
nets  = {}
modes = {}
for stem, _ in RUNS:
    m, mode = build_model(stem, DEVICE)
    if m is None:
        print(f"  [skip] {stem} — checkpoint not found")
        continue
    nets[stem] = m
    modes[stem] = mode

HALF_WIN = 5          # T=11 center index for all non-single-frame modes
PREFER_T = HALF_WIN - 1   # immediate-left neighbour, matches ablation convention

# ── Offset extraction (replicates each mode's internal alignment forward) ────

@torch.no_grad()
def dcn_offset(model, spad):
    """dcn_h2 / dcn_h4 / dcn_h8 / dcn_h16 — PairwiseDCNAlign."""
    mode = model.mode
    raw, f0, B, T, H, W = model._enc_batch(spad)
    if mode == 'dcn_h2':
        fl = list(f0.reshape(B, T, model.c1, H // 2, W // 2).unbind(1))
    elif mode == 'dcn_h4':
        h1, _ = model.dn1(f0)
        fl = list(h1.reshape(B, T, model.c2, H // 4, W // 4).unbind(1))
    elif mode == 'dcn_h8':
        h1, _ = model.dn1(f0); h2, _ = model.dn2(h1)
        fl = list(h2.reshape(B, T, model.c3, H // 8, W // 8).unbind(1))
    elif mode == 'dcn_h16':
        h1, _ = model.dn1(f0); h2, _ = model.dn2(h1); h3, _ = model.dn3(h2)
        fl = list(h3.reshape(B, T, model.c4, H // 16, W // 16).unbind(1))
    ctr, nbr = fl[model.C], fl[PREFER_T]
    al = model.align
    o = al.oc2(al.lr(al.oc1(torch.cat([nbr, ctr], 1))))
    off = o[:, :144]   # dg=8, K=9 -> GK=72 -> GK*2=144
    return (off[:, 0::2].mean(1)[0].float().cpu().numpy(),
            off[:, 1::2].mean(1)[0].float().cpu().numpy())


@torch.no_grad()
def spynet_offset(model, spad):
    """spynet_dcn — FlowGuidedDCNAlign with SpyNet flow prior."""
    raw, f0, B, T, H, W = model._enc_batch(spad)
    raw4 = raw.reshape(B, T, 4, H // 2, W // 2)
    fl   = list(f0.reshape(B, T, model.c1, H // 2, W // 2).unbind(1))
    dn   = [model.denoiser(raw4[:, t]).mean(1, keepdim=True).float()
            for t in range(T)]
    flow = model.spynet(dn[PREFER_T], dn[model.C]).clamp(-64., 64.)
    align = model.align
    ctr_f, nbr_f = fl[model.C], fl[PREFER_T]
    o   = align.oc2(align.lr(align.oc1(torch.cat([nbr_f, ctr_f], 1))))
    res = o[:, :align.GK * 2]
    off = align._prior(flow) + res
    flow_dy = flow[0, 1].float().cpu().numpy()
    flow_dx = flow[0, 0].float().cpu().numpy()
    off_dy  = off[:, 0::2].mean(1)[0].float().cpu().numpy()
    off_dx  = off[:, 1::2].mean(1)[0].float().cpu().numpy()
    return flow_dy, flow_dx, off_dy, off_dx


@torch.no_grad()
def oracle_offset(model, spad, clean_rgb):
    """oracle_flow — FlowGuidedDCNAlign with frozen-RAFT flow prior."""
    raw, f0, B, T, H, W = model._enc_batch(spad)
    fl = list(f0.reshape(B, T, model.c1, H // 2, W // 2).unbind(1))
    ctr_rgb, nbr_rgb = clean_rgb[:, model.C], clean_rgb[:, PREFER_T]
    with autocast('cuda', enabled=False):
        if model._raft_ok and model.raft is not None:
            tr = model.raft._raft_transforms
            c_in, n_in = tr(ctr_rgb.float(), nbr_rgb.float())
            c_in, n_in = c_in.to(spad.device), n_in.to(spad.device)
            flow_full = model.raft(c_in, n_in)[-1]
        else:
            flow_full = model.raft_fb(ctr_rgb, nbr_rgb)
    flow_h2 = F.avg_pool2d(flow_full.float(), 2, 2) * 0.5
    align = model.align
    ctr_f, nbr_f = fl[model.C], fl[PREFER_T]
    o   = align.oc2(align.lr(align.oc1(torch.cat([nbr_f, ctr_f], 1))))
    res = o[:, :align.GK * 2]
    off = align._prior(flow_h2) + res
    return (off[:, 0::2].mean(1)[0].float().cpu().numpy(),
            off[:, 1::2].mean(1)[0].float().cpu().numpy())


@torch.no_grad()
def cascade_offset(model, spad):
    """cascading_dcn — hierarchical CascadeLevel H/8 -> H/4 -> H/2.
    Returns the finest (H/2, cas1) offset, upsampled priors from H/8 and H/4."""
    B, T, H, W = spad.shape
    s3s, s2s, s1s, vs = [], [], [], []
    for t in range(T):
        raw, f1 = model._enc(spad[:, t])
        vs.append(model.rexp(raw))
        f2, s1 = model.dn1(f1)
        f3, s2 = model.dn2(f2)
        s3s.append(f3); s2s.append(s2); s1s.append(s1)

    def up(o):
        return F.interpolate(o.float(), scale_factor=2, mode='bilinear',
                             align_corners=False) * 2.

    C = model.C
    o3, _ = model.cas3(s3s[PREFER_T], s3s[C], prior=None)
    o2, _ = model.cas2(s2s[PREFER_T], s2s[C], prior=up(o3))
    o1, _ = model.cas1(s1s[PREFER_T], s1s[C], vn=vs[PREFER_T], prior=up(o2))
    return (o1[:, 0::2].mean(1)[0].float().cpu().numpy(),
            o1[:, 1::2].mean(1)[0].float().cpu().numpy())


# ── Viz helpers ───────────────────────────────────────────────────────────────

def hsv_flow(dy, dx, p99, out_hw):
    dy_r = cv2.resize(dy.astype(np.float32), (out_hw[1], out_hw[0]),
                      interpolation=cv2.INTER_LINEAR)
    dx_r = cv2.resize(dx.astype(np.float32), (out_hw[1], out_hw[0]),
                      interpolation=cv2.INTER_LINEAR)
    mag, ang = cv2.cartToPolar(dx_r, dy_r)
    mag_n = np.clip(mag / (p99 + 1e-6), 0, 1)
    hsv = np.zeros((*dy_r.shape, 3), np.uint8)
    hsv[..., 0] = np.uint8(ang * 90. / np.pi)
    hsv[..., 1] = 255
    hsv[..., 2] = np.uint8(mag_n * 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def label_img(img, txt):
    out = img.copy()
    cv2.putText(out, txt, (4, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def u8(arr):
    return np.repeat(np.uint8(np.clip(arr, 0, 1) * 255)[..., None], 3, -1)


def hstack(panels, sep_w=3):
    sep = np.full((panels[0].shape[0], sep_w, 3), 25, np.uint8)
    out = [panels[0]]
    for p_ in panels[1:]:
        out += [sep, p_]
    return np.concatenate(out, 1)


def color_wheel_legend(h, w, p99):
    """RGB panel: hue=direction, brightness=|offset| (0..p99). Same convention
    as optical_flow_ablation/viz_full.py::color_wheel_legend."""
    size = min(h, w) - 60
    ax = np.linspace(-1, 1, size, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    r = np.sqrt(xx**2 + yy**2)
    mag, ang = cv2.cartToPolar(xx * p99, yy * p99)
    mag_n = np.clip(mag / (p99 + 1e-6), 0, 1)
    hsv = np.zeros((size, size, 3), np.uint8)
    hsv[..., 0] = np.uint8(ang * 90. / np.pi)
    hsv[..., 1] = 255
    hsv[..., 2] = np.uint8(mag_n * 255)
    wheel = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    img = np.full((h, w, 3), 18, np.uint8)
    y0, x0 = (h - size) // 2, (w - size) // 2
    mask = r <= 1.0
    region = img[y0:y0 + size, x0:x0 + size]
    region[mask] = wheel[mask]
    img[y0:y0 + size, x0:x0 + size] = region

    cx, cy, rad = x0 + size // 2, y0 + size // 2, size // 2
    cv2.circle(img, (cx, cy), rad, (255, 255, 255), 1, cv2.LINE_AA)

    def put(txt, px, py):
        (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(img, txt, (px - tw // 2, py), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)

    put('RIGHT +x', cx + rad + 46, cy + 5)
    put('LEFT -x',  cx - rad - 46, cy + 5)
    put('DOWN +y',  cx, cy + rad + 22)
    put('UP -y',    cx, cy - rad - 10)
    cv2.putText(img, 'Offset direction legend', (4, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f'hue=direction  brightness=|offset| (0..p99={p99:.1f}px)',
                (4, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    return img


def _to_01(raw, p99):
    return torch.clamp(raw / (p99 + 1e-8), 0, 1).float()


def _gam(x):
    return x ** (1. / GAMMA)


# ── Dataset ───────────────────────────────────────────────────────────────────
samples = build_first_window(args.test_root)
ds = SPADDataset(samples, ppp=3.25, bins=7, augment=False, crop=256,
                 eval_crop=0, seed=42, random_pick=False, sps=1,
                 sensor_mode='mono', target_mode='luma')
loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
print(f"\nLoaded {len(samples)} test scenes → {VIZ_OUT}/\n")

# ── Main loop ─────────────────────────────────────────────────────────────────
for bi, batch in enumerate(tqdm(loader, desc='Viz+Offsets')):
    spad  = batch['spad_mono_seq'].to(DEVICE)
    rgb   = batch['clean_rgb_seq'].to(DEVICE)
    gt_s  = batch['target_s'].unsqueeze(1).to(DEVICE)
    pct99 = batch['pct99'].to(DEVICE)
    scene = batch.get('scene', [f'sample_{bi:04d}'])[0]

    p99   = pct99[0].item()
    gt_np = _gam(_to_01(gt_s, p99))[0, 0].cpu().numpy()
    H, W  = gt_np.shape

    preds_np, off_data = {}, {}

    for stem, _ in RUNS:
        if stem not in nets:
            continue
        model = nets[stem]
        mode  = modes[stem]

        with torch.no_grad(), autocast('cuda'):
            if mode == 'quiver':
                spad_5d = spad.unsqueeze(2).half()
                with torch.cuda.amp.autocast():
                    _, out1, _, _ = model(spad_5d)
                pred = out1[:, 0]                      # (B,1,H,W) pct99-domain
            elif mode == 'oracle_flow':
                pred, _ = model(spad, clean_rgb=rgb)
            elif mode == 'single_frame':
                pred, _ = model(spad[:, HALF_WIN:HALF_WIN + 1])
            else:
                pred, _ = model(spad)

        pred_np = _gam(_to_01(pred.float(), p99))[0, 0].cpu().numpy()
        preds_np[stem] = pred_np

        if stem not in OFFSET_STEMS:
            continue
        with torch.no_grad():
            if mode in ('dcn_h2', 'dcn_h4', 'dcn_h8', 'dcn_h16'):
                off_data[stem] = dcn_offset(model, spad)
            elif mode == 'spynet_dcn':
                fdy, fdx, ody, odx = spynet_offset(model, spad)
                off_data[stem] = (ody, odx)
                off_data['_spynet_flow'] = (fdy, fdx)
            elif mode == 'oracle_flow':
                off_data[stem] = oracle_offset(model, spad, rgb)
            elif mode == 'cascading_dcn':
                off_data[stem] = cascade_offset(model, spad)

    # ── Unified offset p99 across all sources ────────────────────────────────
    mags = [np.sqrt(dy**2 + dx**2) for k, (dy, dx) in off_data.items()
            if not k.startswith('_')]
    p99_off = float(np.percentile(
        np.concatenate([m.ravel() for m in mags]), 99)) if mags else 1.
    p99_flow = float(np.percentile(
        np.sqrt(off_data['_spynet_flow'][0]**2 + off_data['_spynet_flow'][1]**2).ravel(), 99)
        ) if '_spynet_flow' in off_data else 1.

    hw = (H, W)
    blank = np.zeros((H, W, 3), np.uint8)

    def pred_col(stem, short):
        img = u8(preds_np.get(stem, np.zeros((H, W))))
        return label_img(img, short)

    def off_col(stem, tag, p99v=None):
        if stem not in off_data:
            return blank
        dy, dx = off_data[stem]
        return label_img(hsv_flow(dy, dx, p99v or p99_off, hw), tag)

    spad_c = spad[0, HALF_WIN].float().cpu().numpy()
    legend_img = color_wheel_legend(H, W, p99_off)

    # ── Individual full-res diagrams ─────────────────────────────────────────
    scene_dir = os.path.join(IND_OUT, f"{scene.replace(os.sep, '_')}_s{bi:04d}")
    os.makedirs(scene_dir, exist_ok=True)
    for stem, short in RUNS:
        if stem not in OFFSET_STEMS or stem not in off_data:
            continue
        diagram = off_col(stem, f'{short}  p99={p99_off:.1f}px')
        cv2.imwrite(os.path.join(scene_dir, f'{short}_offset.png'),
                    cv2.cvtColor(diagram, cv2.COLOR_RGB2BGR))
    if '_spynet_flow' in off_data:
        fdy, fdx = off_data['_spynet_flow']
        flow_diagram = label_img(hsv_flow(fdy, fdx, p99_flow, hw),
                                 f'SpyNet raw flow  p99={p99_flow:.1f}px')
        cv2.imwrite(os.path.join(scene_dir, 'SpyNet_rawflow.png'),
                    cv2.cvtColor(flow_diagram, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(scene_dir, 'direction_legend.png'),
                cv2.cvtColor(legend_img, cv2.COLOR_RGB2BGR))

    # ── Combined panel ────────────────────────────────────────────────────────
    row1 = hstack(
        [label_img(u8(spad_c), 'SPAD')]
        + [pred_col(stem, short) for stem, short in RUNS]
        + [label_img(u8(gt_np), 'GT')]
    )
    row2_cols = [blank]   # under SPAD
    for stem, short in RUNS:
        if stem in OFFSET_STEMS:
            row2_cols.append(off_col(stem, short))
        else:
            row2_cols.append(blank)
    row2_cols.append(legend_img)   # under GT
    row2 = hstack(row2_cols)

    sep   = np.full((4, row1.shape[1], 3), 25, np.uint8)
    panel = np.concatenate([row1, sep, row2], 0)
    fname = f"{scene.replace(os.sep, '_')}_s{bi:04d}.png"
    cv2.imwrite(os.path.join(VIZ_OUT, fname), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

print(f"\nDone. {len(samples)} panels → {VIZ_OUT}/")
print(f"      {len(samples)} per-model diagram sets → {IND_OUT}/")
