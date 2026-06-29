#!/usr/bin/env python3
"""X4K evaluation for color ablation models A1/A2/A3.

4096x2160 frames are processed via 1024x1024 tiles with midpoint-boundary stitching.
Scene-level alpha and pct99 are computed from the full frame (not per tile).
Metrics (PSNR/SSIM/LPIPS) are computed on the full stitched sRGB canvas.
"""
import csv, math, os, sys
import cv2; cv2.setNumThreads(0)
import numpy as np
import torch
from skimage.metrics import structural_similarity

os.chdir('/home/samuel/spad-net-color')
sys.path.insert(0, '/home/samuel/spad-net-color')

from src.data.simulation import scene_stats, simulate_spad, simulate_cmos
from train_color import ColorUNet, load_stage1, get_luma
from src.utils import load_checkpoint
import lpips as lpips_lib

SEQ_LEN    = 11
HALF_WIN   = SEQ_LEN // 2
PPP        = 3.25
BINS       = 7
CMOS_SIGMA = 2.0
GAMMA      = 2.2
TILE_SIZE  = 1024

X4K_ROOT    = '/data/X4K1000FPS/test'
STAGE1_CKPT = 'checkpoints/dcn_h8/best.pth'
CKPTS = [
    ('A1', 'runs/color_a1_i2k/best.pth'),
    ('A2', 'runs/color_a2_i2k/best.pth'),
    ('A3', 'runs/color_a3_i2k/best.pth'),
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_color_model(path):
    ck   = load_checkpoint(path, device)
    mode = ck.get('mode', ck.get('color_mode', 'cmos_only'))
    bc   = ck.get('bc', 32)
    m    = ColorUNet(mode=mode, bc=bc).to(device)
    m.load_state_dict(ck['model'])
    m.eval()
    return m


def make_tile_grid(H, W, ts):
    """Returns list of tile descriptors for midpoint-boundary stitching."""
    n_rows = math.ceil(H / ts)
    n_cols = math.ceil(W / ts)
    ys = (np.round(np.linspace(0, H - ts, n_rows)).astype(int) & ~1).tolist()
    xs = (np.round(np.linspace(0, W - ts, n_cols)).astype(int) & ~1).tolist()
    y_bounds = ([0]
                + [(ys[i] + ts + ys[i + 1]) // 2 for i in range(len(ys) - 1)]
                + [H])
    x_bounds = ([0]
                + [(xs[i] + ts + xs[i + 1]) // 2 for i in range(len(xs) - 1)]
                + [W])
    tiles = []
    for ri, y0 in enumerate(ys):
        for ci, x0 in enumerate(xs):
            cy0, cy1 = y_bounds[ri], y_bounds[ri + 1]
            cx0, cx1 = x_bounds[ci], x_bounds[ci + 1]
            # (tile_origin_y, tile_origin_x, canvas_dst, tile_src)
            tiles.append((y0, x0, cy0, cy1, cx0, cx1,
                          cy0 - y0, cy1 - y0, cx0 - x0, cx1 - x0))
    return tiles


def load_frames(scene_dir):
    pngs = sorted(f for f in os.listdir(scene_dir) if f.endswith('.png'))
    frames = []
    for f in pngs:
        img = cv2.imread(os.path.join(scene_dir, f), cv2.IMREAD_COLOR)
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.)
    return np.stack(frames, 0)   # (T, H, W, 3)


def compute_psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 100.0 if mse == 0 else 10 * math.log10(1.0 / mse)


def compute_ssim(a_hwc, b_hwc):
    return float(structural_similarity(a_hwc, b_hwc, data_range=1.0, channel_axis=2))


def compute_lpips(a_hwc, b_hwc, lpips_fn):
    def _t(x):
        return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().to(device) * 2 - 1
    with torch.no_grad():
        return float(lpips_fn(_t(a_hwc), _t(b_hwc)).item())


# ── Load models ───────────────────────────────────────────────────────────────

print('Loading models...')
models = {}
for tag, path in CKPTS:
    models[tag] = load_color_model(path)
    ep = torch.load(path, map_location='cpu', weights_only=True).get('epoch', '?')
    print(f'  {tag}: epoch={ep}  mode={models[tag].mode}')

stage1   = load_stage1(STAGE1_CKPT, device)
lpips_fn = lpips_lib.LPIPS(net='alex').to(device).eval()

# ── Test scenes ───────────────────────────────────────────────────────────────

scenes = sorted(d for d in os.listdir(X4K_ROOT)
                if os.path.isdir(os.path.join(X4K_ROOT, d)))
print(f'\n{len(scenes)} X4K test scenes | device={device}\n')

all_results = {tag: [] for tag in models}

for si, scene_name in enumerate(scenes):
    scene_dir = os.path.join(X4K_ROOT, scene_name)
    frames    = load_frames(scene_dir)
    T, H, W, _ = frames.shape
    tiles     = make_tile_grid(H, W, TILE_SIZE)
    windows   = [frames[s:s + SEQ_LEN] for s in range(T - SEQ_LEN + 1)]
    n_win     = len(windows)

    print(f'[{si+1:2d}/{len(scenes)}] {scene_name}  '
          f'T={T}  {H}x{W}  {len(tiles)} tiles  {n_win} windows')

    scene_acc = {tag: {'psnr': [], 'ssim': [], 'lpips': []} for tag in models}

    for wi, seq_win in enumerate(windows):
        # Global scene stats from full-resolution 11-frame window
        x_lin_full, alpha, pct99 = scene_stats(seq_win, PPP, HALF_WIN)
        gt_srgb = seq_win[HALF_WIN]   # (H, W, 3) sRGB [0,1]

        # Stitch canvas in alpha*linear space (3-channel RGB)
        pred_canvas = {tag: np.zeros((3, H, W), np.float32) for tag in models}

        for (y0, x0, cy0, cy1, cx0, cx1, ty0, ty1, tx0, tx1) in tiles:
            x_lin_tile = x_lin_full[:, y0:y0 + TILE_SIZE, x0:x0 + TILE_SIZE, :]

            spad_sim = simulate_spad(x_lin_tile, alpha, BINS, HALF_WIN, pct99)
            cmos_sim = simulate_cmos(x_lin_tile, alpha, CMOS_SIGMA)

            spad = torch.from_numpy(spad_sim['spad_mono_seq']).unsqueeze(0).to(device)
            cmos = torch.from_numpy(cmos_sim['cmos_packed']).unsqueeze(0).to(device)
            batch = {
                'spad_mono_seq': spad,
                'target_s': torch.from_numpy(spad_sim['target_s']).unsqueeze(0).to(device),
            }

            with torch.no_grad():
                for tag, model in models.items():
                    luma = get_luma(batch, stage1, device) if model.mode != 'cmos_only' else None
                    pred_tile = model(cmos, luma)[0].float().cpu().numpy()  # (3, ts, ts)
                    pred_canvas[tag][:, cy0:cy1, cx0:cx1] = pred_tile[:, ty0:ty1, tx0:tx1]

        # Convert to sRGB and compute metrics on full stitched 4K canvas
        for tag in models:
            pred_srgb = (torch.from_numpy(pred_canvas[tag]).float()
                         / max(float(alpha), 1e-8)).clamp(0, 1).pow(1.0 / GAMMA)
            pred_srgb = pred_srgb.permute(1, 2, 0).numpy()   # (H, W, 3)

            p = compute_psnr(pred_srgb, gt_srgb)
            s = compute_ssim(pred_srgb, gt_srgb)
            l = compute_lpips(pred_srgb, gt_srgb, lpips_fn)
            scene_acc[tag]['psnr'].append(p)
            scene_acc[tag]['ssim'].append(s)
            scene_acc[tag]['lpips'].append(l)

        if (wi + 1) % 5 == 0 or wi == n_win - 1:
            print(f'  win {wi+1}/{n_win}  '
                  + '  '.join(f'{t}: {np.mean(scene_acc[t]["psnr"]):.2f}dB'
                               for t in models))
            sys.stdout.flush()

    for tag in models:
        mp = float(np.mean(scene_acc[tag]['psnr']))
        ms = float(np.mean(scene_acc[tag]['ssim']))
        ml = float(np.mean(scene_acc[tag]['lpips']))
        print(f'    {tag}: PSNR={mp:.3f}  SSIM={ms:.4f}  LPIPS={ml:.4f}')
        all_results[tag].append({'scene': scene_name, 'psnr': mp, 'ssim': ms, 'lpips': ml})
    sys.stdout.flush()

# ── Aggregate ─────────────────────────────────────────────────────────────────

print('\n' + '='*68)
print(f'  X4K MEAN over {len(scenes)} scenes — {TILE_SIZE}px tiles, sliding window')
print(f'  {"Model":<6}  {"mode":<22}  {"PSNR":>8}  {"SSIM":>8}  {"LPIPS":>8}')
print(f'  {"-"*58}')
for tag, model in models.items():
    ps = [r['psnr']  for r in all_results[tag]]
    ss = [r['ssim']  for r in all_results[tag]]
    ls = [r['lpips'] for r in all_results[tag]]
    print(f'  {tag:<6}  {model.mode:<22}  '
          f'{np.mean(ps):>8.4f}  {np.mean(ss):>8.4f}  {np.mean(ls):>8.4f}')
print('='*68)

# ── Save CSVs ─────────────────────────────────────────────────────────────────

for tag in models:
    csv_path = f'runs/color_{tag.lower()}_i2k/eval_x4k.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['scene', 'psnr', 'ssim', 'lpips'])
        w.writeheader()
        w.writerows(all_results[tag])
        agg = {k: float(np.mean([r[k] for r in all_results[tag]]))
               for k in ['psnr', 'ssim', 'lpips']}
        agg['scene'] = 'MEAN'
        w.writerow(agg)
    print(f'  Saved {csv_path}')
