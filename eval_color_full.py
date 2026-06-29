#!/usr/bin/env python3
"""Full i2k test-set evaluation for color ablation models A1/A2/A3.

Evaluates all three models on all test scenes at native 512x1024 resolution
using all non-overlapping 11-frame windows.
"""
import csv, math, os, sys
import numpy as np
import torch

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

TEST_ROOT = '/home/samuel/dataset/i2-2kfps_v1_png/test'
CKPTS = [
    ('A1', 'runs/color_a1_i2k/best.pth'),
    ('A2', 'runs/color_a2_i2k/best.pth'),
    ('A3', 'runs/color_a3_i2k/best.pth'),
]
STAGE1_CKPT = 'checkpoints/dcn_h8/best.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_color_model(path, device):
    ck   = load_checkpoint(path, device)
    mode = ck.get('mode', ck.get('color_mode', 'cmos_only'))
    bc   = ck.get('bc', 32)
    m    = ColorUNet(mode=mode, bc=bc).to(device)
    m.load_state_dict(ck['model'])
    m.eval()
    return m


def load_frames(scene_dir):
    pngs = sorted(f for f in os.listdir(scene_dir) if f.endswith('.png'))
    frames = []
    import cv2
    for f in pngs:
        img = cv2.imread(os.path.join(scene_dir, f), cv2.IMREAD_COLOR)
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.)
    return np.stack(frames, 0)


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 100.0 if mse == 0 else 10 * math.log10(1.0 / mse)


def ssim_val(a, b):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(a, b, data_range=1.0, channel_axis=2))


def lpips_val(a, b, lpips_fn, device):
    def _t(x):
        return torch.from_numpy(x).permute(2,0,1).unsqueeze(0).float().to(device) * 2.0 - 1.0
    with torch.no_grad():
        return float(lpips_fn(_t(a), _t(b)).squeeze())


def pred_to_srgb(pred_raw, alpha):
    p = torch.from_numpy(pred_raw).float()
    return (p / max(alpha, 1e-8)).clamp(0, 1).pow(1.0 / GAMMA).permute(1, 2, 0).numpy()


# ── load models ───────────────────────────────────────────────────────────────

print('Loading models...')
models = {}
for tag, path in CKPTS:
    models[tag] = load_color_model(path, device)
    ep = torch.load(path, map_location='cpu', weights_only=True).get('epoch', '?')
    print(f'  {tag}: {path}  (epoch={ep}, mode={models[tag].mode})')

stage1  = load_stage1(STAGE1_CKPT, device)
lpips_fn = lpips_lib.LPIPS(net='alex').to(device).eval()

# ── test scenes ───────────────────────────────────────────────────────────────

scenes = sorted(d for d in os.listdir(TEST_ROOT)
                if os.path.isdir(os.path.join(TEST_ROOT, d)))
print(f'\n{len(scenes)} test scenes  |  device={device}\n')

all_results = {tag: [] for tag in models}

for si, scene_name in enumerate(scenes):
    scene_dir = os.path.join(TEST_ROOT, scene_name)
    frames    = load_frames(scene_dir)
    T         = frames.shape[0]
    # sliding window step=1: every frame that can be a centre frame
    windows = [frames[s:s + SEQ_LEN] for s in range(T - SEQ_LEN + 1)]
    n_win   = len(windows)

    scene_acc = {tag: {'psnr': [], 'ssim': [], 'lpips': []} for tag in models}

    for wi, seq_win in enumerate(windows):
        x_lin, alpha, pct99 = scene_stats(seq_win, PPP, HALF_WIN)
        gt_srgb = seq_win[HALF_WIN]   # (H, W, 3)

        spad_sim = simulate_spad(x_lin, alpha, BINS, HALF_WIN, pct99)
        cmos_sim = simulate_cmos(x_lin, alpha, CMOS_SIGMA)

        spad = torch.from_numpy(spad_sim['spad_mono_seq']).unsqueeze(0).to(device)
        cmos = torch.from_numpy(cmos_sim['cmos_packed']).unsqueeze(0).to(device)
        batch = {
            'spad_mono_seq': spad,
            'target_s': torch.from_numpy(spad_sim['target_s']).unsqueeze(0).to(device),
        }

        with torch.no_grad():
            for tag, model in models.items():
                luma     = get_luma(batch, stage1, device) if model.mode != 'cmos_only' else None
                pred_raw = model(cmos, luma)[0].float().cpu().numpy()
                pred_srgb = pred_to_srgb(pred_raw, alpha)

                p = psnr(pred_srgb, gt_srgb)
                s = ssim_val(pred_srgb, gt_srgb)
                l = lpips_val(pred_srgb, gt_srgb, lpips_fn, device)
                scene_acc[tag]['psnr'].append(p)
                scene_acc[tag]['ssim'].append(s)
                scene_acc[tag]['lpips'].append(l)

    print(f'[{si+1:2d}/{len(scenes)}] {scene_name}  ({n_win} win)')
    for tag in models:
        mp = float(np.mean(scene_acc[tag]['psnr']))
        ms = float(np.mean(scene_acc[tag]['ssim']))
        ml = float(np.mean(scene_acc[tag]['lpips']))
        print(f'         {tag}: PSNR={mp:.3f}  SSIM={ms:.4f}  LPIPS={ml:.4f}')
        all_results[tag].append({'scene': scene_name, 'psnr': mp, 'ssim': ms, 'lpips': ml})

    sys.stdout.flush()

# ── aggregate ─────────────────────────────────────────────────────────────────

print('\n' + '='*68)
print(f'  MEAN over {len(scenes)} scenes — native 512x1024, all non-overlapping windows')
print(f'  {"Model":<6}  {"mode":<22}  {"PSNR":>8}  {"SSIM":>8}  {"LPIPS":>8}')
print(f'  {"-"*58}')
for tag, model in models.items():
    ps = [r['psnr']  for r in all_results[tag]]
    ss = [r['ssim']  for r in all_results[tag]]
    ls = [r['lpips'] for r in all_results[tag]]
    print(f'  {tag:<6}  {model.mode:<22}  '
          f'{np.mean(ps):>8.4f}  {np.mean(ss):>8.4f}  {np.mean(ls):>8.4f}')
print('='*68)

# ── save CSVs ─────────────────────────────────────────────────────────────────

for tag in models:
    csv_path = f'runs/color_{tag.lower()}_i2k/eval_i2k_full.csv'
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['scene', 'psnr', 'ssim', 'lpips'])
        w.writeheader()
        w.writerows(all_results[tag])
        agg_row = {k: float(np.mean([r[k] for r in all_results[tag]])) for k in ['psnr','ssim','lpips']}
        agg_row['scene'] = 'MEAN'
        w.writerow(agg_row)
    print(f'  Saved {csv_path}')
