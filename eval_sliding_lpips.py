#!/usr/bin/env python3
"""
Full sliding-window eval with PSNR + SSIM + LPIPS on the i2k test set.
Usage:
    python3 eval_sliding_lpips.py --ckpt runs/color_a4_guided_i2k/latest.pth \
        --stage1 checkpoints/dcn_h8/best.pth --out results_a4_sliding.csv
"""
import argparse, csv, math, os, sys
import cv2
import numpy as np
import torch
import lpips as lpips_lib
from skimage.metrics import structural_similarity

os.chdir('/home/samuel/spad-net-color')
sys.path.insert(0, '/home/samuel/spad-net-color')

from src.data.simulation import scene_stats, simulate_spad, simulate_cmos
from src.utils import load_checkpoint
from train_color import ColorUNet, load_stage1, get_luma, GAMMA

SEQ_LEN    = 11
HALF_WIN   = SEQ_LEN // 2
BINS       = 7
CMOS_SIGMA = 2.0
TEST_ROOT  = '/home/samuel/dataset/i2-2kfps_v1_png/test'


def psnr_np(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 100. if mse == 0 else 10 * math.log10(1. / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt',    required=True)
    ap.add_argument('--stage1',  required=True)
    ap.add_argument('--ppp',     type=float, default=0.)
    ap.add_argument('--cmos_t',  type=int,   default=11)
    ap.add_argument('--out',     default='')
    ap.add_argument('--device',  default='cuda:2')
    args = ap.parse_args()

    device = torch.device(args.device)

    ck   = load_checkpoint(args.ckpt, device)
    ppp  = args.ppp if args.ppp > 0 else float(ck.get('ppp', 3.25))
    mode = ck.get('mode', 'cmos_luma_softmax')
    ep   = ck.get('epoch', '?')
    model = ColorUNet(mode=mode, bc=ck.get('bc', 32)).to(device)
    model.load_state_dict(ck['model']); model.eval()
    stage1 = load_stage1(args.stage1, device)

    # try to infer ppp from training log if not in checkpoint
    log_path = os.path.join(os.path.dirname(args.ckpt), 'train.log')
    if ppp == 3.25 and os.path.isfile(log_path):
        for line in open(log_path):
            if 'ppp=' in line:
                import re
                m = re.search(r'ppp=([\d.]+)', line)
                if m: ppp = float(m.group(1)); break

    out_path = args.out or os.path.join(
        os.path.dirname(args.ckpt),
        f'eval_i2k_sliding_lpips_ep{ep}.csv')

    print(f'Model : {args.ckpt}  ep={ep}  mode={mode}  ppp={ppp}')
    print(f'Stage1: {args.stage1}')
    print(f'Output: {out_path}\n')

    lpips_fn = lpips_lib.LPIPS(net='alex').to(device); lpips_fn.eval()

    scenes = sorted(d for d in os.listdir(TEST_ROOT)
                    if os.path.isdir(os.path.join(TEST_ROOT, d)))

    rows = []
    scene_psnr, scene_ssim, scene_lpips = [], [], []

    for si, scene_name in enumerate(scenes):
        scene_dir = os.path.join(TEST_ROOT, scene_name)
        pngs = sorted(f for f in os.listdir(scene_dir) if f.endswith('.png'))
        frames = np.stack([
            cv2.cvtColor(cv2.imread(os.path.join(scene_dir, p)),
                         cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
            for p in pngs
        ], 0)   # (N, H, W, 3)
        N = len(frames)

        win_psnr, win_ssim, win_lpips = [], [], []
        for w in range(N - SEQ_LEN + 1):
            seq = frames[w:w + SEQ_LEN]   # (11, H, W, 3)
            gt  = seq[HALF_WIN]            # (H, W, 3)

            x_lin, alpha, pct99 = scene_stats(seq, ppp, HALF_WIN)
            sim   = simulate_spad(x_lin, alpha, BINS, HALF_WIN, pct99, sensor_mode='mono')
            cmos_ = simulate_cmos(x_lin, alpha, CMOS_SIGMA, cmos_t=args.cmos_t)

            spad_t = torch.from_numpy(sim['spad_mono_seq']).unsqueeze(0).to(device)
            cmos_t = torch.from_numpy(cmos_['cmos_packed']).unsqueeze(0).to(device)
            batch  = {'spad_mono_seq': spad_t,
                      'target_s': torch.from_numpy(sim['target_s']).unsqueeze(0).to(device)}

            a = max(float(alpha), 1e-8)
            with torch.no_grad():
                luma = get_luma(batch, stage1, device)
                pred = model(cmos_t, luma)

            pred_srgb = (pred[0].float().cpu() / a).clamp(0,1).pow(1./GAMMA
                         ).permute(1,2,0).numpy()

            p = psnr_np(pred_srgb, gt)
            s = float(structural_similarity(pred_srgb, gt, data_range=1., channel_axis=2))
            with torch.no_grad():
                pt = torch.from_numpy(pred_srgb).permute(2,0,1).unsqueeze(0).float()*2-1
                gt_t = torch.from_numpy(gt).permute(2,0,1).unsqueeze(0).float()*2-1
                l = float(lpips_fn(pt.to(device), gt_t.to(device)).item())

            win_psnr.append(p); win_ssim.append(s); win_lpips.append(l)
            rows.append([scene_name, w, f'{p:.4f}', f'{s:.4f}', f'{l:.4f}'])

        sp = float(np.mean(win_psnr)); ss = float(np.mean(win_ssim)); sl = float(np.mean(win_lpips))
        scene_psnr.append(sp); scene_ssim.append(ss); scene_lpips.append(sl)
        print(f'[{si+1:2d}/{len(scenes)}] {scene_name}  {N-SEQ_LEN+1} windows  '
              f'PSNR={sp:.3f}  SSIM={ss:.4f}  LPIPS={sl:.4f}')
        sys.stdout.flush()

    mean_p = float(np.mean(scene_psnr))
    mean_s = float(np.mean(scene_ssim))
    mean_l = float(np.mean(scene_lpips))
    print(f'\n{"─"*60}')
    print(f'MEAN  PSNR={mean_p:.4f}  SSIM={mean_s:.4f}  LPIPS={mean_l:.4f}')
    print(f'{"─"*60}')

    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['scene', 'window', 'psnr', 'ssim', 'lpips'])
        w.writerows(rows)
        w.writerow(['MEAN', '', f'{mean_p:.4f}', f'{mean_s:.4f}', f'{mean_l:.4f}'])
    print(f'Saved → {out_path}')


if __name__ == '__main__':
    main()
