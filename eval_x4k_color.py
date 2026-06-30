#!/usr/bin/env python3
"""X4K zero-shot evaluation for a DUALQUANTA colour model.

4096×2160 frames are processed via 1024×1024 tiles with midpoint-boundary
stitching.  Scene-level alpha and pct99 are computed from the full frame
(not per tile).  Metrics (PSNR / SSIM / LPIPS) are computed on the full
stitched sRGB canvas.

Usage:
    python eval_x4k_color.py \
        --ckpt   checkpoints/table3_4_DUALQUANTA/DUALQUANTA_T11_stage2.pth \
        --stage1 checkpoints/table3_4_DUALQUANTA/stage1_mono_dcn_h8.pth \
        --test_root /path/to/X4K1000FPS/test
"""
import argparse, csv, math, os, sys
import cv2; cv2.setNumThreads(0)
import numpy as np
import torch
from skimage.metrics import structural_similarity

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_DIR)

from src.data.simulation import scene_stats, simulate_spad, simulate_cmos
from train_color import ColorUNet, load_stage1, get_luma, GAMMA
from src.utils import load_checkpoint
import lpips as lpips_lib

SEQ_LEN    = 11
HALF_WIN   = SEQ_LEN // 2
PPP        = 3.25
BINS       = 7
CMOS_SIGMA = 2.0
TILE_SIZE  = 1024


def make_tile_grid(H, W, ts):
    """Returns tile descriptors for midpoint-boundary stitching."""
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
            tiles.append((y0, x0, cy0, cy1, cx0, cx1,
                          cy0 - y0, cy1 - y0, cx0 - x0, cx1 - x0))
    return tiles


def load_frames(scene_dir):
    pngs = sorted(f for f in os.listdir(scene_dir) if f.endswith(".png"))
    frames = []
    for f in pngs:
        img = cv2.imread(os.path.join(scene_dir, f), cv2.IMREAD_COLOR)
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.)
    return np.stack(frames, 0)


def compute_psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 100.0 if mse == 0 else 10 * math.log10(1.0 / mse)


def compute_ssim(a, b):
    return float(structural_similarity(a, b, data_range=1.0, channel_axis=2))


def compute_lpips(a, b, lpips_fn, device):
    def _t(x):
        return torch.from_numpy(x).permute(2,0,1).unsqueeze(0).float().to(device) * 2 - 1
    with torch.no_grad():
        return float(lpips_fn(_t(a), _t(b)).item())


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--ckpt",      required=True,
                    help="Stage-2 checkpoint")
    ap.add_argument("--stage1",    required=True,
                    help="Stage-1 SPADNet checkpoint (stage1_mono_dcn_h8.pth)")
    ap.add_argument("--test_root", required=True,
                    help="Path to X4K1000FPS test directory")
    ap.add_argument("--cmos_t",    type=int,   default=11,
                    help="CMOS integration window length")
    ap.add_argument("--out",       default="",
                    help="Output CSV path (default: <ckpt_dir>/eval_x4k.csv)")
    ap.add_argument("--device",    default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device)

    ck   = load_checkpoint(args.ckpt, device)
    mode = ck.get("mode", "cmos_luma_softmax")
    ep   = ck.get("epoch", "?")
    model = ColorUNet(mode=mode, bc=ck.get("bc", 32)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    stage1   = load_stage1(args.stage1, device)
    lpips_fn = lpips_lib.LPIPS(net="alex").to(device).eval()

    out_path = args.out or os.path.join(
        os.path.dirname(args.ckpt),
        f"eval_x4k_ep{ep}.csv")

    print(f"Model : {args.ckpt}  ep={ep}  mode={mode}  cmos_t={args.cmos_t}")
    print(f"Stage1: {args.stage1}")
    print(f"Output: {out_path}\n")

    scenes = sorted(d for d in os.listdir(args.test_root)
                    if os.path.isdir(os.path.join(args.test_root, d)))
    print(f"{len(scenes)} X4K test scenes | device={device}\n")

    all_results = []
    scene_psnr, scene_ssim, scene_lpips = [], [], []

    for si, scene_name in enumerate(scenes):
        scene_dir = os.path.join(args.test_root, scene_name)
        frames    = load_frames(scene_dir)
        T, H, W, _ = frames.shape
        tiles     = make_tile_grid(H, W, TILE_SIZE)
        windows   = list(range(T - SEQ_LEN + 1))

        print(f"[{si+1:2d}/{len(scenes)}] {scene_name}  "
              f"T={T}  {H}×{W}  {len(tiles)} tiles  {len(windows)} win")

        win_psnr, win_ssim, win_lpips = [], [], []

        for w in windows:
            seq_win = frames[w:w + SEQ_LEN]
            x_lin_full, alpha, pct99 = scene_stats(seq_win, PPP, HALF_WIN)
            gt_srgb = seq_win[HALF_WIN]

            pred_canvas = np.zeros((3, H, W), np.float32)

            for (y0, x0, cy0, cy1, cx0, cx1, ty0, ty1, tx0, tx1) in tiles:
                x_lin_tile = x_lin_full[:, y0:y0+TILE_SIZE, x0:x0+TILE_SIZE, :]

                spad_sim = simulate_spad(x_lin_tile, alpha, BINS, HALF_WIN, pct99,
                                         sensor_mode="mono")
                cmos_sim = simulate_cmos(x_lin_tile, alpha, CMOS_SIGMA,
                                         cmos_t=args.cmos_t)

                spad_t  = torch.from_numpy(spad_sim["spad_mono_seq"]).unsqueeze(0).to(device)
                cmos_in = torch.from_numpy(cmos_sim["cmos_packed"]).unsqueeze(0).to(device)
                batch   = {"spad_mono_seq": spad_t,
                           "target_s": torch.from_numpy(spad_sim["target_s"]).unsqueeze(0).to(device)}

                with torch.no_grad():
                    luma = get_luma(batch, stage1, device) if mode != "cmos_only" else None
                    pred_tile = model(cmos_in, luma)[0].float().cpu().numpy()

                pred_canvas[:, cy0:cy1, cx0:cx1] = pred_tile[:, ty0:ty1, tx0:tx1]

            a = max(float(alpha), 1e-8)
            pred_srgb = (torch.from_numpy(pred_canvas).float() / a
                         ).clamp(0, 1).pow(1. / GAMMA).permute(1, 2, 0).numpy()

            p = compute_psnr(pred_srgb, gt_srgb)
            s = compute_ssim(pred_srgb, gt_srgb)
            l = compute_lpips(pred_srgb, gt_srgb, lpips_fn, device)
            win_psnr.append(p)
            win_ssim.append(s)
            win_lpips.append(l)

            if (w + 1) % 5 == 0 or w == windows[-1]:
                print(f"  win {w+1}/{len(windows)}  "
                      f"PSNR={np.mean(win_psnr):.2f}")
                sys.stdout.flush()

        sp = float(np.mean(win_psnr))
        ss = float(np.mean(win_ssim))
        sl = float(np.mean(win_lpips))
        scene_psnr.append(sp)
        scene_ssim.append(ss)
        scene_lpips.append(sl)
        print(f"  scene mean: PSNR={sp:.3f}  SSIM={ss:.4f}  LPIPS={sl:.4f}")
        all_results.append({"scene": scene_name, "psnr": sp, "ssim": ss, "lpips": sl})
        sys.stdout.flush()

    mean_p = float(np.mean(scene_psnr))
    mean_s = float(np.mean(scene_ssim))
    mean_l = float(np.mean(scene_lpips))
    print(f"\n{'='*60}")
    print(f"X4K MEAN over {len(scenes)} scenes — {TILE_SIZE}px tiles")
    print(f"PSNR={mean_p:.4f}  SSIM={mean_s:.4f}  LPIPS={mean_l:.4f}")
    print(f"{'='*60}")

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scene", "psnr", "ssim", "lpips"])
        w.writeheader()
        w.writerows(all_results)
        w.writerow({"scene": "MEAN", "psnr": mean_p, "ssim": mean_s, "lpips": mean_l})
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
