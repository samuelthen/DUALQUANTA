#!/usr/bin/env python3
"""
SPADNet evaluation script.

Standard evaluation (per-sample):
    python evaluate.py --ckpt runs/dcn_h4/best.pth \
                       --test_root /path/to/test \
                       --device cuda:0

Tiled evaluation (high-resolution X4K-style):
    python evaluate.py --ckpt runs/dcn_h4/best.pth \
                       --test_root /data/X4K1000FPS/test \
                       --tiled --tile_size 1024 \
                       --device cuda:0

Sliding-window evaluation (all windows, not just first):
    python evaluate.py --ckpt runs/dcn_h4/best.pth \
                       --test_root /path/to/test \
                       --sliding_window

Outputs:
    - Prints per-scene and aggregate metrics to stdout
    - Saves results to <ckpt_dir>/eval_results.csv
"""

import argparse
import math
import os
import sys

import cv2
cv2.setNumThreads(0)
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import lpips as lpips_lib
    LPIPS_OK = True
except ImportError:
    LPIPS_OK = False

from src import SPADNet, SPADDataset, TiledDataset, build_test
from src.data.dataset import build_splits, build_first_window, build_sliding, _worker_init, SEQ_LEN
HALF_WIN = SEQ_LEN // 2
from src.metrics import to_display, psnr, ssim, evaluate_batch
from src.utils import load_checkpoint

_TTY     = sys.stderr.isatty()
GAMMA    = 2.2


# ── argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Evaluate SPADNet")
    p.add_argument("--ckpt",      required=True,
                   help="Path to checkpoint (best.pth)")
    p.add_argument("--test_root", default="",
                   help="Root directory containing test sequences")
    p.add_argument("--device",    default="",
                   help="PyTorch device string")
    p.add_argument("--eval_crop", type=int, default=0,
                   help="Centre-crop size for evaluation (0 = full resolution)")
    p.add_argument("--workers",   type=int, default=4)
    p.add_argument("--ppp",       type=float, default=3.25)
    p.add_argument("--bins",      type=int,   default=7)
    p.add_argument("--tiled",     action="store_true",
                   help="Tiled evaluation for high-resolution sequences")
    p.add_argument("--tile_size", type=int, default=1024)
    p.add_argument("--sliding_window", action="store_true",
                   help="Evaluate all sliding windows (default: first window only)")
    p.add_argument("--out_csv",   default="",
                   help="Output CSV path (default: <ckpt_dir>/eval_results.csv)")
    p.add_argument("--no_lpips",  action="store_true",
                   help="Skip LPIPS computation")
    p.add_argument("--sensor_mode", default="",
                   choices=("", "mono", "rggb"),
                   help="Override sensor mode for dataset (default: read from checkpoint)")
    return p.parse_args()


# ── model loading ─────────────────────────────────────────────────────────────

_OLD_MODE_MAP = {
    "single_frame_no_grad": "single_frame",
    "single_frame":         "single_frame",
    "no_align":             "no_align",
    "dcn_d0":               "dcn_h2",
    "dcn_d1":               "dcn_h4",
    "dcn_d2":               "dcn_h8",
    "dcn_d3":               "dcn_h16",
    "spynet_dcn":           "spynet_dcn",
    "cascading_dcn":        "cascading_dcn",
    "oracle_flow":          "oracle_flow",
    "shift_net":            "shift_net",
}

_QUIVER_N_FEATURES = 64
_QUIVER_N_BLOCKS   = 12


def _is_quiver_ckpt(ck):
    # NOTE: "predenoise." is QUIVER-exclusive. Do NOT also match "spynet." —
    # SPADNet's own spynet_dcn mode has a `self.spynet` submodule too, which
    # previously false-positived spynet_dcn.pth into the QUIVER loader.
    state = ck.get("model", {})
    return any(k.startswith("predenoise.") for k in state)


def load_quiver_model(ckpt_path, device):
    import argparse as _ap
    from src.models.quiver_model import QUIVER
    ck = load_checkpoint(ckpt_path, device)
    para = _ap.Namespace(
        inp_ch=1, n_features=_QUIVER_N_FEATURES, n_blocks=_QUIVER_N_BLOCKS,
        past_frames=HALF_WIN, future_frames=HALF_WIN, activation="gelu",
        spynet_path="", load_spynet_weights=False,
    )
    model = QUIVER(para).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.device = device
    model.half()
    model.eval()
    model.mode        = "quiver"
    model.target_mode = "luma"
    model.sensor_mode = "mono"
    model.out_ch      = 1
    ep = ck.get("epoch", "?")
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded {ckpt_path}  (mode=quiver, epoch={ep}, params={n_params:.2f}M)")
    return model, "quiver", "mono"


def _config_for_ckpt(ckpt_path):
    """Bare checkpoints (table1_mono/*.pth) carry no 'mode'/'args' metadata —
    they're paired 1:1 with configs/mono/<same-name>.yaml. Returns that
    config's `model` section, or {} if no matching config exists."""
    stem = os.path.splitext(os.path.basename(ckpt_path))[0]
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "configs", "mono", f"{stem}.yaml")
    if not os.path.isfile(cfg_path):
        return {}
    from src.utils import load_config
    return load_config(cfg_path).get("model", {})


def load_model(ckpt_path, device):
    ck = load_checkpoint(ckpt_path, device)
    if _is_quiver_ckpt(ck):
        return load_quiver_model(ckpt_path, device)
    args_saved = ck.get("args", {})
    cfg_model  = {} if (ck.get("mode") or args_saved) else _config_for_ckpt(ckpt_path)
    # Support new (top-level "mode"), old (args["mode"]), and bare-checkpoint
    # (configs/mono/<name>.yaml) formats, in that priority order.
    raw_mode    = ck.get("mode") or args_saved.get("mode") or cfg_model.get("mode", "dcn_h4")
    mode        = _OLD_MODE_MAP.get(raw_mode, raw_mode)
    bc          = args_saved.get("base_c",    cfg_model.get("base_channels", 32))
    nb          = args_saved.get("n_blocks",  cfg_model.get("n_blocks",      2))
    nfpm        = args_saved.get("n_fpm",     cfg_model.get("n_fpm",         2))
    raft        = args_saved.get("raft_ckpt", cfg_model.get("raft_ckpt",    ""))
    out_ch      = ck.get("out_ch",      args_saved.get("out_ch", 1 if args_saved.get("target_mode", "luma") == "luma" else 3))
    target_mode = ck.get("target_mode", args_saved.get("target_mode", "luma"))
    sensor_mode = ck.get("sensor_mode", args_saved.get("sensor_mode", "mono"))
    T           = args_saved.get("T", cfg_model.get("T", SEQ_LEN))
    model = SPADNet(T=T, bc=bc, nb=nb, nfpm=nfpm, mode=mode, raft_ckpt=raft,
                    out_ch=out_ch, target_mode=target_mode)
    model.load_state_dict(ck["model"], strict=True)
    model.to(device).eval()
    print(f"  Loaded {ckpt_path}  (mode={mode}, sensor={sensor_mode}, "
          f"target={target_mode}, out_ch={out_ch}, "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M)")
    return model, mode, sensor_mode


# ── standard evaluation ───────────────────────────────────────────────────────

@torch.no_grad()
def eval_standard(model, loader, device, lpips_fn=None):
    """Per-sample evaluation. Returns list of per-sample result dicts.

    For both target_mode='rgb' and 'luma': alpha-normalised sRGB PSNR (publishable).
      luma: pred_lin = (pred / alpha / 3).clamp(0,1)  vs  gt = mean(RGB_lin)^(1/γ)
      rgb:  pred_lin = (pred / alpha).clamp(0,1)       vs  gt = GT sRGB (3-ch)
    QUIVER: pct99-normalised output → alpha-normalised conversion.
    """
    if model.mode == "quiver":
        return _eval_standard_quiver(model, loader, device, lpips_fn)
    return _eval_standard_srgb(model, loader, device, lpips_fn)


@torch.no_grad()
def _eval_standard_quiver(model, loader, device, lpips_fn=None):
    """Alpha-normalised sRGB PSNR for QUIVER.

    QUIVER outputs pct99-normalised predictions.  Convert to alpha-normalised:
      pred_lin   = pred_display ^ γ
      pred_srgb  = clip(pred_lin * pct99 / alpha / 3, 0, 1) ^ (1/γ)
      gt_srgb    = clip(mean(GT_RGB_lin), 0, 1) ^ (1/γ)
    """
    from skimage.metrics import structural_similarity
    import math as _math

    results = []

    for batch in tqdm(loader, desc="eval (QUIVER)", disable=not _TTY):
        spad      = batch["spad_mono_seq"].to(device)   # (B,T,H,W)
        alpha_b   = batch["alpha"]                       # (B,)
        pct99_b   = batch["pct99"]                       # (B,)
        gt_rgb_b  = batch["clean_rgb_seq"][:, HALF_WIN] # (B,3,H,W) sRGB
        scenes    = batch.get("scene", [""] * spad.shape[0])

        spad_5d = spad.unsqueeze(2).half()
        with torch.cuda.amp.autocast():
            _, out1, _, _ = model(spad_5d)               # (B,1,1,H,W)

        for b in range(out1.shape[0]):
            a     = alpha_b[b].item()
            p99   = pct99_b[b].item()
            pred_display = out1[b, 0, 0].float().cpu().numpy()  # (H,W) [0,1]
            pred_lin     = pred_display ** GAMMA
            pred_srgb    = np.clip(pred_lin * p99 / max(a, 1e-8) / 3.0, 0, 1) ** (1.0 / GAMMA)

            gt_srgb_hw = gt_rgb_b[b].float().cpu().numpy()       # (3,H,W)
            gt_lin_mean = (gt_srgb_hw ** GAMMA).mean(0)           # (H,W)
            gt_srgb     = np.clip(gt_lin_mean, 0, 1) ** (1.0 / GAMMA)

            p = psnr(torch.from_numpy(pred_srgb), torch.from_numpy(gt_srgb))
            s = float(structural_similarity(pred_srgb, gt_srgb, data_range=1.0))

            lpips_val = float('nan')
            if lpips_fn is not None:
                p_lp = torch.from_numpy(pred_srgb).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).float() * 2 - 1
                g_lp = torch.from_numpy(gt_srgb).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).float() * 2 - 1
                with torch.no_grad():
                    lpips_val = lpips_fn(p_lp.to(device), g_lp.to(device)).item()

            results.append(dict(scene=scenes[b], psnr_lin=float('nan'),
                                psnr_gam=p, ssim_gam=s, lpips_gam=lpips_val))

    return results


@torch.no_grad()
def _eval_standard_srgb(model, loader, device, lpips_fn=None):
    """Alpha-normalised sRGB PSNR/SSIM/LPIPS — publishable metric for all models.

    luma  (out_ch=1): pred_lin = (pred[:,0] / alpha / 3).clamp(0,1)
                      gt_srgb  = mean(GT_RGB_lin, dim=0)^(1/γ)  — grayscale
    rgb   (out_ch=3): pred_lin = (pred / alpha).clamp(0,1)
                      gt_srgb  = GT sRGB centre frame            — 3-channel
    """
    from skimage.metrics import structural_similarity
    import math as _math

    is_luma = (model.target_mode == "luma")
    needs_rgb = (model.mode == "oracle_flow")
    results  = []

    for batch in tqdm(loader, desc="eval (sRGB)", disable=not _TTY):
        spad   = batch["spad_mono_seq"].to(device)
        alpha  = batch["alpha"].to(device)               # (B,)
        gt_rgb_b = batch["clean_rgb_seq"][:, HALF_WIN]   # (B,3,H,W) sRGB [0,1]
        scenes = batch.get("scene", [""] * spad.shape[0])
        rgb    = batch["clean_rgb_seq"].to(device) if needs_rgb else None

        with autocast("cuda"):
            pred, _ = model(spad, clean_rgb=rgb)

        for b in range(pred.shape[0]):
            a = alpha[b].item()
            gt_srgb_b = gt_rgb_b[b].float().cpu()        # (3,H,W) sRGB [0,1]

            if is_luma:
                # pred = α·(R+G+B); divide by α·3 → mean(RGB) in linear space
                pred_lin  = (pred[b, 0].float() / max(a, 1e-8) / 3.0).clamp(0, 1).cpu()
                pred_srgb = pred_lin.unsqueeze(0) ** (1.0 / GAMMA)    # (1,H,W)
                gt_lin    = (gt_srgb_b ** GAMMA).mean(0, keepdim=True) # (1,H,W) linear
                gt_srgb   = gt_lin.clamp(0, 1) ** (1.0 / GAMMA)        # (1,H,W)
            else:
                pred_lin  = (pred[b].float() / max(a, 1e-8)).clamp(0, 1).cpu()  # (3,H,W)
                pred_srgb = pred_lin ** (1.0 / GAMMA)                            # (3,H,W)
                gt_srgb   = gt_srgb_b                                            # (3,H,W)

            mse = float(torch.mean((pred_srgb - gt_srgb) ** 2).item())
            psnr_val = 100.0 if mse == 0 else 10.0 * _math.log10(1.0 / mse)

            ssim_val = float(structural_similarity(
                pred_srgb.permute(1, 2, 0).numpy(),
                gt_srgb.permute(1, 2, 0).numpy(),
                data_range=1.0, channel_axis=2))

            lpips_val = float('nan')
            if lpips_fn is not None:
                if is_luma:
                    p_lp = pred_srgb.repeat(3, 1, 1).unsqueeze(0) * 2 - 1
                    g_lp = gt_srgb.repeat(3, 1, 1).unsqueeze(0)   * 2 - 1
                else:
                    p_lp = pred_srgb.unsqueeze(0) * 2 - 1
                    g_lp = gt_srgb.unsqueeze(0)   * 2 - 1
                with torch.no_grad():
                    lpips_val = lpips_fn(p_lp.to(device), g_lp.to(device)).item()

            results.append(dict(
                scene=scenes[b],
                psnr_lin=float('nan'),
                psnr_gam=psnr_val,
                ssim_gam=ssim_val,
                lpips_gam=lpips_val,
            ))

    return results


# ── tiled evaluation ──────────────────────────────────────────────────────────

@torch.no_grad()
def eval_tiled(model, dataset, loader, device, lpips_fn=None):
    """
    Tiled evaluation with midpoint-boundary stitching.

    luma (out_ch=1): pred_srgb = (pred[:,0] / alpha / 3).clamp(0,1)^(1/2.2) — grayscale
    rgb  (out_ch=3): pred_srgb = (pred / alpha).clamp(0,1)^(1/2.2)           — 3-channel
    LPIPS on full stitched frame; for 4K+ scenes LPIPS uses 1024×1024 centre crop.
    """
    H, W       = dataset.H, dataset.W
    n_tiles    = dataset.n_tiles
    smap       = dataset.stitch_map
    n_scenes   = len(dataset) // n_tiles
    needs_rgb  = (model.mode == "oracle_flow")
    is_quiver  = (model.mode == "quiver")
    is_luma    = (model.target_mode == "luma") or is_quiver
    results    = []

    data_iter = iter(loader)

    for si in tqdm(range(n_scenes), desc="eval tiled", disable=not _TTY):
        tile_batches = [next(data_iter) for _ in range(n_tiles)]
        alpha     = tile_batches[0]["alpha"][0].item()
        pct99_sc  = tile_batches[0]["pct99"][0].item()
        scene_key = tile_batches[0].get("scene_key", [""])[0]

        # Full-frame accumulators: (H,W) for luma, (3,H,W) for rgb
        gt_srgb = torch.zeros(H, W)    if is_luma else torch.zeros(3, H, W)
        pr_srgb = torch.zeros(H, W)    if is_luma else torch.zeros(3, H, W)

        for ti, tb in enumerate(tile_batches):
            cy0, cy1, cx0, cx1, ty0, ty1, tx0, tx1 = smap[ti]
            gt_rgb_t = tb["clean_rgb_seq"][0, HALF_WIN].float()  # (3, H_t, W_t)
            spad = tb["spad_mono_seq"].to(device)

            if is_quiver:
                spad_5d = spad.unsqueeze(2).half()
                with torch.cuda.amp.autocast():
                    _, out1, _, _ = model(spad_5d)
                pred_display = out1[0, 0, 0].float().cpu()
                pred_lin     = pred_display ** GAMMA
                p_t = (pred_lin * pct99_sc / max(alpha, 1e-8) / 3.0).clamp(0, 1) ** (1.0 / GAMMA)
                gt_lin_t = (gt_rgb_t ** GAMMA).mean(0)
                gt_t     = gt_lin_t.clamp(0, 1) ** (1.0 / GAMMA)
            elif is_luma:
                rgb = tb["clean_rgb_seq"].to(device) if needs_rgb else None
                with autocast("cuda"):
                    pred, _ = model(spad, clean_rgb=rgb)
                p_raw = pred[0, 0].float().cpu()
                p_t   = (p_raw / max(alpha, 1e-8) / 3.0).clamp(0, 1) ** (1.0 / GAMMA)
                gt_lin_t = (gt_rgb_t ** GAMMA).mean(0)
                gt_t     = gt_lin_t.clamp(0, 1) ** (1.0 / GAMMA)
            else:
                # RGB model: pred = alpha*(R,G,B)
                rgb = tb["clean_rgb_seq"].to(device) if needs_rgb else None
                with autocast("cuda"):
                    pred, _ = model(spad, clean_rgb=rgb)
                p_raw = pred[0].float().cpu()                             # (3, H_t, W_t)
                p_t   = (p_raw / max(alpha, 1e-8)).clamp(0, 1) ** (1.0 / GAMMA)
                gt_t  = gt_rgb_t                                          # (3, H_t, W_t) sRGB

            if is_luma:
                gt_srgb[cy0:cy1, cx0:cx1] = gt_t[ty0:ty1, tx0:tx1]
                pr_srgb[cy0:cy1, cx0:cx1] = p_t [ty0:ty1, tx0:tx1]
            else:
                gt_srgb[:, cy0:cy1, cx0:cx1] = gt_t[:, ty0:ty1, tx0:tx1]
                pr_srgb[:, cy0:cy1, cx0:cx1] = p_t [:, ty0:ty1, tx0:tx1]

        psnr_gam = psnr(pr_srgb, gt_srgb)
        if is_luma:
            ssim_gam = ssim(pr_srgb.numpy(), gt_srgb.numpy())
            p_lp = pr_srgb.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1) * 2 - 1
            g_lp = gt_srgb.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1) * 2 - 1
        else:
            from skimage.metrics import structural_similarity
            ssim_gam = float(structural_similarity(
                pr_srgb.permute(1, 2, 0).numpy(),
                gt_srgb.permute(1, 2, 0).numpy(),
                data_range=1.0, channel_axis=2))
            # LPIPS: use 1024×1024 centre crop for large frames (avoids OOM)
            cy = max((H - 1024) // 2, 0); cx_ = max((W - 1024) // 2, 0)
            ch = min(1024, H);             cw_ = min(1024, W)
            p_lp = pr_srgb[:, cy:cy+ch, cx_:cx_+cw_].unsqueeze(0) * 2 - 1
            g_lp = gt_srgb[:, cy:cy+ch, cx_:cx_+cw_].unsqueeze(0) * 2 - 1

        lpips_val = float('nan')
        if lpips_fn is not None:
            with torch.no_grad():
                lpips_val = lpips_fn(p_lp.to(device), g_lp.to(device)).item()

        results.append(dict(
            scene=scene_key,
            psnr_lin=float('nan'),
            psnr_gam=psnr_gam,
            ssim_gam=ssim_gam,
            lpips_gam=lpips_val,
        ))

    return results


# ── reporting ──────────────────────────────────────────────────────────────────

def aggregate(results):
    keys = ["psnr_lin", "psnr_gam", "ssim_gam", "lpips_gam"]
    agg  = {}
    for k in keys:
        vals = [r[k] for r in results if not math.isnan(r[k])]
        agg[k] = float(np.mean(vals)) if vals else float('nan')
    return agg


def print_table(results, agg, mode):
    print(f"\n{'='*70}")
    print(f"  SPADNet evaluation  |  mode={mode}  |  n={len(results)}")
    print(f"  {'Scene':<40s}  {'PSNRlin':>8}  {'PSNRgam':>8}  {'SSIMgam':>8}  {'LPIPS':>8}")
    print(f"  {'-'*66}")
    def _fmt(v): return f"{v:>8.3f}" if not math.isnan(v) else "     N/A"
    for r in results:
        print(f"  {r['scene']:<40s}  {_fmt(r['psnr_lin'])}  {_fmt(r['psnr_gam'])}"
              f"  {r['ssim_gam']:>8.4f}  {r['lpips_gam']:>8.4f}")
    print(f"  {'─'*66}")
    print(f"  {'MEAN':<40s}  {_fmt(agg['psnr_lin'])}  {_fmt(agg['psnr_gam'])}"
          f"  {agg['ssim_gam']:>8.4f}  {agg['lpips_gam']:>8.4f}")
    print(f"{'='*70}\n")


def save_csv(results, agg, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("scene,psnr_lin,psnr_gam,ssim_gam,lpips_gam\n")
        for r in results:
            f.write(f"{r['scene']},{r['psnr_lin']:.4f},{r['psnr_gam']:.4f},"
                    f"{r['ssim_gam']:.4f},{r['lpips_gam']:.4f}\n")
        f.write(f"MEAN,{agg['psnr_lin']:.4f},{agg['psnr_gam']:.4f},"
                f"{agg['ssim_gam']:.4f},{agg['lpips_gam']:.4f}\n")
    print(f"  Results saved to {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, mode, ckpt_sensor_mode = load_model(args.ckpt, device)
    sensor_mode = args.sensor_mode or ckpt_sensor_mode

    # LPIPS
    lpips_fn = None
    if LPIPS_OK and not args.no_lpips:
        lpips_fn = lpips_lib.LPIPS(net='alex').to(device).eval()
    elif not args.no_lpips:
        print("  [WARN] lpips not installed — skipping LPIPS metric")

    test_root = args.test_root
    if not test_root:
        raise RuntimeError("--test_root is required")

    if args.tiled:
        # Tiled (X4K-style) evaluation
        from src.data.dataset import _scenes, _pngs, Sample
        dirs = _scenes(test_root, SEQ_LEN)
        if not dirs:
            raise RuntimeError(f"No scenes under {test_root}")
        if args.sliding_window:
            samples = [Sample(os.path.relpath(d, test_root), _pngs(d)[w:w+SEQ_LEN])
                       for d in dirs for w in range(len(_pngs(d)) - SEQ_LEN + 1)]
        else:
            samples = [Sample(os.path.relpath(d, test_root), _pngs(d)[:SEQ_LEN])
                       for d in dirs]

        dataset = TiledDataset(samples, ppp=args.ppp, bins=args.bins,
                               tile_size=args.tile_size,
                               sensor_mode=sensor_mode)
        loader  = DataLoader(dataset, batch_size=1, shuffle=False,
                             num_workers=args.workers, pin_memory=True,
                             worker_init_fn=_worker_init if args.workers > 0 else None)
        results = eval_tiled(model, dataset, loader, device, lpips_fn)

    else:
        # Standard evaluation
        if args.sliding_window:
            samples = build_sliding(test_root)
        else:
            samples = build_first_window(test_root)

        dataset = SPADDataset(
            samples, ppp=args.ppp, bins=args.bins,
            augment=False, crop=256, eval_crop=args.eval_crop,
            seed=42, random_pick=False, sps=1,
            sensor_mode=sensor_mode, target_mode=model.target_mode)
        loader = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=args.workers, pin_memory=True,
                            worker_init_fn=_worker_init if args.workers > 0 else None)
        results = eval_standard(model, loader, device, lpips_fn)

    agg = aggregate(results)
    print_table(results, agg, mode)

    out_csv = args.out_csv
    if not out_csv:
        out_csv = os.path.join(os.path.dirname(args.ckpt), "eval_results.csv")
    save_csv(results, agg, out_csv)


if __name__ == "__main__":
    main()
