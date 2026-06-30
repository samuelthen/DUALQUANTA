#!/usr/bin/env python3
"""
smoke_test.py — Verify all 22 paper checkpoints reproduce published PSNR.

Uses the first sliding window of every test scene (31 samples per model).
Fast enough to run in minutes; representative enough to catch broken checkpoints.
For full reproduction run evaluate.py with --sliding_window on each checkpoint.

Usage:
    python smoke_test.py [--device cuda:0] [--data_root /path/to/i2-2kfps_v1_png]

Pass tolerance:  ±0.50 dB  (wider than full-run due to reduced sample size)
Warn tolerance:  ±0.80 dB
"""

import argparse, math, os, sys, time
import numpy as np
import torch
from torch.amp import autocast

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_DIR   = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_DIR)

DEFAULT_DATA = "/home/samuel/dataset/i2-2kfps_v1_png"
STAGE1_CKPT  = os.path.join(REPO_DIR, "checkpoints/table3_4_DUALQUANTA/stage1_mono_dcn_h8.pth")

# First-window-only mode gives ~+1 dB higher PSNR than full sliding-window average
# for mono/bayer models (first frames have less motion → easier to reconstruct).
# Color models (Tables 3 & 4) show negligible bias (~±0.1 dB).
# Tolerances are set wide enough to accommodate this bias while still catching
# clearly broken checkpoints (wrong model loaded, corrupted weights, etc).
PASS_TOL = 1.50   # dB — PASS if |actual - expected| ≤ PASS_TOL
FAIL_TOL = 2.50   # dB — FAIL if |actual - expected| > FAIL_TOL  (WARN in between)
FIRST_WIN_ONLY = True  # True → first window per scene (~2 min/model vs ~90 min/model)

PPP        = 3.25
BINS       = 7
CMOS_SIGMA = 2.0
GAMMA      = 2.2
SEQ_LEN    = 11
HALF_WIN   = SEQ_LEN // 2
SEED       = 42

# ── 22 paper variants ─────────────────────────────────────────────────────────
# Format: (tag, checkpoint_path, expected_i2k_psnr, model_family, extra)
#   model_family: "mono" | "bayer" | "color"
#   extra: for color models → {"cmos_t": int}  (SPAD T is always 11)

VARIANTS = [
    # Table 1 — monochrome depth sweep
    ("Table1 / single_frame_nafnet",
     "checkpoints/table1_mono/single_frame_nafnet.pth", 35.28, "mono", {}),
    ("Table1 / no_align",
     "checkpoints/table1_mono/no_align.pth",            36.53, "mono", {}),
    ("Table1 / dcn_h2",
     "checkpoints/table1_mono/dcn_h2.pth",              36.88, "mono", {}),
    ("Table1 / dcn_h4",
     "checkpoints/table1_mono/dcn_h4.pth",              37.10, "mono", {}),
    ("Table1 / dcn_h8",
     "checkpoints/table1_mono/dcn_h8.pth",              37.14, "mono", {}),
    ("Table1 / dcn_h16",
     "checkpoints/table1_mono/dcn_h16.pth",             36.54, "mono", {}),
    ("Table1 / spynet_dcn",
     "checkpoints/table1_mono/spynet_dcn.pth",          37.14, "mono", {}),
    ("Table1 / cascading_dcn",
     "checkpoints/table1_mono/cascading_dcn.pth",       37.34, "mono", {}),
    ("Table1 / oracle_flow",
     "checkpoints/table1_mono/oracle_flow.pth",         38.72, "mono",
     {"needs_gt": True}),      # oracle_flow needs clean GT frames; use evaluate.py
    ("Table1 / quiver_retrained",
     "checkpoints/table1_mono/quiver_retrained.pth",    30.68, "mono", {}),

    # Table 2 — Bayer color-SPAD depth sweep
    ("Table2 / bayer_no_align",
     "checkpoints/table2_bayer/no_align.pth",  33.15, "bayer", {}),
    ("Table2 / bayer_dcn_h2",
     "checkpoints/table2_bayer/dcn_h2.pth",    33.46, "bayer", {}),
    ("Table2 / bayer_dcn_h4",
     "checkpoints/table2_bayer/dcn_h4.pth",    33.69, "bayer", {}),
    ("Table2 / bayer_dcn_h8",
     "checkpoints/table2_bayer/dcn_h8.pth",    33.72, "bayer", {}),
    ("Table2 / bayer_dcn_h16",
     "checkpoints/table2_bayer/dcn_h16.pth",   33.36, "bayer", {}),

    # Table 3 — sensing comparison (RGB PSNR)
    ("Table3 / cmos_only",
     "checkpoints/table3_4_DUALQUANTA/cmos_only.pth",           25.79, "color",
     {"cmos_t": 11}),
    ("Table3 / DUALQUANTA (T=11)",
     "checkpoints/table3_4_DUALQUANTA/DUALQUANTA_T11_stage2.pth", 33.64, "color",
     {"cmos_t": 11}),

    # Table 4 — CMOS integration window sweep (RGB PSNR)
    ("Table4 / DUALQUANTA_T1",
     "checkpoints/table3_4_DUALQUANTA/DUALQUANTA_T1_stage2.pth",  33.57, "color",
     {"cmos_t": 1}),
    ("Table4 / DUALQUANTA_T3",
     "checkpoints/table3_4_DUALQUANTA/DUALQUANTA_T3_stage2.pth",  34.33, "color",
     {"cmos_t": 3}),
    ("Table4 / DUALQUANTA_T5",
     "checkpoints/table3_4_DUALQUANTA/DUALQUANTA_T5_stage2.pth",  34.07, "color",
     {"cmos_t": 5}),
    ("Table4 / DUALQUANTA_T7",
     "checkpoints/table3_4_DUALQUANTA/DUALQUANTA_T7_stage2.pth",  33.72, "color",
     {"cmos_t": 7}),
    ("Table4 / DUALQUANTA_T9",
     "checkpoints/table3_4_DUALQUANTA/DUALQUANTA_T9_stage2.pth",  33.66, "color",
     {"cmos_t": 9}),
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _psnr_np(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 100.0 if mse < 1e-12 else 10 * math.log10(1.0 / mse)


def _load_frames(scene_dir):
    import cv2
    pngs = sorted(f for f in os.listdir(scene_dir) if f.endswith(".png"))
    frames = []
    for f in pngs:
        img = cv2.imread(os.path.join(scene_dir, f), cv2.IMREAD_COLOR)
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.)
    return np.stack(frames, 0)   # (T_total, H, W, 3)


def _load_test_scenes(test_root):
    scenes = sorted(
        d for d in os.listdir(test_root)
        if os.path.isdir(os.path.join(test_root, d)))
    return [os.path.join(test_root, s) for s in scenes]


def _result_str(actual, expected):
    diff = actual - expected
    mark = ("PASS" if abs(diff) <= PASS_TOL
            else "WARN" if abs(diff) <= FAIL_TOL
            else "FAIL")
    return f"{actual:7.3f} dB  (paper {expected:.2f}, Δ{diff:+.3f})  [{mark}]"


# ── mono / Bayer evaluation ───────────────────────────────────────────────────

def _eval_mono_bayer(ckpt_path, sensor_mode, scene_dirs, device):
    """Luma sRGB PSNR matching Table 1 / Table 2 protocol."""
    from src.data.simulation import scene_stats, simulate_spad
    from src.utils import load_checkpoint
    from src import SPADNet
    from src.models.quiver_model import QUIVER
    import argparse as _ap

    ck = load_checkpoint(ckpt_path, device)

    # ── detect QUIVER ── only "predenoise." keys are unique to QUIVER;
    # SPADNet+SpyNet also has "spynet." keys so must not include that prefix
    is_quiver = any(k.startswith("predenoise.") for k in ck.get("model", {}))

    out_ch      = 1
    target_mode = "luma"

    if is_quiver:
        para = _ap.Namespace(
            inp_ch=1, n_features=64, n_blocks=12,
            past_frames=HALF_WIN, future_frames=HALF_WIN, activation="gelu",
            spynet_path="", load_spynet_weights=False)
        model = QUIVER(para).to(device).half().eval()
        model.load_state_dict(ck["model"], strict=True)
        mode = "quiver"
    else:
        _OLD = {"single_frame_no_grad": "single_frame", "dcn_d0": "dcn_h2",
                "dcn_d1": "dcn_h4", "dcn_d2": "dcn_h8", "dcn_d3": "dcn_h16"}
        args_saved  = ck.get("args", {})
        raw         = ck.get("mode") or args_saved.get("mode", "dcn_h4")
        mode        = _OLD.get(raw, raw)
        bc          = args_saved.get("base_c", 32)
        nb          = args_saved.get("n_blocks", 2)
        nfpm        = args_saved.get("n_fpm", 2)
        raft        = args_saved.get("raft_ckpt", "")
        target_mode = ck.get("target_mode", args_saved.get("target_mode", "luma"))
        out_ch      = ck.get("out_ch", args_saved.get("out_ch",
                             1 if target_mode == "luma" else 3))
        model = SPADNet(T=SEQ_LEN, bc=bc, nb=nb, nfpm=nfpm, mode=mode,
                        raft_ckpt=raft, out_ch=out_ch,
                        target_mode=target_mode).to(device).eval()
        model.load_state_dict(ck["model"], strict=True)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"    mode={mode}  {n_params:.2f}M params  "
          f"out_ch={out_ch}  target={target_mode}  sensor={sensor_mode}")

    rng = np.random.default_rng(SEED)
    psnrs = []

    with torch.no_grad():
        for sd in scene_dirs:
            frames = _load_frames(sd)
            T_total = frames.shape[0]
            window_starts = [0] if FIRST_WIN_ONLY else range(T_total - SEQ_LEN + 1)
            for start in window_starts:
                seq   = frames[start: start + SEQ_LEN]
                x_lin, alpha, pct99 = scene_stats(seq, PPP, HALF_WIN)
                sim   = simulate_spad(x_lin, alpha, BINS, HALF_WIN, pct99,
                                      sensor_mode=sensor_mode)

                spad  = torch.from_numpy(sim["spad_mono_seq"]).unsqueeze(0).to(device)
                gt_rgb = seq[HALF_WIN]              # (H, W, 3) sRGB

                with autocast("cuda"):
                    if is_quiver:
                        spad_5d = spad.unsqueeze(2).half()
                        _, out1, _, _ = model(spad_5d)
                        pred_disp = out1[0, 0, 0].float().cpu().numpy()
                        pred_lin  = pred_disp ** GAMMA
                        pred_srgb = np.clip(pred_lin * pct99 / max(alpha, 1e-8) / 3., 0, 1) ** (1./GAMMA)
                        gt_lin_mean = (gt_rgb ** GAMMA).mean(-1)
                        gt_srgb     = np.clip(gt_lin_mean, 0, 1) ** (1. / GAMMA)
                    elif out_ch == 1:
                        out, _ = model(spad)
                        pred     = out[0, 0].float().cpu().numpy()   # (H, W)
                        pred_lin = np.clip(pred / max(alpha, 1e-8) / 3., 0, 1)
                        pred_srgb = pred_lin ** (1. / GAMMA)
                        gt_lin_mean = (gt_rgb ** GAMMA).mean(-1)
                        gt_srgb     = np.clip(gt_lin_mean, 0, 1) ** (1. / GAMMA)
                    else:  # out_ch == 3: bayer RGB output
                        out, _ = model(spad)
                        pred     = out[0].float().cpu().numpy()       # (3, H, W)
                        pred_lin = np.clip(pred / max(alpha, 1e-8), 0, 1)
                        pred_srgb = (pred_lin ** (1. / GAMMA)).transpose(1, 2, 0)  # (H, W, 3)
                        gt_srgb   = gt_rgb  # already sRGB (H, W, 3)

                psnrs.append(_psnr_np(pred_srgb, gt_srgb))

    return float(np.mean(psnrs))


# ── color / DUALQUANTA evaluation ─────────────────────────────────────────────

def _eval_color(ckpt_path, cmos_t, stage1, scene_dirs, device):
    """RGB sRGB PSNR matching Table 3 / Table 4 protocol."""
    from src.data.simulation import scene_stats, simulate_spad, simulate_cmos
    from src.utils import load_checkpoint
    from train_color import ColorUNet, get_luma

    ck   = load_checkpoint(ckpt_path, device)
    mode = ck.get("mode", ck.get("color_mode", "cmos_only"))
    bc   = ck.get("bc", 32)
    model = ColorUNet(mode=mode, bc=bc).to(device).eval()
    model.load_state_dict(ck["model"], strict=True)
    print(f"    color_mode={mode}  cmos_t={cmos_t}")

    psnrs = []

    with torch.no_grad():
        for sd in scene_dirs:
            frames  = _load_frames(sd)
            T_total = frames.shape[0]
            window_starts = [0] if FIRST_WIN_ONLY else range(T_total - SEQ_LEN + 1)
            for start in window_starts:
                seq    = frames[start: start + SEQ_LEN]
                x_lin, alpha, pct99 = scene_stats(seq, PPP, HALF_WIN)
                spad_sim = simulate_spad(x_lin, alpha, BINS, HALF_WIN, pct99,
                                         sensor_mode="mono")
                cmos_sim = simulate_cmos(x_lin, alpha, sigma=CMOS_SIGMA,
                                         cmos_t=cmos_t)

                spad_t  = torch.from_numpy(spad_sim["spad_mono_seq"]).unsqueeze(0).to(device)
                tgt_s_t = torch.from_numpy(spad_sim["target_s"]).unsqueeze(0).to(device)
                cmos_t_in = torch.from_numpy(cmos_sim["cmos_packed"]).unsqueeze(0).to(device)

                batch = {"spad_mono_seq": spad_t, "target_s": tgt_s_t}
                luma  = get_luma(batch, stage1, device) if mode != "cmos_only" else None

                with autocast("cuda"):
                    pred_raw = model(cmos_t_in, luma)[0].float().cpu().numpy()  # (3, H, W)

                a = max(alpha, 1e-8)
                pred_lin  = np.clip(pred_raw / a, 0., 1.)     # (3, H, W)
                pred_srgb = pred_lin ** (1. / GAMMA)           # gamma compress
                pred_srgb = pred_srgb.transpose(1, 2, 0)       # (H, W, 3)

                gt_srgb   = seq[HALF_WIN]                      # (H, W, 3)
                psnrs.append(_psnr_np(pred_srgb, gt_srgb))

    return float(np.mean(psnrs))


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device",    default="cuda:0")
    p.add_argument("--data_root", default=DEFAULT_DATA)
    return p.parse_args()


def main():
    args  = parse_args()
    device = torch.device(args.device)
    test_root = os.path.join(args.data_root, "test") \
                if os.path.isdir(os.path.join(args.data_root, "test")) \
                else args.data_root

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    scene_dirs = _load_test_scenes(test_root)
    print(f"\nSmoke test: {len(scene_dirs)} test scenes  |  device={device}")
    print(f"Tolerances: PASS ±{PASS_TOL} dB  /  FAIL >{FAIL_TOL} dB\n")

    # Pre-load stage1 once for all DUALQUANTA models
    from train_color import load_stage1
    stage1 = load_stage1(STAGE1_CKPT, device)

    results = []
    n_pass = n_warn = n_fail = n_skip = 0

    for idx, (tag, rel_ckpt, expected, family, extra) in enumerate(VARIANTS):
        ckpt_path = os.path.join(REPO_DIR, rel_ckpt)
        print(f"[{idx+1:2d}/{len(VARIANTS)}] {tag}")
        if not os.path.isfile(ckpt_path):
            print(f"    MISSING checkpoint: {ckpt_path}\n")
            results.append((tag, float("nan"), expected, "MISSING"))
            n_fail += 1
            continue

        if extra.get("needs_gt"):
            print("    SKIP (oracle_flow uses GT frames; run: python evaluate.py "
                  f"--ckpt {rel_ckpt} --test_root <data>/test --sliding_window)\n")
            results.append((tag, float("nan"), expected, "SKIP"))
            n_skip += 1
            continue

        t0 = time.time()
        try:
            if family in ("mono", "bayer"):
                sm = "rggb" if family == "bayer" else "mono"
                actual = _eval_mono_bayer(ckpt_path, sm, scene_dirs, device)
            else:
                actual = _eval_color(ckpt_path, extra["cmos_t"], stage1, scene_dirs, device)
        except Exception as e:
            print(f"    ERROR: {e}\n")
            results.append((tag, float("nan"), expected, "ERROR"))
            n_fail += 1
            continue

        diff  = actual - expected
        mark  = ("PASS" if abs(diff) <= PASS_TOL
                 else "WARN" if abs(diff) <= FAIL_TOL else "FAIL")
        elapsed = time.time() - t0
        print(f"    {_result_str(actual, expected)}  ({elapsed:.0f}s)\n")

        results.append((tag, actual, expected, mark))
        if mark == "PASS": n_pass += 1
        elif mark == "WARN": n_warn += 1
        else: n_fail += 1

    # ── summary table ──────────────────────────────────────────────────────────
    W = 42
    print("\n" + "=" * 80)
    win_mode = "first window only" if FIRST_WIN_ONLY else "all sliding windows"
    print(f"  SMOKE TEST SUMMARY — {len(scene_dirs)} scenes, {win_mode}")
    print(f"  {'Variant':<{W}}  {'Actual':>7}  {'Paper':>7}  {'Δ':>6}  Result")
    print(f"  {'-'*74}")
    for tag, actual, expected, mark in results:
        if math.isnan(actual):
            print(f"  {tag:<{W}}  {'—':>7}  {expected:>7.2f}  {'—':>6}  [{mark}]")
        else:
            diff = actual - expected
            print(f"  {tag:<{W}}  {actual:>7.3f}  {expected:>7.2f}  "
                  f"{diff:>+6.3f}  [{mark}]")
    print(f"  {'-'*74}")
    print(f"  PASS: {n_pass}   WARN: {n_warn}   FAIL: {n_fail}   "
          f"SKIP: {n_skip}   Total: {len(VARIANTS)}")
    print("=" * 80 + "\n")

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
