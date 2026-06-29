"""
SPAD forward model: scene statistics and binomial simulation.

FIX-1: scene stats (alpha, pct99) computed on full frame; simulation runs on crop only.
"""

import numpy as np

GAMMA = 2.2


# ─────────────────────────────────────────────────────────────────────────────
# RGGB Bayer helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rggb_masks(h: int, w: int, y0: int = 0, x0: int = 0):
    """Boolean masks for an RGGB CFA with global origin (0,0)=R.

    Passing the crop origin (y0, x0) preserves Bayer phase for un-augmented
    crops.  After pre-simulation augmentation always pass y0=x0=0.
    Returns: r_mask, g_mask, b_mask  (each shape h×w, g = g1|g2 combined)
    """
    yy, xx = np.indices((h, w))
    yy = yy + int(y0); xx = xx + int(x0)
    r = (yy % 2 == 0) & (xx % 2 == 0)
    g = ((yy % 2 == 0) & (xx % 2 == 1)) | ((yy % 2 == 1) & (xx % 2 == 0))
    b = (yy % 2 == 1) & (xx % 2 == 1)
    return r, g, b


def augment_sequence_pre(x_lin: np.ndarray, rgb: np.ndarray, rng) -> tuple:
    """Spatial + temporal augmentation applied to clean sequences before simulation.

    Use this for RGGB: augmenting after mosaicking would scramble the CFA
    channel assignment under PixelUnshuffle.  After this function the caller
    should simulate with cfa_y0=cfa_x0=0.
    Statistically equivalent to post-sim augmentation for mono.
    """
    if rng.random() < 0.5:
        k = rng.randint(1, 3)
        x_lin = np.rot90(x_lin, k=k, axes=(1, 2)).copy()
        rgb   = np.rot90(rgb,   k=k, axes=(1, 2)).copy()
    if rng.random() < 0.5:
        x_lin = x_lin[:, :, ::-1, :].copy()
        rgb   = rgb  [:, :, ::-1, :].copy()
    if rng.random() < 0.5:
        x_lin = x_lin[:, ::-1, :, :].copy()
        rgb   = rgb  [:, ::-1, :, :].copy()
    if rng.random() < 0.5:
        x_lin = x_lin[::-1].copy()
        rgb   = rgb  [::-1].copy()
    return x_lin, rgb


def scene_stats(seq_rgb: np.ndarray, ppp: float, ctr: int):
    """
    Scene-level alpha and pct99 from full-resolution frame sequence.

    Args:
        seq_rgb: (T, H, W, 3) float32 in [0, 1]
        ppp: photons-per-pixel target (mean of alpha*S = ppp)
        ctr: centre frame index

    Returns:
        x_lin: (T, H, W, 3) linearised frames
        alpha: scene exposure scalar
        pct99: 99th-percentile of centre-frame photon counts (for eval normalisation)
    """
    x_lin = np.power(np.clip(seq_rgb, 0., 1.), GAMMA).astype(np.float32)
    alpha = ppp / max(float(x_lin.sum(-1).mean()), 1e-8)
    tgt_s = (x_lin[ctr] * alpha).sum(-1)                    # (H, W)
    pct99 = float(np.percentile(tgt_s, 99)) + 1e-8
    return x_lin, alpha, pct99


def simulate_spad(x_lin_patch: np.ndarray, alpha: float,
                  bins: int, ctr: int, pct99: float,
                  sensor_mode: str = "mono",
                  cfa_y0: int = 0, cfa_x0: int = 0) -> dict:
    """
    Binomial SPAD simulation on a patch using scene-level alpha.

    Args:
        x_lin_patch: (T, h, w, 3) linearised crop
        alpha:       scene exposure scalar from scene_stats
        bins:        number of SPAD bins (e.g. 7)
        ctr:         centre frame index
        pct99:       scene-level pct99 from scene_stats (metadata only)
        sensor_mode: "mono" — all channels summed; "rggb" — Bayer CFA mosaic
        cfa_y0/x0:   global crop origin for Bayer phase (0,0 after augmentation)

    Returns dict with:
        spad_mono_seq: (T, h, w) float32 in [0, 1]  (mono or RGGB mosaic)
        target_s:      (h, w)    float32  alpha*(R+G+B) photon counts
        target_rgb_s:  (3, h, w) float32  alpha*(R, G, B) per-channel photons
        pct99:         float     scene-level, for eval normalisation only
        alpha:         float
    """
    T, h, w, _ = x_lin_patch.shape
    scaled = x_lin_patch * alpha                             # (T, h, w, 3)
    frames = []

    if sensor_mode == "mono":
        for t in range(T):
            lam = scaled[t].sum(-1)
            p   = np.clip(1. - np.exp(-lam / bins), 0., 1.)
            frames.append(np.random.binomial(bins, p).astype(np.float32) / bins)
    elif sensor_mode == "rggb":
        r_mask, g_mask, b_mask = _rggb_masks(h, w, cfa_y0, cfa_x0)
        for t in range(T):
            lam = np.zeros((h, w), np.float32)
            lam[r_mask] = scaled[t, :, :, 0][r_mask]
            lam[g_mask] = scaled[t, :, :, 1][g_mask]
            lam[b_mask] = scaled[t, :, :, 2][b_mask]
            p = np.clip(1. - np.exp(-lam / bins), 0., 1.)
            frames.append(np.random.binomial(bins, p).astype(np.float32) / bins)
    else:
        raise ValueError(f"Unknown sensor_mode={sensor_mode!r}; use 'mono' or 'rggb'")

    target_rgb_s = scaled[ctr].transpose(2, 0, 1).astype(np.float32)  # (3, h, w)
    target_s     = scaled[ctr].sum(-1).astype(np.float32)              # (h, w)

    return dict(
        spad_mono_seq=np.stack(frames, 0),   # (T, h, w) in [0, 1]
        target_s=target_s,                   # (h, w) in [0, inf), mean ~= ppp
        target_rgb_s=target_rgb_s,           # (3, h, w) per-channel photons
        pct99=pct99,                         # scene-level, eval only
        alpha=alpha,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CMOS forward model
# ─────────────────────────────────────────────────────────────────────────────

def _bayer_mask(h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w, 3), dtype=np.float32)
    m[0::2, 0::2, 0] = 1.0   # R
    m[0::2, 1::2, 1] = 1.0   # G1
    m[1::2, 0::2, 1] = 1.0   # G2
    m[1::2, 1::2, 2] = 1.0   # B
    return m


def _pack_rggb(raw: np.ndarray) -> np.ndarray:
    """Pack (H, W) Bayer image into (4, H/2, W/2): [R, Gr, Gb, B]."""
    return np.stack([
        raw[0::2, 0::2],
        raw[0::2, 1::2],
        raw[1::2, 0::2],
        raw[1::2, 1::2],
    ], axis=0)


def simulate_cmos(x_lin_patch: np.ndarray, alpha: float,
                  sigma: float = 2.0, cmos_t: int = 1) -> dict:
    """
    CMOS forward simulation on a patch using scene-level alpha.

    Integrates `cmos_t` frames centred at T//2 onto a Bayer grid, adds
    Poisson shot noise and Gaussian read noise (single-read model, longer
    exposure = more photons but same read noise), and quantises to 8-bit.

    cmos_t=1  → single center frame (baseline)
    cmos_t>1  → longer exposure: integrates cmos_t frames, higher SNR

    Args:
        x_lin_patch: (T, h, w, 3) linearised crop (output of scene_stats)
        alpha:        scene exposure scalar from scene_stats
        sigma:        read-noise std dev in ADC counts (applied once)
        cmos_t:       number of frames to integrate (odd, centred at T//2)

    Returns dict with:
        cmos_packed: (4, h/2, w/2) float32 in [0, 1]  — packed RGGB
    """
    T, h, w, _ = x_lin_patch.shape
    ctr  = T // 2
    half = cmos_t // 2
    window   = x_lin_patch[ctr - half : ctr + half + 1]    # (cmos_t, h, w, 3)
    scaled   = window * alpha
    mask     = _bayer_mask(h, w)
    cmos_lam = (scaled.sum(0) * mask).sum(-1)               # (h, w) integrated counts
    cmos_cnt = np.random.poisson(cmos_lam).astype(np.float32)
    cmos_cnt += np.random.normal(0., sigma, cmos_cnt.shape).astype(np.float32)
    cmos_adc = np.clip(np.round(cmos_cnt), 0., 255.).astype(np.float32)
    return dict(cmos_packed=_pack_rggb(cmos_adc / 255.))


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation
# ─────────────────────────────────────────────────────────────────────────────

def augment_sequence(spad: np.ndarray, tgt: np.ndarray, rgb: np.ndarray,
                     rng) -> tuple:
    """
    Consistent spatial + temporal augmentation applied post-crop.

    Args:
        spad: (T, H, W) float32
        tgt:  (H, W)    float32
        rgb:  (T, H, W, 3) float32
        rng:  random.Random instance

    Returns:
        spad, tgt, rgb (same shapes, augmented consistently)
    """
    if rng.random() < 0.5:
        k = rng.randint(1, 3)
        spad = np.rot90(spad, k=k, axes=(1, 2)).copy()
        tgt  = np.rot90(tgt,  k=k).copy()
        rgb  = np.rot90(rgb,  k=k, axes=(1, 2)).copy()
    if rng.random() < 0.5:
        spad = spad[:, :, ::-1].copy()
        tgt  = tgt[:, ::-1].copy()
        rgb  = rgb[:, :, ::-1, :].copy()
    if rng.random() < 0.5:
        spad = spad[:, ::-1, :].copy()
        tgt  = tgt[::-1, :].copy()
        rgb  = rgb[:, ::-1, :, :].copy()
    if rng.random() < 0.5:    # temporal flip — safe for odd T
        spad = spad[::-1].copy()
        rgb  = rgb[::-1].copy()
    return spad, tgt, rgb
