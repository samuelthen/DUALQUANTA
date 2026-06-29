"""
Datasets for SPADNet training and evaluation.

SPADDataset  — standard per-sample dataset (training and eval)
TiledDataset — tiled high-res evaluation (X4K-style)
build_splits — random train/eval scene split
build_test   — sliding-window test samples
"""

import math
import os
import random
from dataclasses import dataclass
from typing import List

import cv2
cv2.setNumThreads(0)
import numpy as np
import torch
from torch.utils.data import Dataset

from .simulation import (scene_stats, simulate_spad,
                         augment_sequence, augment_sequence_pre)

SEQ_LEN  = 11
HALF_WIN = SEQ_LEN // 2
PPP      = 3.25
BINS     = 7
GAMMA    = 2.2


@dataclass
class Sample:
    scene_key:   str
    frame_paths: List[str]


# ── helpers ───────────────────────────────────────────────────────────────────

def _read_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.


def _worker_init(wid):
    s = torch.initial_seed() % 2**32
    random.seed(s + wid)
    np.random.seed(s + wid)


def _scenes(root, min_n):
    out = []
    for d, _, fs in os.walk(root):
        if sum(1 for f in fs if f.lower().endswith('.png')) >= min_n:
            out.append(d)
    return sorted(out)


def _pngs(d):
    return sorted(os.path.join(d, f) for f in os.listdir(d)
                  if f.lower().endswith('.png'))


# ── split builders ────────────────────────────────────────────────────────────

def build_splits(root, frac=0.1, seed=42):
    """
    Split scenes into train / eval sets.

    Returns (train_samples, eval_samples) where each entry is a Sample.
    Eval samples use non-overlapping SEQ_LEN-frame windows.
    """
    dirs = _scenes(root, SEQ_LEN)
    if not dirs:
        raise RuntimeError(f"No scenes under {root}")
    rng = random.Random(seed)
    dirs = dirs[:]
    rng.shuffle(dirs)
    n_ev = max(1, min(int(round(len(dirs) * frac)), len(dirs) - 1))
    ev, tr = dirs[:n_ev], dirs[n_ev:]
    tr_s = [Sample(os.path.relpath(d, root), _pngs(d)) for d in tr]
    ev_s = [Sample(os.path.relpath(d, root), _pngs(d)[w*SEQ_LEN:(w+1)*SEQ_LEN])
            for d in ev for w in range(len(_pngs(d)) // SEQ_LEN)]
    print(f"[split] train={len(tr)} scenes / {len(tr_s)} samples  "
          f"eval={len(ev)} scenes / {len(ev_s)} samples")
    return tr_s, ev_s


def build_first_window(root):
    """One sample per scene — first SEQ_LEN frames only."""
    dirs = _scenes(root, SEQ_LEN)
    if not dirs:
        raise RuntimeError(f"No scenes under {root}")
    return [Sample(os.path.relpath(d, root), _pngs(d)[:SEQ_LEN]) for d in dirs]


def build_test(root):
    """All non-overlapping SEQ_LEN-frame windows across every scene."""
    dirs = _scenes(root, SEQ_LEN)
    if not dirs:
        raise RuntimeError(f"No scenes under {root}")
    return [Sample(os.path.relpath(d, root), _pngs(d)[w*SEQ_LEN:(w+1)*SEQ_LEN])
            for d in dirs for w in range(len(_pngs(d)) // SEQ_LEN)]


def build_sliding(root):
    """All step-1 overlapping SEQ_LEN-frame windows across every scene."""
    dirs = _scenes(root, SEQ_LEN)
    if not dirs:
        raise RuntimeError(f"No scenes under {root}")
    return [Sample(os.path.relpath(d, root), _pngs(d)[w:w+SEQ_LEN])
            for d in dirs for w in range(len(_pngs(d)) - SEQ_LEN + 1)]


# ── main dataset ──────────────────────────────────────────────────────────────

class SPADDataset(Dataset):
    """
    Dataset for SPADNet training and standard evaluation.

    Each item loads a SEQ_LEN-frame RGB sequence, computes scene-level
    statistics on the full frame (FIX-1), then simulates SPAD on the crop.
    """

    def __init__(self, samples, ppp=PPP, bins=BINS,
                 augment=False, crop=256, eval_crop=256,
                 seed=42, random_pick=False, sps=1,
                 sensor_mode="mono", target_mode="luma"):
        self.samples     = samples
        self.ppp         = ppp
        self.bins        = bins
        self.augment     = augment
        self.crop        = crop
        self.eval_crop   = eval_crop
        self.random_pick = random_pick
        self.sps         = max(1, int(sps))
        self.rng         = random.Random(seed)
        self.sensor_mode = sensor_mode
        self.target_mode = target_mode

    def __len__(self):
        return len(self.samples) * self.sps if self.random_pick else len(self.samples)

    def _pick(self, paths):
        if self.random_pick and len(paths) >= SEQ_LEN:
            return [paths[i] for i in sorted(
                self.rng.sample(range(len(paths)), SEQ_LEN))]
        return paths[:SEQ_LEN]

    def _load(self, paths):
        frames, nfb = [], 0
        for p in paths:
            try:
                fr = _read_rgb(p)
            except Exception:
                nfb += 1
                fr = frames[-1].copy() if frames else None
            frames.append(fr)
        valid = next((f for f in frames if f is not None), None)
        if valid is None:
            raise RuntimeError(f"All frames failed: {paths[0]}")
        fallback = np.zeros_like(valid)
        frames = [fallback if f is None else f for f in frames]
        return np.stack(frames, 0), nfb     # (T, H_native, W_native, 3)

    def __getitem__(self, idx):
        seq = None
        for _ in range(20):
            s = (self.samples[idx % len(self.samples)]
                 if self.random_pick else self.samples[idx])
            seq_try, nfb = self._load(self._pick(s.frame_paths))
            if not self.random_pick or nfb <= 2:
                seq = seq_try
                break
            idx = self.rng.randint(0, len(self.samples) - 1)
        if seq is None:
            raise RuntimeError("Failed after 20 retries")

        T, H, W, _ = seq.shape

        # FIX-1: scene stats (alpha, pct99) on FULL frame
        x_lin_full, alpha, pct99_val = scene_stats(seq, self.ppp, HALF_WIN)

        # Crop box — for RGGB snap to even origin so PixelUnshuffle gives R,G1,G2,B
        if self.augment:
            y0 = self.rng.randint(0, H - self.crop)
            x0 = self.rng.randint(0, W - self.crop)
            if self.sensor_mode == "rggb":
                y0 -= y0 % 2
                x0 -= x0 % 2
            ch = cw = self.crop
        elif self.eval_crop > 0:
            y0 = (H - self.eval_crop) // 2
            x0 = (W - self.eval_crop) // 2
            if self.sensor_mode == "rggb":
                y0 -= y0 % 2
                x0 -= x0 % 2
            ch = cw = self.eval_crop
        else:
            y0 = x0 = 0
            ch = H
            cw = W

        x_lin_patch = x_lin_full[:, y0:y0+ch, x0:x0+cw, :]
        rgb_patch   = seq        [:, y0:y0+ch, x0:x0+cw, :]

        # For RGGB, augment BEFORE simulation so CFA origin stays at (0,0)
        if self.augment and self.sensor_mode == "rggb":
            x_lin_patch, rgb_patch = augment_sequence_pre(x_lin_patch, rgb_patch, self.rng)
            cfa_y0, cfa_x0 = 0, 0
        else:
            cfa_y0, cfa_x0 = y0, x0

        sim = simulate_spad(x_lin_patch, alpha, self.bins, HALF_WIN, pct99_val,
                            sensor_mode=self.sensor_mode,
                            cfa_y0=cfa_y0, cfa_x0=cfa_x0)

        spad    = sim["spad_mono_seq"]   # (T, ch, cw)
        tgt     = sim["target_s"]        # (ch, cw) raw alpha*S
        tgt_rgb = sim["target_rgb_s"]    # (3, ch, cw) raw alpha*(R,G,B)

        # Post-crop augmentation (mono only — RGGB is augmented pre-sim above)
        if self.augment and self.sensor_mode == "mono":
            spad, tgt, rgb_patch = augment_sequence(spad, tgt, rgb_patch, self.rng)

        return dict(
            spad_mono_seq=torch.from_numpy(spad.copy()),
            clean_rgb_seq=torch.from_numpy(
                rgb_patch.transpose(0, 3, 1, 2).astype(np.float32).copy()),
            target_s=torch.from_numpy(tgt.copy()),
            target_rgb_s=torch.from_numpy(tgt_rgb.copy()),
            pct99=torch.tensor(pct99_val, dtype=torch.float32),
            alpha=torch.tensor(float(alpha), dtype=torch.float32),
            scene=s.scene_key,
        )


# ── tiled high-res dataset ────────────────────────────────────────────────────

class TiledDataset(Dataset):
    """
    Tiled evaluation dataset for high-resolution sequences (e.g. X4K 4K video).

    Tiles each frame into overlapping patches, with midpoint-boundary stitching
    to reconstruct the full frame without seam artefacts.

    Scene statistics (alpha, pct99) are computed once per scene in __init__
    and cached — not recomputed per tile in __getitem__.
    """

    @staticmethod
    def make_grid(H, W, ts):
        n_rows = math.ceil(H / ts)
        n_cols = math.ceil(W / ts)
        ys = np.round(np.linspace(0, H - ts, n_rows)).astype(int) & ~1
        xs = np.round(np.linspace(0, W - ts, n_cols)).astype(int) & ~1
        return [(f'r{ri}c{ci}', int(y), int(x))
                for ri, y in enumerate(ys) for ci, x in enumerate(xs)]

    def __init__(self, samples, ppp=PPP, bins=BINS, tile_size=1024, sensor_mode="mono"):
        self.ts          = tile_size
        self.ppp         = ppp
        self.bins        = bins
        self.sensor_mode = sensor_mode

        img0 = cv2.imread(samples[0].frame_paths[0], cv2.IMREAD_UNCHANGED)
        self.H, self.W = img0.shape[:2]
        self.grid    = self.make_grid(self.H, self.W, tile_size)
        self.n_tiles = len(self.grid)

        ys = sorted(set(y for _, y, _ in self.grid))
        xs = sorted(set(x for _, _, x in self.grid))

        # Midpoint-boundary stitching (FIXED: use midpoint formula)
        y_bounds = ([0]
                    + [(ys[i] + tile_size + ys[i+1]) // 2 for i in range(len(ys) - 1)]
                    + [self.H])
        x_bounds = ([0]
                    + [(xs[i] + tile_size + xs[i+1]) // 2 for i in range(len(xs) - 1)]
                    + [self.W])

        self.stitch_map = []
        for ri, y0 in enumerate(ys):
            for ci, x0 in enumerate(xs):
                cy0, cy1 = y_bounds[ri], y_bounds[ri + 1]
                cx0, cx1 = x_bounds[ci], x_bounds[ci + 1]
                self.stitch_map.append((cy0, cy1, cx0, cx1,
                                        cy0 - y0, cy1 - y0,
                                        cx0 - x0, cx1 - x0))

        self.items = [(s, nm, y0, x0)
                      for s in samples for nm, y0, x0 in self.grid]

        # Precompute scene-level stats once per scene (not per tile)
        self._alpha_pct99 = {}
        unique_samples = list({s.scene_key: s for s, *_ in self.items}.values())
        print(f"  Precomputing scene stats for {len(unique_samples)} unique scenes…")
        for samp in unique_samples:
            frames = []
            for path in samp.frame_paths[:SEQ_LEN]:
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                frames.append(
                    cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.)
            seq_full = np.stack(frames, 0)
            _, alpha_s, pct99_s = scene_stats(seq_full, ppp, HALF_WIN)
            self._alpha_pct99[samp.scene_key] = (float(alpha_s), float(pct99_s))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        sample, tile_name, y0, x0 = self.items[idx]
        # Use cached scene-level stats
        alpha, pct99 = self._alpha_pct99[sample.scene_key]
        ts = self.ts

        frames = []
        for path in sample.frame_paths[:SEQ_LEN]:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            frames.append(
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.)
        seq = np.stack(frames, 0)   # (T, H, W, 3)

        # Linearise tile crop
        x_lin_tile = np.power(
            np.clip(seq[:, y0:y0+ts, x0:x0+ts, :], 0., 1.), GAMMA
        ).astype(np.float32)
        rgb_p = seq[:, y0:y0+ts, x0:x0+ts, :]

        sim  = simulate_spad(x_lin_tile, alpha, self.bins, HALF_WIN, pct99,
                             sensor_mode=self.sensor_mode, cfa_y0=y0, cfa_x0=x0)
        spad = sim['spad_mono_seq']
        tgt  = sim['target_s']

        return dict(
            spad_mono_seq=torch.from_numpy(spad.copy()),
            clean_rgb_seq=torch.from_numpy(
                rgb_p.transpose(0, 3, 1, 2).astype(np.float32).copy()),
            target_s=torch.from_numpy(tgt.copy()),
            pct99=torch.tensor(pct99, dtype=torch.float32),
            alpha=torch.tensor(float(alpha), dtype=torch.float32),
            scene_key=sample.scene_key,
            tile_name=tile_name,
        )
