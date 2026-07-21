"""Offset-energy analysis utilities for DUALQUANTA.

This module computes the diagnostic figure that bins learned DCN offset energy
by the theoretical alignment detectability map D_align. It is intentionally
import-safe: no argument parsing, file writes, or model loading happens at
import time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import sobel

from src.data.simulation import scene_stats, simulate_spad
from src.models.net import SPADNet


PPP = 3.25
BINS = 7
SEQ_LEN = 11
HALF_WIN = SEQ_LEN // 2
MODEL_STEMS = ("dcn_h2", "dcn_h4", "dcn_h8", "dcn_h16")
EPS = 1e-8

COLORS = {
    "dcn_h2": "#7c3aed",
    "dcn_h4": "#16a34a",
    "dcn_h8": "#2563eb",
    "dcn_h16": "#dc2626",
}
MARKERS = {
    "dcn_h2": "o",
    "dcn_h4": "s",
    "dcn_h8": "^",
    "dcn_h16": "D",
}
LABELS = {
    "dcn_h2": "DCN H/2",
    "dcn_h4": "DCN H/4",
    "dcn_h8": "DCN H/8",
    "dcn_h16": "DCN H/16",
}


def list_first_window_scenes(test_root: Path, max_scenes: int = 31) -> List[Path]:
    """Return sorted scene directories with at least one 11-frame window."""
    scenes = []
    for path in sorted(Path(test_root).iterdir()):
        if path.is_dir() and len(list(path.glob("*.png"))) >= SEQ_LEN:
            scenes.append(path)
    return scenes[:max_scenes]


def load_rgb_sequence(scene_dir: Path) -> Tuple[np.ndarray, List[Path]]:
    """Load the first 11 RGB frames from a scene at native resolution."""
    paths = sorted(Path(scene_dir).glob("*.png"))[:SEQ_LEN]
    if len(paths) < SEQ_LEN:
        raise RuntimeError(f"Need at least {SEQ_LEN} PNG frames in {scene_dir}")
    frames = [np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0 for path in paths]
    return np.stack(frames, axis=0), paths


def make_spad_sequence(seq_rgb: np.ndarray, seed: int, ppp: float = PPP, bins: int = BINS) -> torch.Tensor:
    """Simulate a mono SPAD burst from an RGB sequence using repo defaults."""
    np.random.seed(seed)
    x_lin, alpha, pct99 = scene_stats(seq_rgb, ppp, HALF_WIN)
    sim = simulate_spad(x_lin, alpha, bins, HALF_WIN, pct99, sensor_mode="mono")
    return torch.from_numpy(sim["spad_mono_seq"].copy()).unsqueeze(0)


def downsample_mean_2d(arr: np.ndarray, factor: int) -> np.ndarray:
    """Power-of-two average pooling for a 2D numpy array."""
    if factor == 1:
        return arr
    if factor & (factor - 1):
        raise ValueError(f"Expected power-of-two factor, got {factor}")
    out = arr.astype(np.float64)
    while factor > 1:
        h, w = out.shape
        if h % 2 or w % 2:
            raise ValueError(f"Cannot downsample odd shape {out.shape}")
        out = out.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
        factor //= 2
    return out


def kappa(lam: np.ndarray, bins: int = BINS) -> np.ndarray:
    """Photon factor used in D_align."""
    em1 = np.expm1(lam / bins)
    em1 = np.where(em1 < 1e-12, 1e-12, em1)
    return (lam**2) / (bins * em1)


def compute_dalign_native(
    raft_model,
    raft_transforms,
    seq_rgb: np.ndarray,
    frame_paths: List[Path],
    neighbor_index: int,
    center_index: int,
    device: torch.device,
    ppp: float = PPP,
    bins: int = BINS,
) -> np.ndarray:
    """Compute native-resolution RAFT-based D_align for one scene pair.

    The reference frame is `neighbor_index`, and motion is estimated toward the
    `center_index` frame. This matches the DCN neighbor-to-center alignment
    convention used by the offset-energy extractor.
    """
    x_lin = np.power(np.clip(seq_rgb, 0.0, 1.0), 2.2).astype(np.float64)
    alpha = ppp / max(float(x_lin.sum(axis=-1).mean()), 1e-8)
    lam = alpha * x_lin[neighbor_index].sum(axis=-1)

    gx = sobel(lam, axis=1) / 8.0
    gy = sobel(lam, axis=0) / 8.0
    grad_mag = np.sqrt(gx**2 + gy**2)
    contrast = grad_mag / (lam + EPS)

    img1 = torch.from_numpy(np.asarray(Image.open(frame_paths[neighbor_index]).convert("RGB")).copy())
    img2 = torch.from_numpy(np.asarray(Image.open(frame_paths[center_index]).convert("RGB")).copy())
    img1 = img1.permute(2, 0, 1).unsqueeze(0)
    img2 = img2.permute(2, 0, 1).unsqueeze(0)
    im1, im2 = raft_transforms(img1, img2)

    with torch.no_grad():
        flow = raft_model(im1.to(device), im2.to(device))[-1][0].float().cpu().numpy()
    u = flow[0].astype(np.float64)
    v = flow[1].astype(np.float64)
    d_n = (u * gx + v * gy) / (grad_mag + EPS)
    return np.clip((contrast * d_n) ** 2 * kappa(lam, bins), 0.0, None)


def get_or_compute_dalign_native(
    cache_dir: Path,
    scene_dir: Path,
    seq_rgb: np.ndarray,
    frame_paths: List[Path],
    neighbor_index: int,
    center_index: int,
    raft_model,
    raft_transforms,
    device: torch.device,
) -> np.ndarray:
    """Load cached native D_align, or compute and cache it."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{scene_dir.name}_f{neighbor_index + 1:06d}_to_f{center_index + 1:06d}_dalign.npz"
    if cache_path.exists():
        return np.load(cache_path)["D_align"]

    d_align = compute_dalign_native(
        raft_model,
        raft_transforms,
        seq_rgb,
        frame_paths,
        neighbor_index,
        center_index,
        device,
    )
    np.savez_compressed(
        cache_path,
        D_align=d_align,
        scene=np.array(scene_dir.name),
        neighbor_index=np.array(neighbor_index),
        center_index=np.array(center_index),
    )
    return d_align


def _offset_energy_from_channels(offsets: torch.Tensor) -> np.ndarray:
    dy = offsets[:, 0::2]
    dx = offsets[:, 1::2]
    energy = (dx.square() + dy.square()).mean(dim=1)[0]
    return energy.detach().float().cpu().numpy()


@torch.no_grad()
def dcn_offset_energy(model: SPADNet, spad_seq: torch.Tensor, neighbor_index: int = HALF_WIN - 1) -> np.ndarray:
    """Extract mean kernel/group DCN offset energy for dcn_h2/h4/h8/h16.

    Returns a 2D map on the model's native alignment grid. The quantity is
    mean_k,g ||Delta p||^2 for the selected neighbor-to-center alignment.
    """
    if neighbor_index == model.C:
        raise ValueError("neighbor_index cannot be the center frame.")

    raw, f0, bsz, t_len, h, w = model._enc_batch(spad_seq)
    if model.mode == "dcn_h2":
        fl = list(f0.reshape(bsz, t_len, model.c1, h // 2, w // 2).unbind(1))
    elif model.mode == "dcn_h4":
        h1, _ = model.dn1(f0)
        fl = list(h1.reshape(bsz, t_len, model.c2, h // 4, w // 4).unbind(1))
    elif model.mode == "dcn_h8":
        h1, _ = model.dn1(f0)
        h2, _ = model.dn2(h1)
        fl = list(h2.reshape(bsz, t_len, model.c3, h // 8, w // 8).unbind(1))
    elif model.mode == "dcn_h16":
        h1, _ = model.dn1(f0)
        h2, _ = model.dn2(h1)
        h3, _ = model.dn3(h2)
        fl = list(h3.reshape(bsz, t_len, model.c4, h // 16, w // 16).unbind(1))
    else:
        raise ValueError(f"Unsupported DCN mode: {model.mode}")

    al = model.align
    offsets = al.oc2(al.lr(al.oc1(torch.cat([fl[neighbor_index], fl[model.C]], 1))))[:, :144]
    return _offset_energy_from_channels(offsets)


def make_decile_edges(values: np.ndarray) -> np.ndarray:
    edges = np.quantile(values, np.linspace(0.0, 1.0, 11)).astype(np.float64)
    for i in range(1, edges.size):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf)
    return edges


def assign_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(values, edges[1:-1], right=False), 0, 9)


def scene_level_decile_means(records: Iterable[Tuple[str, np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute global D_align deciles and scene-level normalized means."""
    records = list(records)
    all_dalign = np.concatenate([r[1] for r in records])
    edges = make_decile_edges(all_dalign)
    scene_decile = np.full((len(records), 10), np.nan, dtype=np.float64)

    for si, (_, d_vals, e_vals) in enumerate(records):
        bins = assign_bins(d_vals, edges)
        for bi in range(10):
            mask = bins == bi
            if np.any(mask):
                scene_decile[si, bi] = float(e_vals[mask].mean())

    mean = np.nanmean(scene_decile, axis=0)
    sem = np.nanstd(scene_decile, axis=0, ddof=1) / np.sqrt(np.sum(np.isfinite(scene_decile), axis=0))
    return mean, sem, scene_decile, edges


def offset_energy_records_for_model(
    model: SPADNet,
    model_stem: str,
    scenes: List[Path],
    scene_data: Dict[str, Tuple[np.ndarray, torch.Tensor]],
    device: torch.device,
    neighbor_index: int = HALF_WIN - 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-decile offset-energy statistics for one DCN model."""
    records = []
    for scene_dir in scenes:
        d_align, spad_cpu = scene_data[scene_dir.name]
        energy = dcn_offset_energy(model, spad_cpu.to(device), neighbor_index)
        factor_y = d_align.shape[0] // energy.shape[0]
        factor_x = d_align.shape[1] // energy.shape[1]
        if factor_y != factor_x or d_align.shape[0] % energy.shape[0] or d_align.shape[1] % energy.shape[1]:
            raise RuntimeError(f"{scene_dir.name}: cannot align D_align {d_align.shape} with energy {energy.shape}")

        d_grid = downsample_mean_2d(d_align, factor_y)
        norm_energy = energy / max(float(energy.mean()), EPS)
        records.append((scene_dir.name, d_grid.reshape(-1), norm_energy.reshape(-1)))

    mean, sem, scene_decile, edges = scene_level_decile_means(records)
    scene_names = np.array([r[0] for r in records])
    return mean, sem, scene_decile, edges, scene_names


def save_model_stats(
    path: Path,
    mean: np.ndarray,
    sem: np.ndarray,
    scene_decile: np.ndarray,
    edges: np.ndarray,
    scene_names: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        mean=mean,
        sem=sem,
        scene_decile=scene_decile,
        edges=edges,
        scene_names=scene_names,
    )


def load_model_stats(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cached = np.load(path)
    return cached["mean"], cached["sem"], cached["scene_decile"], cached["edges"], cached["scene_names"]


def plot_offset_energy_summary(
    results: Dict[str, Dict[str, np.ndarray]],
    out_path: Path,
    model_stems: Tuple[str, ...] = MODEL_STEMS,
) -> None:
    """Plot decile curves plus top-decile enrichment bars."""
    plt.rcParams.update(
        {
            "figure.figsize": (7.2, 3.55),
            "font.size": 10.5,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 1.2,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "mathtext.fontset": "stixsans",
        }
    )
    fig, (ax, ax_bar) = plt.subplots(
        1,
        2,
        gridspec_kw={"width_ratios": [3.3, 1.25], "wspace": 0.32},
        layout="constrained",
    )
    x = np.arange(1, 11)
    enrich, bar_labels, bar_colors = [], [], []
    for stem in model_stems:
        y = results[stem]["mean"]
        sem = results[stem]["sem"]
        color = COLORS[stem]
        ax.plot(x, y, MARKERS[stem] + "-", color=color, linewidth=2.0, markersize=4.8, label=LABELS[stem])
        ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.12, linewidth=0)
        enrich.append(float(y[-1]))
        bar_labels.append(LABELS[stem].replace("DCN ", ""))
        bar_colors.append(color)

    ax.set_xlim(1, 10)
    ax.set_xticks(x)
    ax.set_xlabel(r"$D_{\mathrm{align}}$ decile")
    ax.set_ylabel("normalized DCN offset energy")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(False)

    xpos = np.arange(len(enrich))
    ax_bar.bar(xpos, enrich, color=bar_colors, width=0.68)
    ax_bar.set_xticks(xpos)
    ax_bar.set_xticklabels(bar_labels, rotation=35, ha="right")
    ax_bar.set_ylabel("top-decile enrichment")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    for xi, val in zip(xpos, enrich):
        ax_bar.text(xi, val, f"{val:.1f}x", ha="center", va="bottom", fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
