#!/usr/bin/env python3
"""Compute and plot a linear, percentile-clipped D_align heatmap.

The script is self-contained within DUALQUANTA: it loads an i2-2kfps scene,
runs frozen torchvision RAFT on a clean RGB pair, computes Eq. 2 using the same
exposure convention as the repository's SPAD simulator, caches all physical
maps, and writes the publication heatmap plus a JSON summary.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import numpy as np
import torch
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

from src.analysis.offset_energy import (
    BINS,
    PPP,
    compute_dalign_components,
    load_rgb_sequence,
    plot_dalign_linear_clipped,
)


DEFAULT_TEST_ROOT = ROOT / "data" / "i2-2kfps" / "test"
DEFAULT_OUT_DIR = ROOT / "runs" / "dalign_heatmaps"
DEFAULT_SCENE = "test_00008"
DEFAULT_REFERENCE_FRAME = 1
DEFAULT_TARGET_FRAME = 6
DEFAULT_PERCENTILE = 99.0
DEFAULT_GAMMA = 2.2


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-root", type=Path, default=DEFAULT_TEST_ROOT)
    parser.add_argument("--scene", default=DEFAULT_SCENE, help="Scene name or zero-padded numeric ID.")
    parser.add_argument("--reference-frame", type=int, default=DEFAULT_REFERENCE_FRAME, help="One-based frame number.")
    parser.add_argument("--target-frame", type=int, default=DEFAULT_TARGET_FRAME, help="One-based frame number.")
    parser.add_argument("--ppp", type=float, default=PPP)
    parser.add_argument("--bins", type=int, default=BINS)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--clip-percentile", type=float, default=DEFAULT_PERCENTILE)
    parser.add_argument("--device", default=auto_device())
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--no-colorbar", action="store_true")
    parser.add_argument("--refresh", action="store_true", help="Ignore a compatible cached map.")
    return parser.parse_args()


def normalize_scene_name(scene: str) -> str:
    return scene if scene.startswith("test_") else f"test_{int(scene):05d}"


def cache_is_compatible(cache: np.lib.npyio.NpzFile, args: argparse.Namespace, scene: str) -> bool:
    expected = {
        "scene": scene,
        "reference_frame": args.reference_frame,
        "target_frame": args.target_frame,
        "ppp": args.ppp,
        "bins": args.bins,
        "gamma": args.gamma,
    }
    for key, value in expected.items():
        if key not in cache:
            return False
        actual = cache[key].item()
        if isinstance(value, float):
            if not np.isclose(actual, value):
                return False
        elif actual != value:
            return False
    return True


def array_stats(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values)[np.isfinite(values)]
    return {
        "min": float(finite.min()),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(finite.max()),
    }


def main() -> None:
    args = parse_args()
    scene = normalize_scene_name(args.scene)
    scene_dir = args.test_root / scene
    if not scene_dir.is_dir():
        raise FileNotFoundError(
            f"Scene not found: {scene_dir}. Pass the directory containing test_XXXXX folders with --test-root."
        )

    seq_rgb, frame_paths = load_rgb_sequence(scene_dir)
    ref_index = args.reference_frame - 1
    target_index = args.target_frame - 1
    if not 0 <= ref_index < len(frame_paths) or not 0 <= target_index < len(frame_paths):
        raise ValueError(f"Frames must lie within the first {len(frame_paths)} images of {scene}.")

    stem = f"{scene}_f{args.reference_frame:06d}_to_f{args.target_frame:06d}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.out_dir / "cache" / f"{stem}_dalign_maps.npz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    maps = None
    if cache_path.exists() and not args.refresh:
        with np.load(cache_path) as cached:
            if cache_is_compatible(cached, args, scene):
                maps = {key: cached[key] for key in cached.files if key not in {
                    "scene", "reference_frame", "target_frame", "ppp", "bins", "gamma"
                }}
                print(f"Loaded cached maps: {cache_path}", flush=True)

    if maps is None:
        device = torch.device(args.device)
        weights = Raft_Large_Weights.DEFAULT
        raft_model = raft_large(weights=weights).to(device).eval()
        maps = compute_dalign_components(
            raft_model,
            weights.transforms(),
            seq_rgb,
            frame_paths,
            ref_index,
            target_index,
            device,
            ppp=args.ppp,
            bins=args.bins,
            gamma=args.gamma,
        )
        np.savez_compressed(
            cache_path,
            **maps,
            scene=np.asarray(scene),
            reference_frame=np.asarray(args.reference_frame),
            target_frame=np.asarray(args.target_frame),
            ppp=np.asarray(args.ppp),
            bins=np.asarray(args.bins),
            gamma=np.asarray(args.gamma),
        )
        print(f"Saved physical maps: {cache_path}", flush=True)

    clip_tag = f"p{args.clip_percentile:g}".replace(".", "p")
    suffix = "_no_colorbar" if args.no_colorbar else ""
    figure_path = args.out_dir / f"{stem}_dalign_linear_{clip_tag}_clipped{suffix}.png"
    display_vmax = plot_dalign_linear_clipped(
        maps["D_align"],
        figure_path,
        percentile=args.clip_percentile,
        colorbar=not args.no_colorbar,
    )

    flow_mag = np.hypot(maps["flow_u"], maps["flow_v"])
    summary = {
        "scene": scene,
        "reference_frame": args.reference_frame,
        "target_frame": args.target_frame,
        "frame_direction": f"{args.reference_frame}->{args.target_frame}",
        "ppp": args.ppp,
        "bins": args.bins,
        "gamma": args.gamma,
        "alpha": float(np.asarray(maps["alpha"])),
        "display": {
            "normalization": "linear",
            "clip_percentile": args.clip_percentile,
            "vmin": 0.0,
            "vmax": display_vmax,
            "colormap": "inferno",
        },
        "D_align": array_stats(maps["D_align"]),
        "flow_magnitude_px": array_stats(flow_mag),
        "normal_displacement_abs_px": array_stats(np.abs(maps["d_n"])),
        "outputs": {
            "figure": str(figure_path.resolve()),
            "maps": str(cache_path.resolve()),
        },
    }
    stats_path = args.out_dir / f"{stem}_dalign_stats.json"
    stats_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")

    print(f"D_align p99: {summary['D_align']['p99']:.6g}", flush=True)
    print(f"Linear display range: [0, {display_vmax:.6g}]", flush=True)
    print(f"Saved figure: {figure_path}", flush=True)
    print(f"Saved stats: {stats_path}", flush=True)


if __name__ == "__main__":
    main()
