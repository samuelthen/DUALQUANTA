#!/usr/bin/env python3
"""Plot DCN offset energy vs. D_align deciles.

This is the repo-facing CLI for the analysis helpers in
`src.analysis.offset_energy`. It evaluates the four mono DCN checkpoints
(`dcn_h2`, `dcn_h4`, `dcn_h8`, `dcn_h16`) on one native-resolution frame pair
per test scene, then writes:

  runs/offset_energy_dalign/offset_energy_vs_dalign_31scenes_all_dcn.png
  runs/offset_energy_dalign/offset_energy_vs_dalign_31scenes_all_dcn_stats.npz

The default pair is frame 000001 -> 000006, matching the 11-frame burst's
left neighbor to center-frame alignment convention.
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

from src.analysis.offset_energy import (
    HALF_WIN,
    LABELS,
    MODEL_STEMS,
    get_or_compute_dalign_native,
    list_first_window_scenes,
    load_model_stats,
    load_rgb_sequence,
    make_spad_sequence,
    offset_energy_records_for_model,
    plot_offset_energy_summary,
    save_model_stats,
)
from src.models.net import SPADNet
from src.utils import load_checkpoint, load_config


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent
HF_REPO_ID = "samuelthen/DUALQUANTA"


def parse_args() -> argparse.Namespace:
    out_dir = ROOT / "runs" / "offset_energy_dalign"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test-root", type=Path, default=WORKSPACE_ROOT / "i2k_test")
    p.add_argument("--max-scenes", type=int, default=31)
    p.add_argument("--neighbor-index", type=int, default=0)
    p.add_argument("--center-index", type=int, default=HALF_WIN)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--ckpt-dir", type=Path, default=ROOT / "checkpoints" / "table1_mono")
    p.add_argument("--hf-repo-id", default=HF_REPO_ID)
    p.add_argument("--out-dir", type=Path, default=out_dir)
    p.add_argument("--cache-dir", type=Path, default=out_dir / "dalign_native_pair_cache")
    p.add_argument("--model-cache-dir", type=Path, default=out_dir / "model_cache")
    p.add_argument("--out", type=Path, default=out_dir / "offset_energy_vs_dalign_31scenes_all_dcn.png")
    p.add_argument("--stats-out", type=Path, default=out_dir / "offset_energy_vs_dalign_31scenes_all_dcn_stats.npz")
    return p.parse_args()


def resolve_checkpoint(model_stem: str, ckpt_dir: Path, hf_repo_id: str) -> Path:
    local_path = ckpt_dir / f"{model_stem}.pth"
    if local_path.exists():
        return local_path

    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=hf_repo_id,
            filename=f"checkpoints/table1_mono/{model_stem}.pth",
        )
    )


def build_model(model_stem: str, args: argparse.Namespace, device: torch.device) -> SPADNet:
    cfg = load_config(str(ROOT / "configs" / "mono" / f"{model_stem}.yaml"))["model"]
    model = SPADNet(
        T=cfg.get("T", 11),
        bc=cfg.get("base_channels", 32),
        nb=cfg.get("n_blocks", 2),
        nfpm=cfg.get("n_fpm", 2),
        mode=cfg["mode"],
        raft_ckpt=cfg.get("raft_ckpt", ""),
    ).to(device)
    ckpt = load_checkpoint(str(resolve_checkpoint(model_stem, args.ckpt_dir, args.hf_repo_id)), device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model


def save_summary_stats(
    path: Path,
    results: Dict[str, Dict[str, np.ndarray]],
    scenes: list[Path],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        model_stems=np.array(MODEL_STEMS),
        scene_names=np.array([s.name for s in scenes]),
        **{f"{stem}_mean": results[stem]["mean"] for stem in MODEL_STEMS},
        **{f"{stem}_sem": results[stem]["sem"] for stem in MODEL_STEMS},
        **{f"{stem}_scene_decile": results[stem]["scene_decile"] for stem in MODEL_STEMS},
        **{f"{stem}_decile_edges": results[stem]["edges"] for stem in MODEL_STEMS},
    )


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    scenes = list_first_window_scenes(args.test_root, args.max_scenes)
    if not scenes:
        raise RuntimeError(f"No 11-frame scenes found under {args.test_root}")
    print(f"Using {len(scenes)} native-resolution scenes.", flush=True)

    weights = Raft_Large_Weights.DEFAULT
    raft_transforms = weights.transforms()
    raft_model = raft_large(weights=weights).to(device).eval()

    scene_data = {}
    for i, scene_dir in enumerate(scenes):
        seq_rgb, frame_paths = load_rgb_sequence(scene_dir)
        d_align = get_or_compute_dalign_native(
            args.cache_dir,
            scene_dir,
            seq_rgb,
            frame_paths,
            args.neighbor_index,
            args.center_index,
            raft_model,
            raft_transforms,
            device,
        )
        spad = make_spad_sequence(seq_rgb, args.seed + i)
        scene_data[scene_dir.name] = (d_align, spad)
        print(f"[D_align] {i + 1:02d}/{len(scenes):02d} {scene_dir.name} native={d_align.shape}", flush=True)

    del raft_model
    gc.collect()

    args.model_cache_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict[str, np.ndarray]] = {}
    for stem in MODEL_STEMS:
        model_cache = args.model_cache_dir / f"{stem}_native{len(scenes)}.npz"
        if model_cache.exists():
            mean, sem, scene_decile, edges, scene_names = load_model_stats(model_cache)
            print(f"[{stem}] loaded cached model stats: {model_cache}", flush=True)
        else:
            model = build_model(stem, args, device)
            mean, sem, scene_decile, edges, scene_names = offset_energy_records_for_model(
                model,
                stem,
                scenes,
                scene_data,
                device,
                neighbor_index=args.neighbor_index,
            )
            del model
            gc.collect()
            save_model_stats(model_cache, mean, sem, scene_decile, edges, scene_names)
            print(f"[{stem}] saved model stats: {model_cache}", flush=True)

        results[stem] = {
            "mean": mean,
            "sem": sem,
            "scene_decile": scene_decile,
            "edges": edges,
            "scene_names": scene_names,
        }
        print(f"{LABELS[stem]} mean: {np.array2string(mean, precision=4)}", flush=True)
        print(f"{LABELS[stem]} top-decile enrichment: {mean[-1]:.3f}x", flush=True)

    plot_offset_energy_summary(results, args.out)
    save_summary_stats(args.stats_out, results, scenes)
    print(f"Saved figure: {args.out}", flush=True)
    print(f"Saved stats: {args.stats_out}", flush=True)


if __name__ == "__main__":
    main()
