# DUALQUANTA

Official code for **"Local Information Bounds for Few-Bit Alignment: A Finite-B Cramér–Rao Analysis for Learned SPAD Reconstruction"**.

DUALQUANTA is a two-stage dual-camera system that fuses a monochrome SPAD quanta sensor with an RGGB CMOS sensor to reconstruct high-quality colour video from photon-counting data.

---

## Overview

| Stage | Input | Output |
|-------|-------|--------|
| 1 — SPADNet (mono) | 11 SPAD binary quanta frames (mono) | Denoised luma estimate Ŝ |
| 2 — ColorUNet | RGGB CMOS frame + Ŝ | Final colour image |

The monochrome baseline models (Tables 1 & 2) use Stage 1 only.  DUALQUANTA (Tables 3 & 4) uses both stages.

---

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `torch>=2.0`, `torchvision`, `numpy`, `opencv-python`, `scikit-image`, `lpips`, `pyyaml`, `tqdm`.

---

## Dataset

### i2-2kfps

The primary benchmark.  224 train / 25 val / 31 test scenes at 512×1024, 150 frames per scene (PNG, sRGB gamma-2.2).

Download the dataset and point scripts at the root directory (the one containing `train/`, `val/`, `test/` sub-folders).

### X4K1000FPS

Used for zero-shot evaluation at 2160×4096.  Download from the official X4K1000FPS release and point `eval_x4k_color.py` at the `test/` sub-folder.

---

## Checkpoints

All 23 trained model checkpoints are included in this repository:

```
checkpoints/
  table1_mono/                    # Table 1 — monochrome depth sweep (10 models)
    single_frame_nafnet.pth
    no_align.pth
    dcn_h2.pth  /  dcn_h4.pth  /  dcn_h8.pth  /  dcn_h16.pth
    spynet_dcn.pth  /  cascading_dcn.pth  /  oracle_flow.pth
    quiver_retrained.pth

  table2_bayer/                   # Table 2 — Bayer SPAD depth sweep (6 models)
    no_align.pth
    dcn_h2.pth  /  dcn_h4.pth  /  dcn_h8.pth  /  dcn_h16.pth
    rggb_luma_dcn_h8.pth          # Bayer-input → luma (Stage-1 luma baseline)

  table3_4_DUALQUANTA/            # Tables 3 & 4 — DUALQUANTA (8 files, 7 models)
    stage1_mono_dcn_h8.pth        # Stage-1 backbone (shared)
    cmos_only.pth                 # Ablation: CMOS-only baseline
    DUALQUANTA_T1_stage2.pth  ..  DUALQUANTA_T11_stage2.pth   # T ∈ {1,3,5,7,9,11}
```

---

## Reproducing Paper Results

All evaluation scripts use the same simulation parameters as training:
`B = 7` bins, `λ/B = 3.25` PPP per frame, `T = 11` frames, `γ = 2.2`.

### Table 1 — Monochrome SPAD (luma PSNR)

```bash
# Example: DCN at H/8 (paper default, 37.14 dB)
python evaluate.py \
    --ckpt  checkpoints/table1_mono/dcn_h8.pth \
    --test_root /path/to/i2-2kfps_v1_png/test \
    --sliding_window
```

Replace `dcn_h8.pth` with any checkpoint from `checkpoints/table1_mono/`:

| Checkpoint | Expected PSNR |
|------------|---------------|
| `single_frame_nafnet.pth` | 35.28 |
| `no_align.pth` | 36.53 |
| `dcn_h2.pth` | 36.88 |
| `dcn_h4.pth` | 37.10 |
| `dcn_h8.pth` | 37.14 |
| `dcn_h16.pth` | 36.54 |
| `spynet_dcn.pth` | 37.14 |
| `cascading_dcn.pth` | 37.34 |
| `oracle_flow.pth` | 38.72 |
| `quiver_retrained.pth` | 30.68 |

### Table 2 — Bayer SPAD → RGB

These models take a Bayer (RGGB) SPAD quanta sequence as input and reconstruct a full-colour RGB image (`out_ch=3`, `sensor=rggb`, `target=rgb`).  Alignment depth is the same sweep as Table 1 but with Bayer sensor input.

```bash
python evaluate.py \
    --ckpt        checkpoints/table2_bayer/dcn_h8.pth \
    --test_root   /path/to/i2-2kfps_v1_png/test \
    --sensor_mode rggb \
    --sliding_window
```

| Checkpoint | Alignment | Expected PSNR (RGB) |
|------------|-----------|---------------------|
| `no_align.pth` | none | 33.15 |
| `dcn_h2.pth` | DCN at H/2 | 33.46 |
| `dcn_h4.pth` | DCN at H/4 | 33.69 |
| `dcn_h8.pth` | DCN at H/8 | 33.72 |
| `dcn_h16.pth` | DCN at H/16 | 33.36 |

**Bayer-input luma baseline** (`rggb_luma_dcn_h8.pth`) — RGGB SPAD input with luma output (`out_ch=1`, `target=luma`).  Evaluated with luma PSNR (same metric as Table 1):

```bash
python evaluate.py \
    --ckpt        checkpoints/table2_bayer/rggb_luma_dcn_h8.pth \
    --test_root   /path/to/i2-2kfps_v1_png/test \
    --sensor_mode rggb \
    --sliding_window
```

### Tables 3 & 4 — DUALQUANTA (RGB PSNR)

Use `eval_sliding_lpips.py` which outputs PSNR, SSIM, and LPIPS.

```bash
# DUALQUANTA with T=11 CMOS integration window (Table 3 default, 33.64 dB)
python eval_sliding_lpips.py \
    --ckpt   checkpoints/table3_4_DUALQUANTA/DUALQUANTA_T11_stage2.pth \
    --stage1 checkpoints/table3_4_DUALQUANTA/stage1_mono_dcn_h8.pth \
    --test_root /path/to/i2-2kfps_v1_png/test \
    --cmos_t 11

# CMOS-only ablation (Table 3, 25.79 dB)
python eval_sliding_lpips.py \
    --ckpt   checkpoints/table3_4_DUALQUANTA/cmos_only.pth \
    --stage1 checkpoints/table3_4_DUALQUANTA/stage1_mono_dcn_h8.pth \
    --test_root /path/to/i2-2kfps_v1_png/test
```

Table 4 — CMOS integration window sweep:

| Checkpoint | `--cmos_t` | Expected PSNR |
|------------|-----------|---------------|
| `DUALQUANTA_T1_stage2.pth` | 1 | 33.57 |
| `DUALQUANTA_T3_stage2.pth` | 3 | 34.33 |
| `DUALQUANTA_T5_stage2.pth` | 5 | 34.07 |
| `DUALQUANTA_T7_stage2.pth` | 7 | 33.72 |
| `DUALQUANTA_T9_stage2.pth` | 9 | 33.66 |
| `DUALQUANTA_T11_stage2.pth` | 11 | 33.64 |

### Zero-shot X4K Evaluation (2160×4096)

```bash
python eval_x4k_color.py \
    --ckpt   checkpoints/table3_4_DUALQUANTA/DUALQUANTA_T11_stage2.pth \
    --stage1 checkpoints/table3_4_DUALQUANTA/stage1_mono_dcn_h8.pth \
    --test_root /path/to/X4K1000FPS/test \
    --cmos_t 11
```

### Smoke Test

A fast sanity check (first window per scene, ~2 min per model) that verifies all 23 checkpoints load and produce PSNR in the expected ballpark:

```bash
python smoke_test.py --device cuda:0 --data_root /path/to/i2-2kfps_v1_png
```

---

## Training

### Stage 1 — SPADNet (monochrome)

```bash
# Reproduce Table 1 default (DCN at H/8)
python train.py \
    --config  configs/mono/dcn_h8.yaml \
    --data_root /path/to/i2-2kfps_v1_png \
    --ckpt_dir  runs/dcn_h8

# Bayer RGGB→RGB variant (Table 2, DCN at H/8)
python train.py \
    --config  configs/bayer/dcn_h8.yaml \
    --data_root /path/to/i2-2kfps_v1_png \
    --ckpt_dir  runs/bayer_dcn_h8

# Bayer RGGB→luma variant (supplemental Stage-1 baseline)
python train.py \
    --config  configs/bayer/rggb_luma_dcn_h8.yaml \
    --data_root /path/to/i2-2kfps_v1_png \
    --ckpt_dir  runs/rggb_luma_dcn_h8
```

All config files are in `configs/mono/` and `configs/bayer/`.  CLI flags override YAML values.

### Stage 2 — DUALQUANTA colour

```bash
# Train DUALQUANTA with T=11 CMOS window (Table 3 & 4)
python train_color.py \
    --ablation   3 \
    --config     configs/color/DUALQUANTA_T11.yaml \
    --data_root  /path/to/i2-2kfps_v1_png \
    --stage1_ckpt checkpoints/table3_4_DUALQUANTA/stage1_mono_dcn_h8.pth \
    --ckpt_dir   runs/DUALQUANTA_T11

# CMOS-only ablation (ablation 1)
python train_color.py \
    --ablation   1 \
    --config     configs/color/cmos_only.yaml \
    --data_root  /path/to/i2-2kfps_v1_png \
    --stage1_ckpt checkpoints/table3_4_DUALQUANTA/stage1_mono_dcn_h8.pth \
    --ckpt_dir   runs/cmos_only
```

Ablation modes: `1` = CMOS-only, `2` = CMOS + luma direct, `3` = DUALQUANTA (softmax fusion), `4` = guided alignment.

---

## Repository Structure

```
DUALQUANTA/
  src/
    models/
      net.py              # SPADNet (Stage 1): NAFNet + DCNv2 alignment
      backbone.py         # NAFBlock, DownBlock, UpBlock
      alignment.py        # DCNv2, SpyNet, flow-guided alignment modules
      quiver_model.py     # QUIVER baseline (Chennuri et al. 2024)
    data/
      simulation.py       # scene_stats(), simulate_spad(), simulate_cmos()
      dataset.py          # SPADDataset, TiledDataset, build_splits()
    utils.py              # load_checkpoint(), load_config()
    losses.py             # CharbonnierLoss, reconstruction_loss()
    metrics.py            # psnr(), ssim(), evaluate_batch()
  train.py                # Stage-1 training (mono + Bayer)
  train_color.py          # Stage-2 training (DUALQUANTA colour)
  evaluate.py             # Stage-1 evaluation (Tables 1 & 2)
  eval_sliding_lpips.py   # Stage-2 evaluation (Tables 3 & 4)
  eval_x4k_color.py       # Zero-shot X4K evaluation
  smoke_test.py           # Fast checkpoint sanity check (all 22 models)
  configs/
    mono/                 # 11 configs for Table 1
    bayer/                # 6 configs for Table 2
    color/                # 8 configs for Tables 3 & 4
  checkpoints/
    table1_mono/
    table2_bayer/
    table3_4_DUALQUANTA/
```

---

## Citation

```bibtex
@article{then2025dualquanta,
  title   = {Local Information Bounds for Few-Bit Alignment:
             A Finite-$B$ Cram\'{e}r--Rao Analysis for Learned SPAD Reconstruction},
  author  = {Then, Samuel and others},
  year    = {2025},
}
```

---

## License

MIT License. See `LICENSE` for details.
