# DUALQUANTA

Official implementation of:

> **Local Information Bounds for Few-Bit Alignment: A Finite-B Cramér–Rao Analysis for Learned SPAD Reconstruction**

DUALQUANTA reconstructs high-quality colour video from photon-counting data by fusing a monochrome SPAD quanta sensor with a conventional RGGB CMOS sensor.

---

## System Overview

| Stage | Input | Output |
|-------|-------|--------|
| 1 — SPADNet | T=11 SPAD binary frames (mono) | Denoised luma estimate Ŝ |
| 2 — ColorUNet | RGGB CMOS frame + Ŝ | Reconstructed colour image |

Tables 1 & 2 use Stage 1 only. Tables 3 & 4 use both stages together.

Simulation parameters (fixed across all experiments):
`B = 7` histogram bins · `λ/B = 3.25` PPP/frame · `T = 11` frames · `γ = 2.2`

---

## Quick Start

```bash
# 1. Clone the code
git clone https://github.com/samuelthen/DUALQUANTA.git
cd DUALQUANTA

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download checkpoints from Hugging Face
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="samuelthen/DUALQUANTA",
                  local_dir=".",
                  ignore_patterns=["*.md", "*.gitattributes"])
EOF

# 4. Sanity-check all 23 checkpoints (first window per scene, ~2 min/model)
python smoke_test.py --device cuda:0 --data_root /path/to/i2-2kfps_v1_png

# 5. Reproduce Table 1 default result (DCN at H/8, 37.14 dB)
python evaluate.py \
    --ckpt      checkpoints/table1_mono/dcn_h8.pth \
    --test_root /path/to/i2-2kfps_v1_png/test \
    --sliding_window

# 6. Reproduce DUALQUANTA result (Table 3, 33.64 dB RGB PSNR)
python eval_sliding_lpips.py \
    --ckpt      checkpoints/table3_4_DUALQUANTA/DUALQUANTA_T11_stage2.pth \
    --stage1    checkpoints/table3_4_DUALQUANTA/stage1_mono_dcn_h8.pth \
    --test_root /path/to/i2-2kfps_v1_png/test \
    --cmos_t    11
```

---

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `torch>=2.0`, `torchvision`, `numpy`, `opencv-python`, `scikit-image`, `lpips`, `pyyaml`, `tqdm`, `huggingface_hub`.

### Alignment analysis

The repository includes self-contained CLIs for the 31-scene DCN offset-energy analysis and the paper's linear p99-clipped `D_align` heatmap. See [`ANALYSIS_DALIGN.md`](ANALYSIS_DALIGN.md) for the equations, conventions, cache layout, and reproduction commands.

---

## Datasets

### i2-2kfps (primary benchmark)

224 train / 25 val / 31 test scenes · 512×1024 · 150 frames/scene · PNG · sRGB (γ=2.2)

Download from the official i2-2kfps release and place it so the root contains `train/`, `val/`, `test/` sub-folders.

### X4K1000FPS (zero-shot generalisation)

2160×4096 · used only for evaluation, not training.

Download from the official X4K1000FPS release and point `eval_x4k_color.py` at the `test/` sub-folder.

---

## Checkpoints

All 23 trained checkpoints are hosted on Hugging Face:
**[samuelthen/DUALQUANTA](https://huggingface.co/samuelthen/DUALQUANTA)**

### Downloading checkpoints

**All at once (recommended):**
```bash
# CLI
huggingface-cli download samuelthen/DUALQUANTA \
    --local-dir . --include "checkpoints/**"

# Python
from huggingface_hub import snapshot_download
snapshot_download("samuelthen/DUALQUANTA", local_dir=".",
                  ignore_patterns=["*.md", "*.gitattributes"])
```

**Single file:**
```python
from huggingface_hub import hf_hub_download
path = hf_hub_download("samuelthen/DUALQUANTA",
                        "checkpoints/table1_mono/dcn_h8.pth")
```

### Checkpoint layout

```
checkpoints/
  table1_mono/               # Table 1 — mono SPAD depth sweep
    single_frame_nafnet.pth  # 35.28 dB
    no_align.pth             # 36.53 dB
    dcn_h2.pth               # 36.88 dB
    dcn_h4.pth               # 37.10 dB
    dcn_h8.pth               # 37.14 dB  ← paper default
    dcn_h16.pth              # 36.54 dB
    spynet_dcn.pth           # 37.14 dB
    cascading_dcn.pth        # 37.34 dB
    oracle_flow.pth          # 38.72 dB  (requires GT frames)
    quiver_retrained.pth     # 30.68 dB  (QUIVER baseline, 627 MB)

  table2_bayer/              # Table 2 — Bayer SPAD depth sweep (RGGB→RGB)
    no_align.pth             # 33.15 dB
    dcn_h2.pth               # 33.46 dB
    dcn_h4.pth               # 33.69 dB
    dcn_h8.pth               # 33.72 dB
    dcn_h16.pth              # 33.36 dB
    rggb_luma_dcn_h8.pth     # 35.38 dB  (RGGB→luma luma PSNR, SSIM 0.896)

  table3_4_DUALQUANTA/       # Tables 3 & 4 — full two-stage system
    stage1_mono_dcn_h8.pth   # Stage-1 backbone (shared by all Stage-2 models)
    cmos_only.pth            # 25.79 dB  (CMOS-only ablation, Table 3)
    DUALQUANTA_T1_stage2.pth   # 33.57 dB  (T=1)
    DUALQUANTA_T3_stage2.pth   # 34.33 dB  (T=3)
    DUALQUANTA_T5_stage2.pth   # 34.07 dB  (T=5)
    DUALQUANTA_T7_stage2.pth   # 33.72 dB  (T=7)
    DUALQUANTA_T9_stage2.pth   # 33.66 dB  (T=9)
    DUALQUANTA_T11_stage2.pth  # 33.64 dB  (T=11) ← Table 3 default
```

---

## Reproducing Paper Results

### Table 1 — Monochrome SPAD (luma PSNR)

```bash
python evaluate.py \
    --ckpt      checkpoints/table1_mono/<checkpoint>.pth \
    --test_root /path/to/i2-2kfps_v1_png/test \
    --sliding_window
```

| Checkpoint | PSNR |
|---|---|
| `single_frame_nafnet.pth` | 35.28 |
| `no_align.pth` | 36.53 |
| `dcn_h2.pth` | 36.88 |
| `dcn_h4.pth` | 37.10 |
| `dcn_h8.pth` | **37.14** |
| `dcn_h16.pth` | 36.54 |
| `spynet_dcn.pth` | 37.14 |
| `cascading_dcn.pth` | 37.34 |
| `oracle_flow.pth` | 38.72 |
| `quiver_retrained.pth` | 30.68 |

> `oracle_flow.pth` requires clean ground-truth frames for RAFT optical flow and cannot be run with `evaluate.py` alone. Use a full evaluation script that provides clean frames.

### Table 2 — Bayer SPAD → RGB (RGB PSNR)

RGGB SPAD input, full-colour RGB output (`sensor=rggb`, `target=rgb`, `out_ch=3`):

```bash
python evaluate.py \
    --ckpt        checkpoints/table2_bayer/<checkpoint>.pth \
    --test_root   /path/to/i2-2kfps_v1_png/test \
    --sensor_mode rggb \
    --sliding_window
```

| Checkpoint | Alignment | PSNR |
|---|---|---|
| `no_align.pth` | none | 33.15 |
| `dcn_h2.pth` | DCN H/2 | 33.46 |
| `dcn_h4.pth` | DCN H/4 | 33.69 |
| `dcn_h8.pth` | DCN H/8 | **33.72** |
| `dcn_h16.pth` | DCN H/16 | 33.36 |

**Bayer→luma baseline** (`rggb_luma_dcn_h8.pth`, RGGB input / luma output, luma PSNR):

```bash
python evaluate.py \
    --ckpt        checkpoints/table2_bayer/rggb_luma_dcn_h8.pth \
    --test_root   /path/to/i2-2kfps_v1_png/test \
    --sensor_mode rggb \
    --sliding_window
# Expected: 35.38 dB luma PSNR / SSIM 0.896
```

### Tables 3 & 4 — DUALQUANTA (RGB PSNR + SSIM + LPIPS)

```bash
python eval_sliding_lpips.py \
    --ckpt      checkpoints/table3_4_DUALQUANTA/<stage2_checkpoint>.pth \
    --stage1    checkpoints/table3_4_DUALQUANTA/stage1_mono_dcn_h8.pth \
    --test_root /path/to/i2-2kfps_v1_png/test \
    --cmos_t    <T>
```

**Table 3 — sensing comparison:**

| Model | `--ckpt` | `--cmos_t` | PSNR |
|---|---|---|---|
| CMOS-only | `cmos_only.pth` | 11 | 25.79 |
| DUALQUANTA | `DUALQUANTA_T11_stage2.pth` | 11 | **33.64** |

**Table 4 — CMOS integration window sweep:**

| `--ckpt` | `--cmos_t` | PSNR |
|---|---|---|
| `DUALQUANTA_T1_stage2.pth` | 1 | 33.57 |
| `DUALQUANTA_T3_stage2.pth` | 3 | **34.33** |
| `DUALQUANTA_T5_stage2.pth` | 5 | 34.07 |
| `DUALQUANTA_T7_stage2.pth` | 7 | 33.72 |
| `DUALQUANTA_T9_stage2.pth` | 9 | 33.66 |
| `DUALQUANTA_T11_stage2.pth` | 11 | 33.64 |

### Zero-shot X4K (2160×4096)

```bash
python eval_x4k_color.py \
    --ckpt      checkpoints/table3_4_DUALQUANTA/DUALQUANTA_T11_stage2.pth \
    --stage1    checkpoints/table3_4_DUALQUANTA/stage1_mono_dcn_h8.pth \
    --test_root /path/to/X4K1000FPS/test \
    --cmos_t    11
```

### Smoke Test

Runs all 23 checkpoints on the first temporal window of each test scene (~2 min/model). Useful as a quick sanity check after downloading:

```bash
python smoke_test.py --device cuda:0 --data_root /path/to/i2-2kfps_v1_png
```

Expected output: all models PASS (within ±1.5 dB of paper values) except `oracle_flow` (SKIP — requires GT frames) and `quiver_retrained` (WARN — first-window bias within ±2.5 dB).

---

## Training

### Stage 1 — SPADNet

```bash
# Monochrome (Table 1 default: DCN at H/8)
python train.py \
    --config    configs/mono/dcn_h8.yaml \
    --data_root /path/to/i2-2kfps_v1_png \
    --ckpt_dir  runs/dcn_h8

# Bayer RGGB→RGB (Table 2)
python train.py \
    --config    configs/bayer/dcn_h8.yaml \
    --data_root /path/to/i2-2kfps_v1_png \
    --ckpt_dir  runs/bayer_dcn_h8

# Bayer RGGB→luma (Stage-1 luma baseline)
python train.py \
    --config    configs/bayer/rggb_luma_dcn_h8.yaml \
    --data_root /path/to/i2-2kfps_v1_png \
    --ckpt_dir  runs/rggb_luma_dcn_h8
```

All configs are in `configs/mono/` and `configs/bayer/`. CLI flags override YAML values.

### Stage 2 — ColorUNet

```bash
# DUALQUANTA T=11 (Table 3 & 4)
python train_color.py \
    --ablation    3 \
    --config      configs/color/DUALQUANTA_T11.yaml \
    --data_root   /path/to/i2-2kfps_v1_png \
    --stage1_ckpt checkpoints/table3_4_DUALQUANTA/stage1_mono_dcn_h8.pth \
    --ckpt_dir    runs/DUALQUANTA_T11

# CMOS-only ablation
python train_color.py \
    --ablation    1 \
    --config      configs/color/cmos_only.yaml \
    --data_root   /path/to/i2-2kfps_v1_png \
    --stage1_ckpt checkpoints/table3_4_DUALQUANTA/stage1_mono_dcn_h8.pth \
    --ckpt_dir    runs/cmos_only
```

Ablation modes: `1` = CMOS-only · `2` = CMOS + luma direct · `3` = DUALQUANTA (softmax fusion) · `4` = guided alignment.

---

## Repository Structure

```
DUALQUANTA/
  src/
    models/
      net.py              # SPADNet (Stage 1): NAFNet backbone + DCNv2 alignment
      backbone.py         # NAFBlock, DownBlock, UpBlock
      alignment.py        # DCNv2, SpyNet, flow-guided alignment
      quiver_model.py     # QUIVER baseline (Chennuri et al. 2024)
    data/
      simulation.py       # scene_stats(), simulate_spad(), simulate_cmos()
      dataset.py          # SPADDataset, TiledDataset, build_splits()
    utils.py              # load_checkpoint(), load_config()
    losses.py             # CharbonnierLoss, reconstruction_loss()
    metrics.py            # psnr(), ssim(), evaluate_batch()
  configs/
    mono/                 # 10 configs for Table 1 variants
    bayer/                # 6 configs for Table 2 variants
    color/                # 8 configs for Tables 3 & 4 variants
  train.py                # Stage-1 training (mono + Bayer)
  train_color.py          # Stage-2 training (DUALQUANTA colour)
  evaluate.py             # Stage-1 evaluation (Tables 1 & 2)
  eval_sliding_lpips.py   # Stage-2 evaluation (Tables 3 & 4)
  eval_x4k_color.py       # Zero-shot X4K evaluation
  smoke_test.py           # Fast sanity check across all 23 checkpoints
  checkpoints/            # Downloaded from HuggingFace (not in git)
```

---

## Citation

```bibtex
@article{then2025dualquanta,
  title   = {Local Information Bounds for Few-Bit Alignment:
             A Finite-$B$ Cram\'{e}r--Rao Analysis for Learned {SPAD} Reconstruction},
  author  = {Then, Samuel and others},
  year    = {2025},
}
```

---

## License

MIT License. See `LICENSE` for details.
