# Alignment Detectability Analysis

The repository contains both parts of the DCN-versus-detectability analysis:

- `plot_offset_energy_vs_dalign.py` computes DCN offset-energy enrichment over 31 scenes.
- `plot_dalign_heatmap.py` computes the physical map and renders the paper heatmap.

Neither script depends on the legacy analysis directory outside this repository. The i2-2kfps data and trained checkpoints remain external inputs.

## Linear clipped heatmap

The paper visualization is a **linear** `inferno` map with display values clipped at the scene's 99th percentile:

```python
imshow(D_align, cmap="inferno", vmin=0, vmax=percentile(D_align, 99))
```

This clipping is display-only. The cached `D_align` values and all decile calculations use the unclipped physical map.

Run the default frame-1 to frame-6 analysis with:

```bash
python plot_dalign_heatmap.py \
  --test-root /path/to/i2-2kfps_v1_png/test \
  --scene test_00008
```

The command writes the heatmap, a JSON summary, and a reusable compressed map cache under `runs/dalign_heatmaps/`. Add `--no-colorbar` for a clean image-only panel. Subsequent runs reuse the RAFT/map cache unless `--refresh` is supplied.

## Physical map

Stored RGB is linearized with gamma 2.2. The photon intensity for the reference frame is

```text
alpha     = 3.25 / mean_{all 11 frames, pixels}(R_linear + G_linear + B_linear)
lambda(x) = alpha * [R_linear(x) + G_linear(x) + B_linear(x)].
```

Frozen torchvision RAFT-large estimates clean-RGB motion from frame 1 to frame 6. With Sobel derivatives divided by 8,

```text
c(x)       = |grad lambda(x)| / [lambda(x) + epsilon]
d_n(x)     = [u(x) grad_x lambda(x) + v(x) grad_y lambda(x)]
             / [|grad lambda(x)| + epsilon]
kappa(x)   = lambda(x)^2 / {B [exp(lambda(x)/B) - 1]}
D_align(x) = [c(x) d_n(x)]^2 kappa(x).
```

The operating constants are `PPP=3.25`, `B=7`, burst length `T=11`, reference frame `1`, and target frame `6`.

## DCN deciles

The DCN script evaluates `dcn_h2`, `dcn_h4`, `dcn_h8`, and `dcn_h16`. At every alignment-grid location, its offset energy is

```text
E_offset(x) = mean over 8 deformable groups and 9 kernel points of (dx^2 + dy^2).
```

Each scene is normalized by its mean offset energy. Full-resolution `D_align` is mean-pooled onto the corresponding DCN grid, then grouped using common decile boundaries for that scale. Scene-level bin means are averaged so large scenes do not masquerade as independent replicates.

```bash
python plot_offset_energy_vs_dalign.py \
  --test-root /path/to/i2-2kfps_v1_png/test \
  --ckpt-dir checkpoints/table1_mono
```

All unclipped maps are cached under `runs/offset_energy_dalign/`; percentile clipping is never used for decile assignment.
