"""
Evaluation metrics for SPADNet.

Evaluation protocol:
  1. Normalise raw photon counts by scene-level pct99.
  2. Gamma-compress to display space.
  3. Report PSNR and SSIM in display space.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

GAMMA = 2.2


def to_display(photons: torch.Tensor, pct99: float) -> torch.Tensor:
    """
    Convert raw photon counts to display-space [0, 1].

    Step 1: normalise by scene-level pct99 → pct99-normalised linear [0, 1]
    Step 2: gamma-compress                 → display-space [0, 1]
    """
    return (photons.clamp(0) / pct99).clamp(0, 1) ** (1.0 / GAMMA)


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Peak signal-to-noise ratio (dB). Returns 100.0 for identical inputs."""
    mse = F.mse_loss(pred, target).item()
    return 100.0 if mse == 0 else 10.0 * math.log10(1.0 / mse)


def ssim(pred_np: np.ndarray, target_np: np.ndarray) -> float:
    """Structural similarity (scikit-image implementation)."""
    from skimage.metrics import structural_similarity
    return float(structural_similarity(pred_np, target_np, data_range=1.0))


@torch.no_grad()
def evaluate_batch(pred_raw: torch.Tensor,
                   gt_raw:   torch.Tensor,
                   pct99_batch: torch.Tensor):
    """
    Per-sample metrics for a batch.

    Args:
        pred_raw:    (B, 1, H, W) predicted photon counts
        gt_raw:      (B, 1, H, W) ground-truth photon counts
        pct99_batch: (B,) scene-level pct99 values

    Returns:
        psnr_lin: list of float — PSNR in pct99-normalised linear space
        psnr_gam: list of float — PSNR in display (gamma) space
        ssim_gam: list of float — SSIM in display (gamma) space
    """
    plin, pgam, sgam = [], [], []
    for b in range(pred_raw.shape[0]):
        p99 = pct99_batch[b].item()
        p_r = pred_raw[b, 0].float().cpu()
        g_r = gt_raw  [b, 0].float().cpu()

        # Linear-space PSNR (pct99-normalised, no gamma)
        p_lin = (p_r / p99).clamp(0, 1)
        g_lin = (g_r / p99).clamp(0, 1)
        plin.append(psnr(p_lin, g_lin))

        # Display-space PSNR + SSIM
        p_d = to_display(p_r, p99)
        g_d = to_display(g_r, p99)
        pgam.append(psnr(p_d, g_d))
        sgam.append(ssim(p_d.numpy(), g_d.numpy()))

    return plin, pgam, sgam
