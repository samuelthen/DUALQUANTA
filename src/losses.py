"""
Loss functions for SPADNet.

CharbonnierLoss   — smooth L1 approximation
gradient_loss     — L1 on image gradients
reconstruction_loss — combined loss (Charbonnier + optional gradient term)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

PPP = 3.25   # photons-per-pixel; mean(alpha*S) = PPP always


class CharbonnierLoss(nn.Module):
    """Charbonnier (pseudo-Huber) loss: mean( sqrt((p-g)^2 + eps^2) )."""
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))


def gradient_loss(pred, target):
    """L1 loss on horizontal and vertical image gradients."""
    dx_p = pred[..., :, 1:]   - pred[..., :, :-1]
    dy_p = pred[..., 1:, :]   - pred[..., :-1, :]
    dx_t = target[..., :, 1:] - target[..., :, :-1]
    dy_t = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(dx_p, dx_t) + F.l1_loss(dy_p, dy_t)


def reconstruction_loss(pred, target, criterion, use_grad_loss=True):
    """
    Combined reconstruction loss.

    pred, target: (B, 1, H, W) raw photon counts alpha*S.
    Normalise by PPP for scale-invariant loss (mean target ~= 1.0 after norm).

    Loss = Charbonnier(pred/PPP, target/PPP)
           [+ 0.1 * gradient_loss(pred/PPP, target/PPP)]
    """
    p = pred   / PPP
    g = target / PPP
    loss = criterion(p, g)
    if use_grad_loss:
        loss = loss + 0.1 * gradient_loss(p, g)
    return loss
