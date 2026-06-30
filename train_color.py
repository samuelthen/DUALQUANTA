#!/usr/bin/env python3
"""
train_color.py — RGB colour ablations on the spad-net U-Net.

Three ablation variants that isolate how much colour information
can be recovered from CMOS alone vs. CMOS + SPAD luma:

  Ablation 1  (--ablation 1)
    Input:   blurred CMOS packed RGGB  (4ch, H/2 × W/2)
    Head:    3-ch softplus  →  (alpha*R_lin, alpha*G_lin, alpha*B_lin)
    No SPAD path.

  Ablation 2  (--ablation 2)
    Input:   CMOS (4ch) + SPAD luma (1ch, downsampled to H/2)  →  5ch
    Head:    3-ch softplus  →  (alpha*R_lin, alpha*G_lin, alpha*B_lin)
    Luma:    stage-1 SPADNet output (--stage1_ckpt) or GT alpha*S.

  Ablation 3  (--ablation 3)
    Input:   CMOS (4ch) + SPAD luma (1ch, downsampled to H/2)  →  5ch
    Head:    3-ch softmax (colour fractions r, g, b)
             × full-res SPAD luma  →  (alpha*R_lin, alpha*G_lin, alpha*B_lin)
    Luma:    stage-1 SPADNet output (--stage1_ckpt) or GT alpha*S.

Loss: Charbonnier only on pred/PPP vs target/PPP (identical scale to mono).

Usage:
    python train_color.py --ablation 1 --data_root /data/X4K/train --ckpt_dir runs/color_a1
    python train_color.py --ablation 2 --stage1_ckpt runs/dcn_h4/best.pth ...
    python train_color.py --ablation 3 --stage1_ckpt runs/dcn_h4/best.pth ...
"""

import argparse
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

import cv2
cv2.setNumThreads(0)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import wandb
    WANDB_OK = True
except ImportError:
    WANDB_OK = False

from src.data.dataset import _worker_init, _scenes, _pngs, Sample, build_splits, build_test
from src.data.simulation import scene_stats, simulate_spad, simulate_cmos
from src.models.backbone import NAFBlock, DownBlock, UpBlock
from src.models.net import SPADNet
from src.models.alignment import DCNv2
from src.losses import CharbonnierLoss
from src.utils import load_config, save_checkpoint, load_checkpoint

_TTY = sys.stderr.isatty()

SEQ_LEN  = 11
HALF_WIN = SEQ_LEN // 2
PPP      = 3.25
BINS     = 7
GAMMA    = 2.2
CMOS_SIGMA = 2.0

ABLATION_MODES = {
    1: "cmos_only",
    2: "cmos_luma_direct",
    3: "cmos_luma_softmax",
    4: "cmos_guided_align",
}


# ── dataset ───────────────────────────────────────────────────────────────────

def _read_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.


def _augment_patch(x_lin: np.ndarray, rng: random.Random) -> np.ndarray:
    """Consistent spatial + temporal augmentation on linearised (T, H, W, 3) patch."""
    if rng.random() < 0.5:
        k = rng.randint(1, 3)
        x_lin = np.rot90(x_lin, k=k, axes=(1, 2)).copy()
    if rng.random() < 0.5:
        x_lin = x_lin[:, :, ::-1, :].copy()
    if rng.random() < 0.5:
        x_lin = x_lin[:, ::-1, :, :].copy()
    if rng.random() < 0.5:
        x_lin = x_lin[::-1].copy()
    return x_lin


class CMOSColorDataset(Dataset):
    """
    Loads RGB sequences and simulates both SPAD (mono) and CMOS (colour).

    Batch items:
        spad_mono_seq : (T, H, W)    SPAD binary frames in [0, 1]
        cmos_packed   : (4, H/2, W/2) packed RGGB Bayer in [0, 1]
        target_rgb_s  : (3, H, W)    alpha * [R_lin, G_lin, B_lin]  (training target)
        target_s      : (H, W)       alpha * S  (mono target, used as GT luma)
        pct99         : float        scene-level 99th-percentile of alpha*S
        alpha         : float        exposure gain
        scene         : str
    """

    def __init__(self, samples: List[Sample],
                 ppp: float = PPP, bins: int = BINS, cmos_sigma: float = CMOS_SIGMA,
                 augment: bool = False, crop: int = 256, eval_crop: int = 256,
                 seed: int = 42, random_pick: bool = False, sps: int = 1,
                 cmos_t: int = 1):
        self.samples     = samples
        self.ppp         = ppp
        self.bins        = bins
        self.cmos_sigma  = cmos_sigma
        self.augment     = augment
        self.crop        = crop
        self.eval_crop   = eval_crop
        self.random_pick = random_pick
        self.sps         = max(1, int(sps))
        self.rng         = random.Random(seed)
        self.cmos_t      = int(cmos_t)

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
        return np.stack([fallback if f is None else f for f in frames], 0), nfb

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

        # Scene stats on full frame (alpha, pct99 independent of crop)
        x_lin_full, alpha, pct99_val = scene_stats(seq, self.ppp, HALF_WIN)

        # Crop
        if self.augment:
            y0 = self.rng.randint(0, H - self.crop)
            x0 = self.rng.randint(0, W - self.crop)
            ch = cw = self.crop
        elif self.eval_crop > 0:
            y0 = (H - self.eval_crop) // 2
            x0 = (W - self.eval_crop) // 2
            ch = cw = self.eval_crop
        else:
            y0 = x0 = 0; ch = H; cw = W

        x_lin_patch = x_lin_full[:, y0:y0+ch, x0:x0+cw, :].copy()

        # Augment on linearised patch before simulation (keeps Bayer pattern valid)
        if self.augment:
            x_lin_patch = _augment_patch(x_lin_patch, self.rng)

        # SPAD simulation
        spad_sim = simulate_spad(x_lin_patch, alpha, self.bins, HALF_WIN, pct99_val)
        spad     = spad_sim["spad_mono_seq"]   # (T, h, w)
        target_s = spad_sim["target_s"]        # (h, w)  alpha*S

        # CMOS simulation — integrates self.cmos_t frames centred at half_win.
        # cmos_t=1: single center frame (baseline); cmos_t>1: longer exposure.
        # Full T=11 patch is passed; simulate_cmos selects the cmos_t window.
        cmos_sim    = simulate_cmos(x_lin_patch, alpha, self.cmos_sigma, cmos_t=self.cmos_t)
        cmos_packed = cmos_sim["cmos_packed"]  # (4, h/2, w/2)

        # Colour target: alpha * [R_lin, G_lin, B_lin] at centre frame
        target_rgb_s = (x_lin_patch[HALF_WIN] * alpha).transpose(2, 0, 1).astype(np.float32)

        return dict(
            spad_mono_seq=torch.from_numpy(spad.copy()),
            cmos_packed=torch.from_numpy(cmos_packed),
            target_rgb_s=torch.from_numpy(target_rgb_s),
            target_s=torch.from_numpy(target_s.copy()),
            pct99=torch.tensor(pct99_val, dtype=torch.float32),
            alpha=torch.tensor(float(alpha), dtype=torch.float32),
            scene=s.scene_key,
        )


# ── model ─────────────────────────────────────────────────────────────────────

class ColorUNet(nn.Module):
    """
    spad-net U-Net backbone adapted for colour output.

    mode="cmos_only"         input: (B, 4, H/2, W/2) packed RGGB
    mode="cmos_luma_direct"  input: (B, 5, H/2, W/2) RGGB + downsampled luma
    mode="cmos_luma_softmax" same 5ch; output multiplied by full-res luma (Eq. 12)
    mode="cmos_guided_align" deformable guide→SPAD alignment before backbone

    cmos_guided_align design:
      1. stem(cmos, 4ch) → guide_feat  [no luma at input]
      2. predict DCNv2 offsets from cat(guide_feat, luma_feat); zero-init → identity start
      3. warp guide_feat to SPAD coord system via DCNv2
      4. inject luma_down alongside warped feats; run UNet
      5. recompose as softmax × T_hat  (identical to cmos_luma_softmax Eq. 12)
    Reuses the same DCNv2 primitive as Stage-1; no new mechanism, only new target.

    Head output upsampled 2× via PixelShuffle → (B, 3, H, W).
    """

    MODES = tuple(ABLATION_MODES.values())

    def __init__(self, mode: str = "cmos_only", bc: int = 32, nb: int = 2, nfpm: int = 2):
        super().__init__()
        assert mode in self.MODES, f"Unknown mode '{mode}'"
        self.mode = mode
        c1, c2, c3, c4 = bc, bc * 2, bc * 4, bc * 8

        # cmos_guided_align: 4-ch stem (CMOS only); luma injected post-alignment
        in_ch = 4 if mode in ("cmos_only", "cmos_guided_align") else 5
        self.stem = nn.Conv2d(in_ch, c1, 3, 1, 1)
        self.fpm  = nn.Sequential(*[NAFBlock(c1) for _ in range(nfpm)])

        self.dn1 = DownBlock(c1, c2, nb)
        self.dn2 = DownBlock(c2, c3, nb)
        self.dn3 = DownBlock(c3, c4, nb)
        self.mid = nn.Sequential(*[NAFBlock(c4) for _ in range(nb * 2)])
        self.up3 = UpBlock(c4, c3, c3, nb)
        self.up2 = UpBlock(c3, c2, c2, nb)
        self.up1 = UpBlock(c2, c1, c1, nb)

        self.head = nn.Sequential(
            nn.Conv2d(c1, c1 * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(c1, 3, 3, 1, 1),
        )

        if mode == "cmos_guided_align":
            dg, K = 8, 9
            self.luma_feat_proj = nn.Conv2d(1, c1, 3, 1, 1)
            self.off_conv1 = nn.Conv2d(c1 * 2, c1, 3, 1, 1)
            self.off_conv2 = nn.Conv2d(c1, dg * K * 3, 3, 1, 1)
            nn.init.zeros_(self.off_conv2.weight)
            nn.init.zeros_(self.off_conv2.bias)
            self.off_lr = nn.LeakyReLU(0.1, True)
            self.align_dcn = DCNv2(c1, c1, 3, 1, dg)
            self.inject_conv = nn.Conv2d(c1 + 1, c1, 3, 1, 1)

    # ── backbone helpers ──────────────────────────────────────────────────────

    def _stem_feats(self, x):
        return self.fpm(self.stem(x))

    def _unet_from_feats(self, x):
        x, s1 = self.dn1(x)
        x, s2 = self.dn2(x)
        x, s3 = self.dn3(x)
        x = self.mid(x)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        return self.head(x)

    def _encode_decode(self, x):
        return self._unet_from_feats(self._stem_feats(x))

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, cmos: torch.Tensor, luma: Optional[torch.Tensor] = None):
        """
        Args:
            cmos: (B, 4, H/2, W/2) packed RGGB
            luma: (B, 1, H, W) SPAD luma in alpha*S space
                  — required for all modes except cmos_only

        Returns:
            (B, 3, H, W) in alpha*photon-count space
        """
        if self.mode == "cmos_only":
            logits = self._encode_decode(cmos)
            return F.softplus(logits.float())

        if self.mode == "cmos_guided_align":
            # 1. Guide features from 4-ch CMOS stem
            guide_feat = self._stem_feats(cmos)                          # (B, c1, H/2, W/2)

            # 2. Luma features at guide resolution for offset prediction
            luma_down  = F.interpolate(luma.float(), cmos.shape[-2:],
                                       mode='bilinear', align_corners=False)  # (B, 1, H/2, W/2)
            luma_feat  = self.luma_feat_proj(luma_down)                  # (B, c1, H/2, W/2)

            # 3. Predict DCN offsets: align guide → SPAD luma coord system
            o = self.off_conv2(self.off_lr(self.off_conv1(
                torch.cat([guide_feat, luma_feat], 1))))                  # (B, dg*K*3, H/2, W/2)
            n_off = 8 * 9 * 2  # dg=8, K=9, 2 coords → 144
            guide_feat_aligned = self.align_dcn(
                guide_feat, o[:, :n_off], o[:, n_off:])                  # (B, c1, H/2, W/2)

            # 4. Inject luma alongside aligned guide feats
            fused  = self.inject_conv(
                torch.cat([guide_feat_aligned, luma_down], 1))            # (B, c1, H/2, W/2)

            # 5. UNet backbone + simplex recomposition (Eq. 12)
            logits = self._unet_from_feats(fused)                         # (B, 3, H, W)
            chroma = F.softmax(logits.float(), dim=1)
            return chroma * luma.float()

        # ── cmos_luma_direct / cmos_luma_softmax ─────────────────────────────
        luma_hh = F.interpolate(luma.float(), cmos.shape[-2:],
                                mode='bilinear', align_corners=False)
        x_in   = torch.cat([cmos, luma_hh], dim=1)
        logits = self._encode_decode(x_in)

        if self.mode == "cmos_luma_direct":
            return F.softplus(logits.float())

        # cmos_luma_softmax: colour fractions × full-res luma
        chroma = F.softmax(logits.float(), dim=1)          # (B, 3, H, W), sum=1
        return chroma * luma.float()                        # broadcast (B,1,H,W)


# ── stage-1 luma helper ───────────────────────────────────────────────────────

_OLD_MODE_MAP = {
    "dcn_d0": "dcn_h2", "dcn_d1": "dcn_h4",
    "dcn_d2": "dcn_h8", "dcn_d3": "dcn_h16",
}


def load_stage1(ckpt_path: str, device, T: int = SEQ_LEN) -> Optional[SPADNet]:
    if not ckpt_path or not os.path.isfile(ckpt_path):
        return None
    ck   = load_checkpoint(ckpt_path, device)
    raw  = ck.get("mode") or ck.get("args", {}).get("mode", "dcn_h4")
    mode = _OLD_MODE_MAP.get(raw, raw)
    bc   = ck.get("args", {}).get("base_c", 32)
    m    = SPADNet(T=T, bc=bc, mode=mode).to(device)
    m.load_state_dict(ck["model"], strict=True)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    print(f"  [stage-1] loaded {ckpt_path}  mode={mode}")
    return m


@torch.no_grad()
def get_luma(batch: dict, stage1: Optional[SPADNet], device) -> torch.Tensor:
    """
    Return (B, 1, H, W) luma in alpha*S space.

    Uses stage-1 SPADNet output when available, otherwise GT alpha*S.
    """
    if stage1 is not None:
        spad = batch["spad_mono_seq"].to(device)
        with autocast("cuda"):
            luma, _ = stage1(spad)
        return luma.float()
    # GT luma: target_s is alpha*(R+G+B) = sum of target_rgb_s
    return batch["target_s"].unsqueeze(1).to(device).float()


# ── loss ──────────────────────────────────────────────────────────────────────

def color_loss(pred: torch.Tensor, target: torch.Tensor,
               criterion: CharbonnierLoss) -> torch.Tensor:
    """Charbonnier on PPP-normalised (B, 3, H, W) photon counts."""
    return criterion(pred / PPP, target / PPP)


# ── evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_color_batch(pred_raw: torch.Tensor,
                     gt_raw: torch.Tensor,
                     alpha_batch: torch.Tensor):
    """
    Per-sample RGB metrics in gamma-compressed sRGB space.

    pred_raw, gt_raw: (B, 3, H, W) in alpha*R_lin space.
    Dividing by alpha recovers R_lin; gamma-compressing gives sRGB [0,1],
    identical to the original loaded image. PSNR/SSIM are standard.
    """
    import math
    from skimage.metrics import structural_similarity

    pgam_all, sgam_all = [], []
    for b in range(pred_raw.shape[0]):
        a = max(alpha_batch[b].item(), 1e-8)
        p = pred_raw[b].float().clamp(0).cpu()    # (3, H, W)
        g = gt_raw  [b].float().clamp(0).cpu()

        # /alpha → R_lin → gamma → sRGB [0,1]
        p_d = (p / a).clamp(0, 1) ** (1.0 / GAMMA)
        g_d = (g / a).clamp(0, 1) ** (1.0 / GAMMA)

        mse = float(((p_d - g_d) ** 2).mean())
        pgam_all.append(100.0 if mse == 0 else 10.0 * math.log10(1.0 / mse))

        p_np = p_d.permute(1, 2, 0).numpy()  # (H, W, 3)
        g_np = g_d.permute(1, 2, 0).numpy()
        sgam_all.append(float(structural_similarity(
            p_np, g_np, data_range=1.0, channel_axis=2)))

    return pgam_all, sgam_all


# ── data loaders ──────────────────────────────────────────────────────────────

def make_loader(ds, batch_size, shuffle, num_workers, drop_last=False):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      drop_last=drop_last,
                      persistent_workers=(num_workers > 0),
                      worker_init_fn=_worker_init if num_workers > 0 else None)


# ── training epoch ────────────────────────────────────────────────────────────

def train_epoch(model, stage1, loader, opt, scaler, criterion,
                device, epoch, max_steps=0, wandb_on=False, gstep=0):
    model.train()
    total, n = 0., 0
    needs_luma = model.mode != "cmos_only"
    pbar = tqdm(loader, desc=f"Ep{epoch:03d}[{model.mode}]",
                leave=False, disable=not _TTY,
                total=max_steps if max_steps > 0 else len(loader))

    for batch in pbar:
        cmos   = batch["cmos_packed"].to(device)
        target = batch["target_rgb_s"].to(device)
        luma   = get_luma(batch, stage1, device) if needs_luma else None

        opt.zero_grad(set_to_none=True)
        with autocast("cuda"):
            pred = model(cmos, luma)
            loss = color_loss(pred, target, criterion)

        if not torch.isfinite(loss):
            print(f"\n  [SKIP ep{epoch}/step{gstep}] non-finite loss, skipping",
                  flush=True)
            opt.zero_grad(set_to_none=True)
            n += 1; gstep += 1
            if max_steps > 0 and n >= max_steps:
                break
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(opt)

        has_bad_grad = any(
            p.grad is not None and not torch.isfinite(p.grad).all()
            for p in model.parameters() if p.requires_grad)
        if has_bad_grad:
            print(f"\n  [SKIP-GRAD ep{epoch}/step{gstep}] NaN gradient, skipping",
                  flush=True)
            opt.zero_grad(set_to_none=True)
            scaler.update()
            n += 1; gstep += 1
            if max_steps > 0 and n >= max_steps:
                break
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        total += loss.item(); n += 1; gstep += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        if not _TTY and n % 200 == 0:
            print(f"  Ep{epoch:03d} step {n}/{max_steps if max_steps else '?'}  "
                  f"loss={total/n:.4f}", flush=True)
        if wandb_on and WANDB_OK:
            wandb.log({"train/loss": loss.item()}, step=gstep)
        if max_steps > 0 and n >= max_steps:
            break

    return total / max(n, 1), gstep


# ── eval loop ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_eval(model, stage1, loader, device, max_batches=0):
    model.eval()
    needs_luma = model.mode != "cmos_only"
    pgam_all, sgam_all = [], []

    for bi, batch in enumerate(tqdm(loader, desc="eval", leave=False,
                                    disable=not _TTY)):
        cmos  = batch["cmos_packed"].to(device)
        target = batch["target_rgb_s"].to(device)
        alpha = batch["alpha"].to(device)
        luma  = get_luma(batch, stage1, device) if needs_luma else None

        with autocast("cuda"):
            pred = model(cmos, luma)

        pg, sg = eval_color_batch(pred.float(), target.float(), alpha)
        pgam_all.extend(pg); sgam_all.extend(sg)
        if max_batches > 0 and (bi + 1) >= max_batches:
            break

    _m = lambda lst: float(np.mean(lst)) if lst else float('nan')
    return dict(psnr_gam=_m(pgam_all), ssim_gam=_m(sgam_all))


def print_results(res, tag="EVAL"):
    print(f"\n{'='*52}")
    print(f"  {tag}")
    print(f"  PSNR (gam) : {res['psnr_gam']:>7.4f} dB  <- paper metric")
    print(f"  SSIM (gam) : {res['ssim_gam']:>7.4f}")
    print(f"{'='*52}")


# ── argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train colour ablation on spad-net U-Net")
    p.add_argument("--ablation",      type=int, required=True, choices=[1, 2, 3, 4],
                   help="1=cmos_only  2=cmos+luma direct  3=cmos+luma softmax  4=cmos_guided_align")
    p.add_argument("--config",        default="configs/base.yaml")
    p.add_argument("--data_root",     default="",    help="Root of PNG sequences")
    p.add_argument("--ckpt_dir",      default="runs/color_a1")
    p.add_argument("--ckpt",          default="",    help="Pretrained weights (model only)")
    p.add_argument("--resume",        default="",    help="Full resume checkpoint")
    p.add_argument("--stage1_ckpt",   default="",
                   help="Frozen stage-1 SPADNet checkpoint for luma (ablations 2/3). "
                        "If omitted, GT alpha*S is used as luma proxy.")
    p.add_argument("--device",        default="")
    p.add_argument("--epochs",        type=int,   default=0)
    p.add_argument("--steps_per_epoch", type=int, default=0)
    p.add_argument("--batch",         type=int,   default=0)
    p.add_argument("--lr",            type=float, default=0.)
    p.add_argument("--workers",       type=int,   default=-1)
    p.add_argument("--crop",          type=int,   default=0)
    p.add_argument("--eval_crop",     type=int,   default=-1)
    p.add_argument("--ppp",           type=float, default=0.)
    p.add_argument("--bins",          type=int,   default=0)
    p.add_argument("--cmos_sigma",    type=float, default=0.,
                   help="CMOS read-noise std dev in ADC counts (default 2.0)")
    p.add_argument("--sps",           type=int,   default=0)
    p.add_argument("--eval_only",     action="store_true")
    p.add_argument("--first_window",  action="store_true",
                   help="eval_only: one window per scene (first SEQ_LEN frames only)")
    p.add_argument("--eval_batches",  type=int,   default=0)
    p.add_argument("--save_every",    type=int,   default=0)
    p.add_argument("--wandb_project", default="")
    p.add_argument("--wandb_run",     default="")
    p.add_argument("--no_wandb",      action="store_true")
    p.add_argument("--reset_best",    action="store_true",
                   help="Ignore stored best_psnr_gam on resume (use when eval metric changed)")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--cmos_t",        type=int,   default=1,
                   help="CMOS integration frames (1=center only, 3/5/7/9 for longer exposure)")
    return p.parse_args()


def apply_overrides(cfg, args):
    if args.epochs > 0:           cfg["train"]["epochs"]          = args.epochs
    if args.steps_per_epoch > 0:  cfg["train"]["steps_per_epoch"] = args.steps_per_epoch
    if args.batch > 0:            cfg["train"]["batch"]           = args.batch
    if args.lr > 0:               cfg["train"]["lr"]              = args.lr
    if args.workers >= 0:         cfg["data"]["workers"]          = args.workers
    if args.crop > 0:             cfg["data"]["crop"]             = args.crop
    if args.eval_crop >= 0:       cfg["data"]["eval_crop"]        = args.eval_crop
    if args.ppp > 0:              cfg["data"]["ppp"]              = args.ppp
    if args.bins > 0:             cfg["data"]["bins"]             = args.bins
    if args.sps > 0:              cfg["data"]["sps"]              = args.sps
    if args.save_every > 0:       cfg["train"]["save_every"]      = args.save_every
    if args.eval_batches > 0:     cfg["train"]["eval_batches"]    = args.eval_batches
    if args.wandb_project:        cfg["train"]["wandb_project"]   = args.wandb_project
    return cfg


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    mode = ABLATION_MODES[args.ablation]

    cfg = load_config(args.config) if os.path.isfile(args.config) else {}
    cfg.setdefault("model", {}); cfg.setdefault("data", {}); cfg.setdefault("train", {})
    cfg = apply_overrides(cfg, args)

    mc = cfg["model"]; dc = cfg["data"]; tc = cfg["train"]

    seed = args.seed
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))

    epochs       = tc.get("epochs", 100)
    steps_per_ep = tc.get("steps_per_epoch", 2000)
    batch_size   = tc.get("batch", 4)
    lr           = tc.get("lr", 2e-4)
    weight_decay = tc.get("weight_decay", 1e-4)
    betas        = tuple(tc.get("betas", [0.9, 0.9]))
    lr_min       = tc.get("lr_min", 1e-6)
    save_every   = tc.get("save_every", 5)
    eval_batches = tc.get("eval_batches", 25)
    wandb_project = tc.get("wandb_project", "DUALQUANTA")
    workers      = dc.get("workers", 4)
    ppp          = dc.get("ppp", PPP)
    bins         = dc.get("bins", BINS)
    crop         = dc.get("crop", 256)
    eval_crop    = dc.get("eval_crop", 256)
    sps          = dc.get("sps", 1)
    eval_frac    = dc.get("eval_frac", 0.1)
    cmos_sigma   = args.cmos_sigma if args.cmos_sigma > 0 else CMOS_SIGMA

    torch.backends.cudnn.benchmark = not args.eval_only
    os.makedirs(args.ckpt_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  ColorAblation  |  ablation={args.ablation}  mode={mode}")
    print(f"  device={device}  epochs={epochs}  batch={batch_size}")
    print(f"  cmos_sigma={cmos_sigma}  ppp={ppp}  bins={bins}  cmos_t={args.cmos_t}")
    luma_src = f"stage-1 ({args.stage1_ckpt})" if args.stage1_ckpt else "GT alpha*S"
    if mode != "cmos_only":
        print(f"  luma source: {luma_src}")
    print(f"{'='*60}\n")

    model = ColorUNet(
        mode=mode,
        bc=mc.get("base_channels", 32),
        nb=mc.get("n_blocks", 2),
        nfpm=mc.get("n_fpm", 2),
    ).to(device)

    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"  trainable={n_tr:.3f} M\n")

    stage1 = load_stage1(args.stage1_ckpt, device, T=SEQ_LEN) if mode != "cmos_only" else None

    if args.resume and os.path.isfile(args.resume):
        ck = load_checkpoint(args.resume, device)
        model.load_state_dict(ck["model"], strict=False)
        print(f"  [loaded] {args.resume}  ep={ck.get('epoch', '?')}")
    elif args.ckpt and os.path.isfile(args.ckpt):
        ck = load_checkpoint(args.ckpt, device)
        model.load_state_dict(ck["model"], strict=False)
        print(f"  [loaded] {args.ckpt}")

    use_wb = (not args.no_wandb) and WANDB_OK
    if use_wb:
        run_name = args.wandb_run or f"color_a{args.ablation}"
        wb_cfg = {"ablation": args.ablation, "color_mode": mode,
                  "cmos_sigma": cmos_sigma}
        wb_cfg.update(mc); wb_cfg.update(dc); wb_cfg.update(tc)
        wandb.init(project=wandb_project, name=run_name, config=wb_cfg,
                   settings=wandb.Settings(init_timeout=300))

    if not args.data_root:
        raise RuntimeError("--data_root is required")
    tr_root = args.data_root
    if os.path.isdir(os.path.join(tr_root, "train")):
        tr_root = os.path.join(tr_root, "train")

    def _ds(samples, aug, rp=False, ec=None):
        return CMOSColorDataset(
            samples, ppp=ppp, bins=bins, cmos_sigma=cmos_sigma,
            augment=aug, crop=crop,
            eval_crop=(eval_crop if ec is None else ec),
            seed=seed, random_pick=rp, sps=sps_eff,
            cmos_t=args.cmos_t)

    if args.eval_only:
        ts_root = (os.path.join(args.data_root, "test")
                   if os.path.isdir(os.path.join(args.data_root, "test"))
                   else tr_root)
        src = build_test(ts_root)
        sps_eff = 1
        res = run_eval(model, stage1,
                       make_loader(_ds(src, False, ec=eval_crop), 1, False, workers),
                       device, eval_batches)
        print_results(res, f"TEST | ablation {args.ablation} | {mode}")
        if use_wb:
            wandb.log(res); wandb.finish()
        return

    tr_s, ev_s = build_splits(tr_root, eval_frac, seed)
    sps_eff = sps
    if steps_per_ep > 0 and tr_s:
        need = math.ceil(steps_per_ep * batch_size / len(tr_s))
        if sps_eff < need:
            print(f"  [auto] sps {sps_eff}→{need}")
            sps_eff = need

    # One window per eval scene — ensures all scenes are covered in every eval pass
    _ev_seen: set = set()
    ev_one = [s for s in ev_s
              if s.scene_key not in _ev_seen and not _ev_seen.add(s.scene_key)]
    print(f"  eval: {len(ev_one)} scenes (1 window each, from {len(ev_s)} total windows)")

    tr_ld = make_loader(_ds(tr_s, True, rp=True), batch_size, True,
                        workers, drop_last=True)
    ev_ld = make_loader(_ds(ev_one, False), 1, False, workers) if ev_one else None
    steps = steps_per_ep if steps_per_ep > 0 else len(tr_ld)

    criterion = CharbonnierLoss()
    opt   = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr, weight_decay=weight_decay, betas=betas)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, lr_min)
    scaler = GradScaler("cuda", init_scale=256)
    best   = 0.
    gstep  = 0
    start_epoch = 0

    if args.resume and os.path.isfile(args.resume):
        ck = load_checkpoint(args.resume, device)
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        start_epoch = ck["epoch"]
        best   = 0. if args.reset_best else ck.get("best_psnr_gam", 0.)
        gstep  = ck.get("gstep", start_epoch * steps)
        print(f"  [resumed] {args.resume}  (epoch {start_epoch}, best={best:.3f}"
              f"{' — reset' if args.reset_best else ''})")

    logf = os.path.join(args.ckpt_dir, "log.csv")

    for ep in range(start_epoch + 1, epochs + 1):
        t0 = time.time()
        tloss, gstep = train_epoch(
            model, stage1, tr_ld, opt, scaler, criterion,
            device, ep, steps, use_wb, gstep)
        sched.step()

        pg = sg = float('nan')
        if ev_ld:
            res = run_eval(model, stage1, ev_ld, device, eval_batches)
            pg, sg = res["psnr_gam"], res["ssim_gam"]

        lr_now = opt.param_groups[0]["lr"]
        print(f"Ep{ep:03d}  loss={tloss:.4f}  PSNRgam={pg:.3f}  "
              f"SSIMgam={sg:.4f}  lr={lr_now:.1e}  {time.time()-t0:.0f}s")

        if use_wb:
            wandb.log(dict(epoch=ep, loss=tloss, psnr_gam=pg,
                           ssim_gam=sg, lr=lr_now), step=gstep)

        with open(logf, "a") as f:
            f.write(f"{ep},{tloss:.6f},{pg:.4f},{sg:.4f},{lr_now:.2e}\n")

        is_best = pg > best
        if is_best:
            best = pg

        state = dict(epoch=ep, gstep=gstep, ablation=args.ablation, mode=mode,
                     model=model.state_dict(),
                     optimizer=opt.state_dict(), scheduler=sched.state_dict(),
                     best_psnr_gam=best)

        save_checkpoint(state, os.path.join(args.ckpt_dir, "latest.pth"))
        if is_best:
            save_checkpoint(state, os.path.join(args.ckpt_dir, "best.pth"))
        elif ep % save_every == 0:
            save_checkpoint(state, os.path.join(args.ckpt_dir, f"ep{ep:04d}.pth"))

    if use_wb:
        wandb.finish()


if __name__ == "__main__":
    main()
