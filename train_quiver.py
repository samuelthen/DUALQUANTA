#!/usr/bin/env python3
"""
QUIVER training script — retrains QUIVER's actual architecture (Chennuri et
al. 2024) under this project's own SPAD forward model / dataset pipeline, so
it's a fair same-data-same-protocol baseline for Table 1 (see
configs/mono/quiver_retrained.yaml: "QUIVER retrained under paper protocol").

This is QUIVER's real network (src/models/quiver_model.py — ported from the
released repo, not re-implemented), trained here with SPADNet's dataloader
instead of QUIVER's own MP4-based one. Only the code needed to run/train that
architecture lives in this repo — not the original QUIVER repo (baselines,
website assets, other architectures, MP4 dataloader, etc. all stay out).

Output domain (verified empirically against checkpoints/table1_mono/
quiver_retrained.pth — its out1 range/mean matches this transform almost
exactly, confirming it's what the existing checkpoint was trained against):
    target = clamp(target_s / pct99, 0, 1) ** (1/GAMMA)   — pct99-normalised,
    gamma-encoded display-domain image in [0, 1] (matches the ORIGINAL
    QUIVER paper's own gt_seq/255 convention, just pct99 instead of a fixed
    255 since our forward model has no fixed white level).
    This is NOT the same domain as SPADNet's raw alpha*S softplus output —
    do not compare across models without evaluate.py's domain conversion.

Multi-scale loss (matches original QUIVER training weighting):
    0.85 * L(out1, gt_full) + 0.10 * L(out2, gt_h2) + 0.05 * L(out3, gt_h4)
    + 0.20 * L(preden[:,center], gt_full)
  gt_h2/gt_h4 are avg-pooled in the LINEAR (pre-gamma) domain, then
  pct99-normalised + gamma-encoded, matching the pipeline order above.
  NOTE: preden is supervised on its centre frame only (SPADDataset provides
  a clean target for the centre frame, not all T frames like QUIVER's
  original per-frame dataloader) — a documented simplification, not a bug.

Usage:
    python train_quiver.py --data_root /path/to/data --ckpt_dir runs/quiver_retrained \
                           --device cuda:0
"""

import argparse
import math
import os
import random
import sys
import time

import cv2
cv2.setNumThreads(0)
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
    WANDB_OK = True
except ImportError:
    WANDB_OK = False

from src import SPADDataset, build_splits, build_test
from src.data.dataset import _worker_init
from src.losses import CharbonnierLoss
from src.models.quiver_model import QUIVER
from src.utils import load_config, save_checkpoint, load_checkpoint

_TTY = sys.stderr.isatty()

SEQ_LEN  = 11
HALF_WIN = SEQ_LEN // 2
GAMMA    = 2.2


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train QUIVER (retrained-under-paper-protocol baseline)")
    p.add_argument("--config",    default="configs/mono/quiver_retrained.yaml")
    p.add_argument("--data_root", default="", help="Root dir with PNG frame sequences")
    p.add_argument("--ckpt_dir",  default="runs/quiver_retrained")
    p.add_argument("--ckpt",      default="", help="Pretrained weights (model only)")
    p.add_argument("--resume",    default="", help="Full resume: model+opt+sched+epoch")
    p.add_argument("--device",    default="")
    p.add_argument("--epochs",          type=int,   default=0)
    p.add_argument("--steps_per_epoch", type=int,   default=0)
    p.add_argument("--batch",           type=int,   default=0)
    p.add_argument("--lr",              type=float, default=0.)
    p.add_argument("--workers",         type=int,   default=-1)
    p.add_argument("--crop",            type=int,   default=0)
    p.add_argument("--eval_crop",       type=int,   default=0)
    p.add_argument("--ppp",             type=float, default=0.)
    p.add_argument("--bins",            type=int,   default=0)
    p.add_argument("--sps",             type=int,   default=0)
    p.add_argument("--eval_only",       action="store_true")
    p.add_argument("--eval_batches",    type=int, default=0)
    p.add_argument("--save_every",      type=int, default=0)
    p.add_argument("--wandb_project",   default="")
    p.add_argument("--wandb_run",       default="")
    p.add_argument("--no_wandb",        action="store_true")
    p.add_argument("--seed",            type=int, default=42)
    return p.parse_args()


def apply_cli_overrides(cfg, args):
    if args.epochs > 0:          cfg["train"]["epochs"] = args.epochs
    if args.steps_per_epoch > 0: cfg["train"]["steps_per_epoch"] = args.steps_per_epoch
    if args.batch > 0:           cfg["train"]["batch"] = args.batch
    if args.lr > 0:               cfg["train"]["lr"] = args.lr
    if args.workers >= 0:        cfg["data"]["workers"] = args.workers
    if args.crop > 0:            cfg["data"]["crop"] = args.crop
    if args.eval_crop > 0:       cfg["data"]["eval_crop"] = args.eval_crop
    if args.ppp > 0:              cfg["data"]["ppp"] = args.ppp
    if args.bins > 0:            cfg["data"]["bins"] = args.bins
    if args.sps > 0:             cfg["data"]["sps"] = args.sps
    if args.save_every > 0:      cfg["train"]["save_every"] = args.save_every
    if args.wandb_project:       cfg["train"]["wandb_project"] = args.wandb_project
    return cfg


def make_loader(ds, batch_size, shuffle, num_workers, drop_last=False):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      drop_last=drop_last,
                      persistent_workers=(num_workers > 0),
                      worker_init_fn=_worker_init if num_workers > 0 else None)


# ── target construction ────────────────────────────────────────────────────────

def _gam_targets(gt_s, pct99):
    """gt_s: (B,1,H,W) raw alpha*S (linear). Returns (gt_full, gt_h2, gt_h4),
    each pct99-normalised + gamma-encoded, matching QUIVER's own display-domain
    output convention (verified against the existing quiver_retrained.pth)."""
    p99 = pct99.view(-1, 1, 1, 1)
    def _tf(x):
        return torch.clamp(x / (p99 + 1e-8), 0, 1) ** (1. / GAMMA)
    gt_full = _tf(gt_s)
    gt_h2   = _tf(F.avg_pool2d(gt_s, 2, 2))
    gt_h4   = _tf(F.avg_pool2d(gt_s, 4, 4))
    return gt_full, gt_h2, gt_h4


# ── training epoch ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, opt, scaler, criterion, device, epoch,
                max_steps=0, wandb_on=False, gstep=0):
    model.train()
    total, n = 0., 0
    pbar = tqdm(loader, desc=f"Ep{epoch:03d}[quiver]", leave=False,
               disable=not _TTY, total=max_steps if max_steps > 0 else len(loader))

    for batch in pbar:
        spad  = batch["spad_mono_seq"].unsqueeze(2).to(device)   # (B,T,1,H,W)
        gt_s  = batch["target_s"].unsqueeze(1).to(device)         # (B,1,H,W)
        pct99 = batch["pct99"].to(device)
        gt_full, gt_h2, gt_h4 = _gam_targets(gt_s, pct99)

        opt.zero_grad(set_to_none=True)
        with autocast("cuda"):
            preden, out1, out2, out3 = model(spad)
            preden_ctr = preden[:, HALF_WIN]                      # (B,1,H,W)
            loss = (0.85 * criterion(out1[:, 0].float(), gt_full)
                  + 0.10 * criterion(out2[:, 0].float(), gt_h2)
                  + 0.05 * criterion(out3[:, 0].float(), gt_h4)
                  + 0.20 * criterion(preden_ctr.float(), gt_full))

        if not torch.isfinite(loss):
            print(f"\n  [SKIP ep{epoch}/step{gstep}] non-finite loss, batch skipped", flush=True)
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
            print(f"\n  [SKIP-GRAD ep{epoch}/step{gstep}] NaN/inf gradient, skipping", flush=True)
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
        if wandb_on and WANDB_OK:
            wandb.log({"train/loss": loss.item()}, step=gstep)
        if max_steps > 0 and n >= max_steps:
            break

    return total / max(n, 1), gstep


# ── evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_eval(model, loader, device, max_batches=0):
    """PSNR/SSIM in the same pct99-normalised gamma domain used for training
    (i.e. NOT the alpha-normalised sRGB metric evaluate.py reports for the
    paper table — this is a fast in-loop proxy metric for model selection)."""
    model.eval()
    psnrs, ssims = [], []
    from skimage.metrics import structural_similarity as _ssim

    for bi, batch in enumerate(tqdm(loader, desc="eval", leave=False, disable=not _TTY)):
        spad  = batch["spad_mono_seq"].unsqueeze(2).to(device)
        gt_s  = batch["target_s"].unsqueeze(1).to(device)
        pct99 = batch["pct99"].to(device)
        gt_full, _, _ = _gam_targets(gt_s, pct99)

        with autocast("cuda"):
            _, out1, _, _ = model(spad)
        pred = torch.clamp(out1[:, 0].float(), 0, 1)

        for b in range(pred.shape[0]):
            p_np = pred[b, 0].cpu().numpy()
            g_np = gt_full[b, 0].cpu().numpy()
            mse = float(np.mean((p_np - g_np) ** 2))
            psnrs.append(100.0 if mse == 0 else 10.0 * math.log10(1.0 / mse))
            ssims.append(float(_ssim(p_np, g_np, data_range=1.0)))
        if max_batches > 0 and (bi + 1) >= max_batches:
            break

    return dict(psnr_gam=float(np.mean(psnrs)) if psnrs else float('nan'),
                ssim_gam=float(np.mean(ssims)) if ssims else float('nan'))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg = load_config(args.config) if os.path.isfile(args.config) else {}
    cfg.setdefault("model", {}); cfg.setdefault("data", {}); cfg.setdefault("train", {})
    cfg = apply_cli_overrides(cfg, args)
    dc = cfg["data"]; tc = cfg["train"]

    seed = args.seed
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs        = tc.get("epochs", 100)
    steps_per_ep  = tc.get("steps_per_epoch", 2000)
    batch_size    = tc.get("batch", 4)
    lr            = tc.get("lr", 2e-4)
    weight_decay  = tc.get("weight_decay", 1e-4)
    betas         = tuple(tc.get("betas", [0.9, 0.9]))
    lr_min        = tc.get("lr_min", 1e-6)
    save_every    = tc.get("save_every", 5)
    wandb_project = tc.get("wandb_project", "DUALQUANTA")
    workers       = dc.get("workers", 4)
    ppp           = dc.get("ppp", 3.25)
    bins          = dc.get("bins", 7)
    crop          = dc.get("crop", 256)
    eval_crop     = dc.get("eval_crop", 256)
    sps           = dc.get("sps", 1)
    eval_frac     = dc.get("eval_frac", 0.1)

    torch.backends.cudnn.benchmark = not args.eval_only
    os.makedirs(args.ckpt_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  QUIVER (retrained-under-paper-protocol)")
    print(f"  device={device}   epochs={epochs}   batch={batch_size}")
    print(f"  target: pct99-normalised gamma display domain [0,1]")
    print(f"{'='*60}\n")

    para = argparse.Namespace(
        inp_ch=1, n_features=64, n_blocks=12,
        past_frames=HALF_WIN, future_frames=HALF_WIN,
        activation="gelu", spynet_path="", load_spynet_weights=False)
    model = QUIVER(para).to(device)
    model.device = device

    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"  trainable={n_tr:.2f} M\n")

    if not args.resume and args.ckpt and os.path.isfile(args.ckpt):
        ck = load_checkpoint(args.ckpt, device)
        model.load_state_dict(ck["model"], strict=False)
        print(f"  [loaded] {args.ckpt}")

    use_wb = (not args.no_wandb) and WANDB_OK
    if use_wb:
        run_name = args.wandb_run or "quiver_retrained"
        wandb.init(project=wandb_project, name=run_name, config={**dc, **tc})

    tr_root = args.data_root
    if not tr_root:
        raise RuntimeError("--data_root is required")
    if os.path.isdir(os.path.join(tr_root, "train")):
        tr_root = os.path.join(tr_root, "train")

    def _ds(samples, aug, rp=False, ec=None):
        return SPADDataset(
            samples, ppp=ppp, bins=bins, augment=aug, crop=crop,
            eval_crop=(eval_crop if ec is None else ec),
            seed=seed, random_pick=rp, sps=sps_eff,
            sensor_mode="mono", target_mode="luma")

    if args.eval_only:
        ts_root = os.path.join(args.data_root, "test") \
                  if os.path.isdir(os.path.join(args.data_root, "test")) else tr_root
        samples = build_test(ts_root)
        sps_eff = 1
        res = run_eval(model, make_loader(_ds(samples, False, ec=eval_crop), 1, False, workers),
                      device, args.eval_batches)
        print(f"\nTEST | quiver  PSNRgam={res['psnr_gam']:.3f}  SSIMgam={res['ssim_gam']:.4f}\n")
        if use_wb:
            wandb.log(res); wandb.finish()
        return

    tr_s, ev_s = build_splits(tr_root, eval_frac, seed)
    sps_eff = sps
    if steps_per_ep > 0 and tr_s:
        need = math.ceil(steps_per_ep * batch_size / len(tr_s))
        if sps_eff < need:
            print(f"  [auto] sps {sps_eff}->{need}")
            sps_eff = need

    tr_ld = make_loader(_ds(tr_s, True, rp=True), batch_size, True, workers, drop_last=True)
    ev_ld = make_loader(_ds(ev_s, False), 1, False, workers) if ev_s else None
    steps = steps_per_ep if steps_per_ep > 0 else len(tr_ld)

    criterion = CharbonnierLoss()
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=lr, weight_decay=weight_decay, betas=betas)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, lr_min)
    scaler = GradScaler("cuda")
    best, gstep, start_epoch = 0., 0, 0

    if args.resume and os.path.isfile(args.resume):
        ck = load_checkpoint(args.resume, device)
        model.load_state_dict(ck["model"], strict=False)
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        start_epoch = ck["epoch"]
        best  = ck.get("best_psnr_gam", 0.)
        gstep = ck.get("gstep", start_epoch * steps)
        print(f"  [resumed] {args.resume}  (epoch {start_epoch}, best={best:.3f})")

    logf = os.path.join(args.ckpt_dir, "log.csv")

    for ep in range(start_epoch + 1, epochs + 1):
        t0 = time.time()
        tloss, gstep = train_epoch(model, tr_ld, opt, scaler, criterion,
                                   device, ep, steps, use_wb, gstep)
        sched.step()

        pg = sg = float('nan')
        if ev_ld:
            res = run_eval(model, ev_ld, device, args.eval_batches)
            pg, sg = res["psnr_gam"], res["ssim_gam"]

        lr_now = opt.param_groups[0]["lr"]
        print(f"Ep{ep:03d}  loss={tloss:.4f}  PSNRgam={pg:.3f}  SSIMgam={sg:.4f}  "
              f"lr={lr_now:.1e}  {time.time()-t0:.0f}s")

        if use_wb:
            wandb.log(dict(epoch=ep, loss=tloss, psnr_gam=pg, ssim_gam=sg, lr=lr_now), step=gstep)
        with open(logf, "a") as f:
            f.write(f"{ep},{tloss:.6f},{pg:.4f},{sg:.4f},{lr_now:.2e}\n")

        is_best = pg > best
        if is_best:
            best = pg

        state = dict(epoch=ep, gstep=gstep, model=model.state_dict(),
                    optimizer=opt.state_dict(), scheduler=sched.state_dict(),
                    best_psnr_gam=best, mode="quiver", sensor_mode="mono", out_ch=1)

        save_checkpoint(state, os.path.join(args.ckpt_dir, "latest.pth"))
        if is_best:
            save_checkpoint(state, os.path.join(args.ckpt_dir, "best.pth"))
        elif ep % save_every == 0:
            save_checkpoint(state, os.path.join(args.ckpt_dir, f"ep{ep:04d}.pth"))

    if use_wb:
        wandb.finish()


if __name__ == "__main__":
    main()
