#!/usr/bin/env python3
"""
SPADNet training script.

Usage:
    python train.py --config configs/dcn_h4.yaml \
                    --data_root /path/to/data \
                    --ckpt_dir  runs/dcn_h4 \
                    --device    cuda:0

CLI flags override config values. Use --no_grad_loss to disable the gradient
loss term (equivalent to the single_frame_no_grad baseline in the paper).
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
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
    WANDB_OK = True
except ImportError:
    WANDB_OK = False

from src import SPADNet, SPADDataset, build_splits
from src.data.dataset import _worker_init
from src.losses import CharbonnierLoss, reconstruction_loss
from src.metrics import evaluate_batch
from src.utils import load_config, save_checkpoint, load_checkpoint

_TTY = sys.stderr.isatty()

SEQ_LEN = 11


# ── argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train SPADNet")
    p.add_argument("--config",    default="configs/dcn_h4.yaml",
                   help="YAML config path")
    p.add_argument("--data_root", default="",
                   help="Root directory containing PNG frame sequences")
    p.add_argument("--ckpt_dir",  default="runs/dcn_h4",
                   help="Output directory for checkpoints and logs")
    p.add_argument("--ckpt",      default="",
                   help="Pretrained checkpoint to load (model weights only)")
    p.add_argument("--resume",    default="",
                   help="Full resume: model + optimizer + scheduler + epoch")
    p.add_argument("--device",    default="",
                   help="PyTorch device string (default: cuda if available)")

    # Config overrides
    p.add_argument("--mode",             default="",
                   choices=[""] + list(SPADNet.MODES))
    p.add_argument("--no_grad_loss",     action="store_true",
                   help="Disable gradient loss term (Charbonnier-only)")
    p.add_argument("--epochs",           type=int,   default=0)
    p.add_argument("--steps_per_epoch",  type=int,   default=0)
    p.add_argument("--batch",            type=int,   default=0)
    p.add_argument("--lr",               type=float, default=0.)
    p.add_argument("--workers",          type=int,   default=-1)
    p.add_argument("--crop",             type=int,   default=0)
    p.add_argument("--eval_crop",        type=int,   default=0)
    p.add_argument("--ppp",              type=float, default=0.)
    p.add_argument("--bins",             type=int,   default=0)
    p.add_argument("--sps",              type=int,   default=0,
                   help="Samples per scene (0 = auto)")
    p.add_argument("--raft_ckpt",        default="")
    p.add_argument("--eval_only",        action="store_true")
    p.add_argument("--eval_batches",     type=int, default=0)
    p.add_argument("--save_every",       type=int, default=0)
    p.add_argument("--wandb_project",    default="")
    p.add_argument("--wandb_run",        default="")
    p.add_argument("--no_wandb",         action="store_true")
    p.add_argument("--seed",             type=int, default=42)
    return p.parse_args()


def apply_cli_overrides(cfg, args):
    """Merge explicit CLI args on top of config dict."""
    if args.mode:
        cfg["model"]["mode"] = args.mode
    if args.no_grad_loss:
        cfg["model"]["grad_loss"] = False
    if args.epochs > 0:
        cfg["train"]["epochs"] = args.epochs
    if args.steps_per_epoch > 0:
        cfg["train"]["steps_per_epoch"] = args.steps_per_epoch
    if args.batch > 0:
        cfg["train"]["batch"] = args.batch
    if args.lr > 0:
        cfg["train"]["lr"] = args.lr
    if args.workers >= 0:
        cfg["data"]["workers"] = args.workers
    if args.crop > 0:
        cfg["data"]["crop"] = args.crop
    if args.eval_crop > 0:
        cfg["data"]["eval_crop"] = args.eval_crop
    if args.ppp > 0:
        cfg["data"]["ppp"] = args.ppp
    if args.bins > 0:
        cfg["data"]["bins"] = args.bins
    if args.sps > 0:
        cfg["data"]["sps"] = args.sps
    if args.raft_ckpt:
        cfg["model"]["raft_ckpt"] = args.raft_ckpt
    if args.save_every > 0:
        cfg["train"]["save_every"] = args.save_every
    if args.wandb_project:
        cfg["train"]["wandb_project"] = args.wandb_project
    return cfg


# ── data loaders ──────────────────────────────────────────────────────────────

def make_loader(ds, batch_size, shuffle, num_workers, drop_last=False):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      drop_last=drop_last,
                      persistent_workers=(num_workers > 0),
                      worker_init_fn=_worker_init if num_workers > 0 else None)


# ── training epoch ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, opt, scaler, criterion, use_grad_loss,
                device, epoch, max_steps=0, wandb_on=False, gstep=0):
    model.train()
    total, n = 0., 0
    needs_rgb = (model.mode == "oracle_flow")
    pbar = tqdm(loader, desc=f"Ep{epoch:03d}[{model.mode}]",
                leave=False, disable=not _TTY,
                total=max_steps if max_steps > 0 else len(loader))

    for batch in pbar:
        spad = batch["spad_mono_seq"].to(device)
        gt_s = batch["target_s"].unsqueeze(1).to(device)   # (B,1,H,W) alpha*S
        rgb  = batch["clean_rgb_seq"].to(device) if needs_rgb else None

        opt.zero_grad(set_to_none=True)
        with autocast("cuda"):
            pred, _ = model(spad, clean_rgb=rgb)
            loss     = reconstruction_loss(pred, gt_s, criterion,
                                           use_grad_loss=use_grad_loss)

        # FIX-2d: skip non-finite loss batch
        if not torch.isfinite(loss):
            print(f"\n  [SKIP ep{epoch}/step{gstep}] non-finite loss={loss.item()}, "
                  f"batch skipped", flush=True)
            opt.zero_grad(set_to_none=True)
            n += 1; gstep += 1
            pbar.set_postfix(loss="NaN-skip")
            if wandb_on and WANDB_OK:
                wandb.log({"train/loss": float('nan')}, step=gstep)
            if max_steps > 0 and n >= max_steps:
                break
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(opt)

        # Guard against NaN/inf gradients (defense-in-depth)
        has_bad_grad = any(
            p.grad is not None and not torch.isfinite(p.grad).all()
            for p in model.parameters() if p.requires_grad)
        if has_bad_grad:
            print(f"\n  [SKIP-GRAD ep{epoch}/step{gstep}] NaN/inf gradient, skipping",
                  flush=True)
            opt.zero_grad(set_to_none=True)
            scaler.update()
            n += 1; gstep += 1
            pbar.set_postfix(loss="NaN-grad")
            if wandb_on and WANDB_OK:
                wandb.log({"train/loss": float('nan')}, step=gstep)
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
    model.eval()
    plin_all, pgam_all, sgam_all = [], [], []
    needs_rgb = (model.mode == "oracle_flow")

    for bi, batch in enumerate(tqdm(loader, desc="eval", leave=False,
                                    disable=not _TTY)):
        spad  = batch["spad_mono_seq"].to(device)
        gt_s  = batch["target_s"].unsqueeze(1).to(device)
        pct99 = batch["pct99"].to(device)
        rgb   = batch["clean_rgb_seq"].to(device) if needs_rgb else None

        with autocast("cuda"):
            pred, _ = model(spad, clean_rgb=rgb)

        plin, pgam, sgam = evaluate_batch(pred, gt_s, pct99)
        plin_all.extend(plin); pgam_all.extend(pgam); sgam_all.extend(sgam)
        if max_batches > 0 and (bi + 1) >= max_batches:
            break

    def _safe(lst):
        import numpy as np
        return float(np.mean(lst)) if lst else float('nan')

    return dict(psnr_lin=_safe(plin_all),
                psnr_gam=_safe(pgam_all),
                ssim_gam=_safe(sgam_all))


def print_results(res, tag="EVAL"):
    print(f"\n{'='*52}")
    print(f"  {tag}")
    print(f"  PSNR (lin) : {res['psnr_lin']:>7.4f} dB")
    print(f"  PSNR (gam) : {res['psnr_gam']:>7.4f} dB  <- paper metric")
    print(f"  SSIM (gam) : {res['ssim_gam']:>7.4f}")
    print(f"{'='*52}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Load and merge config
    cfg = load_config(args.config) if os.path.isfile(args.config) else {}
    cfg.setdefault("model", {})
    cfg.setdefault("data",  {})
    cfg.setdefault("train", {})
    cfg = apply_cli_overrides(cfg, args)

    mc = cfg["model"]; dc = cfg["data"]; tc = cfg["train"]

    # Seed
    seed = args.seed
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mode         = mc.get("mode", "dcn_h4")
    use_grad_loss = mc.get("grad_loss", True)
    epochs        = tc.get("epochs", 100)
    steps_per_ep  = tc.get("steps_per_epoch", 2000)
    batch_size    = tc.get("batch", 4)
    lr            = tc.get("lr", 2e-4)
    weight_decay  = tc.get("weight_decay", 1e-4)
    betas         = tuple(tc.get("betas", [0.9, 0.9]))
    lr_min        = tc.get("lr_min", 1e-6)
    save_every    = tc.get("save_every", 5)
    wandb_project = tc.get("wandb_project", "spad-net")
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
    print(f"  SPADNet  |  mode={mode}  |  grad_loss={use_grad_loss}")
    print(f"  device={device}   epochs={epochs}   batch={batch_size}")
    print(f"  target: raw alpha*S photon counts (mean=PPP={ppp})")
    print(f"{'='*60}\n")

    # Build model
    model = SPADNet(
        T=mc.get("T", SEQ_LEN),
        bc=mc.get("base_channels", 32),
        nb=mc.get("n_blocks", 2),
        nfpm=mc.get("n_fpm", 2),
        mode=mode,
        raft_ckpt=mc.get("raft_ckpt", ""),
    ).to(device)

    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    n_tt = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  trainable={n_tr:.3f} M   total={n_tt:.3f} M\n")

    # Load pretrained weights (not a full resume)
    if not args.resume and args.ckpt and os.path.isfile(args.ckpt):
        ck = load_checkpoint(args.ckpt, device)
        model.load_state_dict(ck["model"], strict=False)
        print(f"  [loaded] {args.ckpt}")

    # W&B
    use_wb = (not args.no_wandb) and WANDB_OK
    if use_wb:
        run_name = args.wandb_run or f"{mode}_{'grad' if use_grad_loss else 'nograd'}"
        wandb.init(project=wandb_project, name=run_name, config={**mc, **dc, **tc})

    # Data
    tr_root = args.data_root
    if not tr_root:
        raise RuntimeError("--data_root is required")
    if os.path.isdir(os.path.join(tr_root, "train")):
        tr_root = os.path.join(tr_root, "train")

    def _ds(samples, aug, rp=False, ec=None):
        return SPADDataset(
            samples, ppp=ppp, bins=bins,
            augment=aug, crop=crop,
            eval_crop=(eval_crop if ec is None else ec),
            seed=seed, random_pick=rp, sps=sps_eff)

    if args.eval_only:
        from src import build_test
        ts_root = os.path.join(args.data_root, "test") \
                  if os.path.isdir(os.path.join(args.data_root, "test")) else tr_root
        src = build_test(ts_root)
        sps_eff = 1
        res = run_eval(model,
                       make_loader(_ds(src, False, ec=eval_crop), 1, False, workers),
                       device, args.eval_batches)
        print_results(res, f"TEST | {mode}")
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

    tr_ld = make_loader(_ds(tr_s, True, rp=True), batch_size, True,
                        workers, drop_last=True)
    ev_ld = make_loader(_ds(ev_s, False), 1, False, workers) if ev_s else None
    steps = steps_per_ep if steps_per_ep > 0 else len(tr_ld)

    criterion = CharbonnierLoss()
    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr, weight_decay=weight_decay, betas=betas)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, lr_min)
    scaler = GradScaler("cuda")
    best   = 0.
    gstep  = 0
    start_epoch = 0

    # Full resume
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
        tloss, gstep = train_epoch(
            model, tr_ld, opt, scaler, criterion, use_grad_loss,
            device, ep, steps, use_wb, gstep)
        sched.step()

        pg = sg = float('nan')
        if ev_ld:
            res = run_eval(model, ev_ld, device, args.eval_batches)
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

        state = dict(epoch=ep, gstep=gstep, model=model.state_dict(),
                     optimizer=opt.state_dict(), scheduler=sched.state_dict(),
                     best_psnr_gam=best, mode=mode,
                     use_grad_loss=use_grad_loss)

        save_checkpoint(state, os.path.join(args.ckpt_dir, "latest.pth"))
        if is_best:
            save_checkpoint(state, os.path.join(args.ckpt_dir, "best.pth"))
        elif ep % save_every == 0:
            save_checkpoint(state, os.path.join(args.ckpt_dir, f"ep{ep:04d}.pth"))

    if use_wb:
        wandb.finish()


if __name__ == "__main__":
    main()
