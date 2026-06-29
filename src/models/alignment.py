"""
Alignment modules for SPADNet.

DCNv2             — deformable conv primitive (always float32, FIX-2a/b/c)
GateFusion        — gated residual fusion
PairwiseDCNAlign  — pairwise feature-space DCN (dcn_h2/h4/h8/h16)
PreDenoiser       — light denoising head for SpyNet input
SpyNet            — 6-level 1-channel SpyNet (FIX-2a: float32 accumulation)
SpyNetRGB         — 6-level 3-channel SpyNet (oracle fallback)
FlowGuidedDCNAlign — flow-prior DCN (spynet_dcn, oracle_flow)
CascadeLevel      — single level of hierarchical cascade
GroupedSpatialShift
ShiftNetAlign
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d as tv_dcn

from .backbone import NAFBlock


# ── DCN primitive ─────────────────────────────────────────────────────────────

class DCNv2(nn.Module):
    """
    Deformable Conv v2 primitive.
    Always executes in float32 — FIX-2a: avoids fp16 overflow in deform_conv2d.
    """
    def __init__(self, ic, oc, k=3, pad=1, dg=8):
        super().__init__()
        self.pad = pad
        self.dg  = dg
        self.w   = nn.Parameter(torch.zeros(oc, ic, k, k))
        self.b   = nn.Parameter(torch.zeros(oc))
        nn.init.kaiming_uniform_(self.w, a=0.1)

    def forward(self, x, off, msk):
        xf  = x.float()
        of  = off.float()                    # FIX-2a: float32
        mf  = torch.sigmoid(msk).float()     # FIX-2a: float32
        out = tv_dcn(xf, of, self.w.float(), self.b.float(),
                     padding=(self.pad, self.pad), mask=mf)
        return out.to(x.dtype)


# ── Gated residual fusion ──────────────────────────────────────────────────────

class GateFusion(nn.Module):
    """Gated residual fusion: w * f + (1-w) * c, w predicted from [f; c]."""
    def __init__(self, c):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(2 * c, 2 * c, 3, 1, 1, groups=2 * c),
            nn.Conv2d(2 * c, 1, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, f, c):
        w = torch.sigmoid(self.gate(torch.cat([f, c], 1)))
        return w * f + (1 - w) * c


# ── Pairwise feature DCN ──────────────────────────────────────────────────────

class PairwiseDCNAlign(nn.Module):
    """
    Pairwise feature-space DCN alignment.
    Used for dcn_h2 (align at H/2), dcn_h4 (H/4), dcn_h8 (H/8), dcn_h16 (H/16).
    """
    def __init__(self, T, c, dg=8):
        super().__init__()
        K = 9
        self.T   = T
        self.ctr = T // 2
        self.oc1  = nn.Conv2d(c * 2, c, 3, 1, 1)
        self.oc2  = nn.Conv2d(c, dg * K * 3, 3, 1, 1)
        nn.init.zeros_(self.oc2.weight)
        nn.init.zeros_(self.oc2.bias)
        self.lr   = nn.LeakyReLU(0.1, True)
        self.dcn  = DCNv2(c, c, 3, 1, dg)
        self.pbff = nn.Conv2d(T * c, c, 1)
        self.beta = nn.Parameter(torch.full((1, T * c, 1, 1), 0.1))
        self.reg  = GateFusion(c)

    def forward(self, fl):
        ctr = fl[self.ctr]
        al  = []
        for t, ft in enumerate(fl):
            if t == self.ctr:
                al.append(ctr)
                continue
            o = self.oc2(self.lr(self.oc1(torch.cat([ft, ctr], 1))))
            al.append(self.dcn(ft, o[:, :144], o[:, 144:]))
        return self.reg(ctr + self.pbff(torch.cat(al, 1) * self.beta), ctr)


# ── PreDenoiser ───────────────────────────────────────────────────────────────

class PreDenoiser(nn.Module):
    """Light denoising head that preprocesses PixelUnshuffle(2) SPAD patches."""
    def __init__(self, c=32, nb=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, c, 3, 1, 1),
            *[NAFBlock(c) for _ in range(nb)],
            nn.Conv2d(c, 4, 3, 1, 1))

    def forward(self, x):
        return torch.clamp(x + self.net(x), 0., 1.)


# ── SpyNet (1-channel) ────────────────────────────────────────────────────────

class _SpyBasic(nn.Module):
    def __init__(self, ic):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ic, 32, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(64, 32, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(32, 16, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(16,  2, 3, 1, 1))

    def forward(self, x):
        return self.net(x)


def _spy_warp(x, flow):
    B, C, H, W = x.shape
    gy, gx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=x.device),
        torch.arange(W, dtype=torch.float32, device=x.device),
        indexing='ij')
    f  = flow.float()
    nx = (gx + f[:, 0]) / max(W - 1, 1) * 2 - 1
    ny = (gy + f[:, 1]) / max(H - 1, 1) * 2 - 1
    return F.grid_sample(x.float(), torch.stack([nx, ny], -1),
                         mode='bilinear', align_corners=True,
                         padding_mode='zeros').to(x.dtype)


class SpyNet(nn.Module):
    """6-level 1-channel SpyNet. FIX-2a: float32 flow accumulation."""
    def __init__(self):
        super().__init__()
        self.levels = nn.ModuleList([_SpyBasic(4) for _ in range(6)])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Zero-init final layer to prevent flow divergence at init
        for level in self.levels:
            nn.init.zeros_(level.net[-1].weight)
            if level.net[-1].bias is not None:
                nn.init.zeros_(level.net[-1].bias)

    def forward(self, ref, supp):
        ref_f  = ref.float()
        supp_f = supp.float()
        rp = [ref_f];  sp = [supp_f]
        for _ in range(5):
            rp.append(F.avg_pool2d(rp[-1], 2, 2))
            sp.append(F.avg_pool2d(sp[-1], 2, 2))
        B, _, Hl, Wl = rp[-1].shape
        flow = torch.zeros(B, 2, Hl, Wl, device=ref.device, dtype=torch.float32)
        for i in range(6):
            rl = rp[5 - i]; sl = sp[5 - i]; _, _, H, W = rl.shape
            if i > 0:
                flow = F.interpolate(flow, size=(H, W),
                                     mode='bilinear', align_corners=False) * 2.
            sw    = _spy_warp(sl, flow)
            delta = self.levels[i](torch.cat([rl, sw, flow], 1)).float()
            flow  = flow + delta
        return flow


class SpyNetRGB(nn.Module):
    """6-level 3-channel SpyNet (oracle fallback). FIX-2a: float32 flow accumulation."""
    def __init__(self):
        super().__init__()
        self.levels = nn.ModuleList([_SpyBasic(8) for _ in range(6)])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Zero-init final layer
        for level in self.levels:
            nn.init.zeros_(level.net[-1].weight)
            if level.net[-1].bias is not None:
                nn.init.zeros_(level.net[-1].bias)

    def forward(self, ref, supp):
        ref_f  = ref.float()
        supp_f = supp.float()
        rp = [ref_f];  sp = [supp_f]
        for _ in range(5):
            rp.append(F.avg_pool2d(rp[-1], 2, 2))
            sp.append(F.avg_pool2d(sp[-1], 2, 2))
        B, _, Hl, Wl = rp[-1].shape
        flow = torch.zeros(B, 2, Hl, Wl, device=ref.device, dtype=torch.float32)
        for i in range(6):
            rl = rp[5 - i]; sl = sp[5 - i]; _, _, H, W = rl.shape
            if i > 0:
                flow = F.interpolate(flow, size=(H, W),
                                     mode='bilinear', align_corners=False) * 2.
            sw    = _spy_warp(sl, flow)
            delta = self.levels[i](torch.cat([rl, sw, flow], 1)).float()
            flow  = flow + delta
        return flow


# ── Flow-guided DCN ───────────────────────────────────────────────────────────

class FlowGuidedDCNAlign(nn.Module):
    """
    Flow-guided pairwise DCN. Flow offsets are used as prior for DCN offsets.
    Used for spynet_dcn and oracle_flow modes.
    """
    def __init__(self, T, c, dg=8):
        super().__init__()
        K = 9
        self.T   = T
        self.ctr = T // 2
        self.GK  = dg * K
        self.oc1  = nn.Conv2d(c * 2, c, 3, 1, 1)
        self.oc2  = nn.Conv2d(c, dg * K * 3, 3, 1, 1)
        nn.init.zeros_(self.oc2.weight)
        nn.init.zeros_(self.oc2.bias)
        self.lr   = nn.LeakyReLU(0.1, True)
        self.dcn  = DCNv2(c, c, 3, 1, dg)
        self.pbff = nn.Conv2d(T * c, c, 1)
        self.beta = nn.Parameter(torch.full((1, T * c, 1, 1), 0.1))
        self.reg  = GateFusion(c)

    def _prior(self, flow):
        B, _, H, W = flow.shape
        GK = self.GK
        p = torch.empty(B, 2 * GK, H, W, device=flow.device, dtype=flow.dtype)
        p[:, 0::2] = flow[:, 1:2].expand(-1, GK, -1, -1)   # dy → row offsets
        p[:, 1::2] = flow[:, 0:1].expand(-1, GK, -1, -1)   # dx → col offsets
        return p

    def forward(self, fl, flow_list=None):
        ctr = fl[self.ctr]
        al  = []
        for t, ft in enumerate(fl):
            if t == self.ctr:
                al.append(ctr)
                continue
            o = self.oc2(self.lr(self.oc1(torch.cat([ft, ctr], 1))))
            res, msk = o[:, :self.GK * 2], o[:, self.GK * 2:]
            off = (self._prior(flow_list[t]) + res) \
                  if (flow_list and flow_list[t] is not None) else res
            al.append(self.dcn(ft, off, msk))
        return self.reg(ctr + self.pbff(torch.cat(al, 1) * self.beta), ctr)


# ── Cascade level ─────────────────────────────────────────────────────────────

class CascadeLevel(nn.Module):
    """Single level of the hierarchical cascade alignment."""
    def __init__(self, in_c, val_level=False, bc=32):
        super().__init__()
        self.val  = val_level
        self.oc1  = nn.Conv2d(in_c * 2, in_c, 3, 1, 1)
        self.oc2  = nn.Conv2d(in_c, 8 * 9 * 3, 3, 1, 1)
        nn.init.zeros_(self.oc2.weight)
        nn.init.zeros_(self.oc2.bias)
        self.lr  = nn.LeakyReLU(0.1, True)
        self.dcn = DCNv2(bc if val_level else in_c,
                         bc if val_level else in_c, 3, 1, 8)

    def forward(self, fn, fc, vn=None, prior=None):
        o = self.oc2(self.lr(self.oc1(torch.cat([fn, fc], 1))))
        res, msk = o[:, :144], o[:, 144:]
        off = (prior + res) if prior is not None else res
        return off, self.dcn(vn if (self.val and vn is not None) else fn, off, msk)


# ── ShiftNet alignment ────────────────────────────────────────────────────────

class GroupedSpatialShift(nn.Module):
    """
    Grouped Spatial Shift from the ShiftNet paper.
    Splits features into 8 groups, shifts each group in a cardinal/diagonal
    direction, then applies a depth-wise 3x3 smoothing convolution.
    """
    def __init__(self, c, base_shift=5):
        super().__init__()
        self.shifts = [
            (-base_shift, -base_shift), (-base_shift, 0), (-base_shift, base_shift),
            (0,           -base_shift),                   (0,           base_shift),
            (base_shift,  -base_shift), (base_shift,  0), (base_shift,  base_shift),
        ]
        self.M       = len(self.shifts)
        self.slice_c = c // self.M
        assert c % self.M == 0, f"Channels {c} must be divisible by {self.M}"
        self.dw_conv = nn.Conv2d(c, c, 3, 1, 1, groups=c)

    def forward(self, x):
        B, C, H, W = x.shape
        out = torch.zeros_like(x)
        for i, (dx, dy) in enumerate(self.shifts):
            c_start = i * self.slice_c
            c_end   = c_start + self.slice_c
            slice_  = x[:, c_start:c_end]
            pad_l = max(dx,  0); pad_r = max(-dx, 0)
            pad_t = max(dy,  0); pad_b = max(-dy, 0)
            padded = F.pad(slice_, (pad_l, pad_r, pad_t, pad_b))
            y0 = max(-dy, 0); y1 = y0 + H
            x0 = max(-dx, 0); x1 = x0 + W
            out[:, c_start:c_end] = padded[:, :, y0:y1, x0:x1]
        return self.dw_conv(out)


class ShiftNetAlign(nn.Module):
    """
    ShiftNet-style alignment adapted to the 11-frame star-pattern.
    Direct replacement for PairwiseDCNAlign.
    """
    def __init__(self, T, c, base_shift=5):
        super().__init__()
        self.T   = T
        self.ctr = T // 2
        self.gss = GroupedSpatialShift(c, base_shift)
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(c * 2, c, 5, 1, 2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(c, c, 3, 1, 1))
        self.pbff = nn.Conv2d(T * c, c, 1)
        self.beta = nn.Parameter(torch.full((1, T * c, 1, 1), 0.1))
        self.reg  = GateFusion(c)

    def forward(self, fl):
        ctr_feat = fl[self.ctr]
        al = []
        for t, ft in enumerate(fl):
            if t == self.ctr:
                al.append(ctr_feat)
                continue
            shifted_ft = self.gss(ft)
            fused_pair = self.fuse_conv(torch.cat([shifted_ft, ctr_feat], dim=1))
            al.append(fused_pair)
        out = ctr_feat + self.pbff(torch.cat(al, 1) * self.beta)
        return self.reg(out, ctr_feat)
