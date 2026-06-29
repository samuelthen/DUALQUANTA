"""
SPADNet — main model.

Reconstructs a clean image from T=11 SPAD quanta frames.
Output: raw photon counts alpha*S via F.softplus (non-negative, unbounded).

Mode naming (public → internal logic):
  single_frame  → single-frame baseline (no temporal fusion)
  no_align      → 11-frame concat, no alignment
  dcn_h2        → pairwise DCN at H/2  (was dcn_d0)
  dcn_h4        → pairwise DCN at H/4  (was dcn_d1) [PROPOSED / QuantaColor]
  dcn_h8        → pairwise DCN at H/8  (was dcn_d2)
  dcn_h16       → pairwise DCN at H/16 (was dcn_d3)
  spynet_dcn    → PreDenoiser + SpyNet + flow-guided DCN
  cascading_dcn → hierarchical cascade H/8→H/4→H/2  [QuantaColor+]
  oracle_flow   → frozen RAFT + flow-guided DCN
  shift_net     → grouped spatial shift alignment
"""

import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from .backbone import NAFBlock, DownBlock, UpBlock
from .alignment import (
    DCNv2, GateFusion,
    PairwiseDCNAlign, PreDenoiser,
    SpyNet, SpyNetRGB, FlowGuidedDCNAlign,
    CascadeLevel, ShiftNetAlign,
)

SEQ_LEN  = 11
HALF_WIN = SEQ_LEN // 2

_MODES = (
    "single_frame", "no_align",
    "dcn_h2", "dcn_h4", "dcn_h8", "dcn_h16",
    "spynet_dcn", "cascading_dcn", "oracle_flow", "shift_net",
)


class SPADNet(nn.Module):
    """
    SPADNet: SPAD quanta-image reconstruction network.

    Args:
        T:             sequence length (default 11)
        bc:            base channel count (default 32)
        nb:            NAFBlocks per encoder/decoder level (default 2)
        nfpm:          NAFBlocks in the frame-processing module (default 2)
        mode:          alignment strategy (see _MODES)
        raft_ckpt:     path to RAFT checkpoint for oracle_flow mode
    """

    MODES = _MODES

    def __init__(self, T=SEQ_LEN, bc=32, nb=2, nfpm=2, mode="dcn_h4", raft_ckpt="",
                 out_ch=1, target_mode="luma"):
        super().__init__()
        assert mode in _MODES, f"Unknown mode '{mode}'. Choose from {_MODES}"
        self.T           = T
        self.C           = T // 2
        self.mode        = mode
        self.bc          = bc
        self.out_ch      = int(out_ch)
        self.target_mode = target_mode
        c1, c2, c3, c4 = bc, bc * 2, bc * 4, bc * 8
        self.c1 = c1; self.c2 = c2; self.c3 = c3; self.c4 = c4

        # ── Encoder shared stem ─────────────────────────────────────────────
        self.pus  = nn.PixelUnshuffle(2)
        self.stem = nn.Conv2d(4, c1, 3, 1, 1)
        self.fpm  = nn.Sequential(*[NAFBlock(c1) for _ in range(nfpm)])

        # ── U-Net encoder / bottleneck / decoder ─────────────────────────────
        self.dn1 = DownBlock(c1, c2, nb)
        self.dn2 = DownBlock(c2, c3, nb)
        self.dn3 = DownBlock(c3, c4, nb)
        self.mid = nn.Sequential(*[NAFBlock(c4) for _ in range(nb * 2)])
        self.up3 = UpBlock(c4, c3, c3, nb)
        self.up2 = UpBlock(c3, c2, c2, nb)
        self.up1 = UpBlock(c2, c1, c1, nb)

        # Head: raw feature → softplus activation applied in _out()
        self.head = nn.Sequential(
            nn.Conv2d(c1, c1 * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(c1, self.out_ch, 3, 1, 1))

        # ── Mode-specific alignment modules ──────────────────────────────────
        if mode == "single_frame":
            pass   # no alignment — uses only centre frame
        elif mode == "no_align":
            self.pbff_a = nn.Conv2d(T * c1, c1, 1)
        elif mode == "dcn_h2":
            self.align = PairwiseDCNAlign(T, c1)
        elif mode == "dcn_h4":
            self.align = PairwiseDCNAlign(T, c2)
        elif mode == "dcn_h8":
            self.align = PairwiseDCNAlign(T, c3)
        elif mode == "dcn_h16":
            self.align = PairwiseDCNAlign(T, c4)
        elif mode == "spynet_dcn":
            self.denoiser = PreDenoiser(c1, nb=4)
            self.spynet   = SpyNet()
            self.align    = FlowGuidedDCNAlign(T, c1)
        elif mode == "cascading_dcn":
            self.cas3 = CascadeLevel(c3, bc=c1)
            self.cas2 = CascadeLevel(c2, bc=c1)
            self.cas1 = CascadeLevel(c1, val_level=True, bc=c1)
            self.pf3  = nn.Conv2d(T * c3, c3, 1)
            self.pf2  = nn.Conv2d(T * c2, c2, 1)
            self.pf1  = nn.Conv2d(T * c1, c1, 1)
            self.b3   = nn.Parameter(torch.full((1, T * c3, 1, 1), 0.1))
            self.b2   = nn.Parameter(torch.full((1, T * c2, 1, 1), 0.1))
            self.b1   = nn.Parameter(torch.full((1, T * c1, 1, 1), 0.1))
            self.rexp = nn.Conv2d(4, c1, 1, bias=False)
            self.reg  = GateFusion(c1)
        elif mode == "shift_net":
            self.align = ShiftNetAlign(T, c2)
        elif mode == "oracle_flow":
            self._init_oracle(raft_ckpt)
            self.align = FlowGuidedDCNAlign(T, c1)

    # ── Oracle RAFT init ──────────────────────────────────────────────────────

    def _init_oracle(self, raft_ckpt=""):
        self._raft_ok = False
        self.raft     = None
        if raft_ckpt and os.path.isfile(raft_ckpt):
            try:
                from torchvision.models.optical_flow import raft_large
                m = raft_large(weights=None)
                m.load_state_dict(torch.load(raft_ckpt, map_location='cpu',
                                             weights_only=True))
                for p in m.parameters():
                    p.requires_grad_(False)
                m._raft_transforms = None
                self.raft = m
                self._raft_ok = True
                print(f"[Oracle] Frozen RAFT-Large from {raft_ckpt}")
                return
            except Exception as e:
                warnings.warn(f"[Oracle] local ckpt failed ({e}), trying download")
        try:
            from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
            weights = Raft_Large_Weights.DEFAULT
            m = raft_large(weights=weights)
            for p in m.parameters():
                p.requires_grad_(False)
            m._raft_transforms = weights.transforms()
            self.raft = m
            self._raft_ok = True
            print("[Oracle] Frozen RAFT-Large (pretrained)")
        except Exception as e:
            warnings.warn(f"[Oracle] RAFT unavailable ({e}). Using SpyNetRGB fallback.")
            self.raft_fb = SpyNetRGB()

    def train(self, mode=True):
        super().train(mode)
        if self.mode == "oracle_flow" and getattr(self, '_raft_ok', False) \
                and self.raft is not None:
            self.raft.eval()
        return self

    # ── Output activation ─────────────────────────────────────────────────────

    def _out(self, x):
        """
        Predict raw photon counts alpha*S via softplus.
        Cast to float32 before softplus: under fp16, exp(x) overflows for x>88.
        """
        return F.softplus(self.head(x).float())

    # ── Encoder helpers ───────────────────────────────────────────────────────

    def _enc(self, fr_BHW):
        raw = self.pus(fr_BHW.unsqueeze(1))
        return raw, self.fpm(self.stem(raw))

    def _enc_batch(self, spad):
        B, T, H, W = spad.shape
        raw = self.pus(spad.reshape(B * T, H, W).unsqueeze(1))
        return raw, self.fpm(self.stem(raw)), B, T, H, W

    def _decode(self, x, s3, s2, s1):
        x = self.mid(x)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        return self._out(x)

    # ── Forward dispatch ──────────────────────────────────────────────────────

    def forward(self, spad, clean_rgb=None):
        """
        Args:
            spad:      (B, T, H, W) SPAD quanta frames in [0, 1]
            clean_rgb: (B, T, 3, H, W) clean RGB (only for oracle_flow mode)

        Returns:
            pred: (B, 1, H, W) raw photon counts alpha*S
            None: reserved for auxiliary outputs (currently unused)
        """
        m = self.mode
        if   m == "single_frame":   return self._single(spad)
        elif m == "no_align":       return self._no_align(spad)
        elif m == "dcn_h2":         return self._dcn_h2(spad)
        elif m == "dcn_h4":         return self._dcn_h4(spad)
        elif m == "dcn_h8":         return self._dcn_h8(spad)
        elif m == "dcn_h16":        return self._dcn_h16(spad)
        elif m == "shift_net":      return self._shift_net(spad)
        elif m == "spynet_dcn":     return self._spynet(spad)
        elif m == "cascading_dcn":  return self._cascade(spad)
        elif m == "oracle_flow":
            assert clean_rgb is not None, "oracle_flow requires clean_rgb"
            return self._oracle(spad, clean_rgb)

    # ── Mode implementations ──────────────────────────────────────────────────

    def _single(self, spad):
        _, f = self._enc(spad[:, self.C])
        x, s1 = self.dn1(f)
        x, s2 = self.dn2(x)
        x, s3 = self.dn3(x)
        return self._decode(x, s3, s2, s1), None

    def _no_align(self, spad):
        raw, f0, B, T, H, W = self._enc_batch(spad)
        fused = self.pbff_a(f0.reshape(B, T * self.c1, H // 2, W // 2))
        x, s1 = self.dn1(fused)
        x, s2 = self.dn2(x)
        x, s3 = self.dn3(x)
        return self._decode(x, s3, s2, s1), None

    def _dcn_h2(self, spad):
        """DCN alignment at H/2 (stem feature level)."""
        raw, f0, B, T, H, W = self._enc_batch(spad)
        fl    = list(f0.reshape(B, T, self.c1, H // 2, W // 2).unbind(1))
        fused = self.align(fl)
        x, s1 = self.dn1(fused)
        x, s2 = self.dn2(x)
        x, s3 = self.dn3(x)
        return self._decode(x, s3, s2, s1), None

    def _dcn_h4(self, spad):
        """DCN alignment at H/4 (after first downblock). [PROPOSED / QuantaColor]"""
        raw, f0, B, T, H, W = self._enc_batch(spad)
        h1, s1 = self.dn1(f0)
        hl     = list(h1.reshape(B, T, self.c2, H // 4, W // 4).unbind(1))
        s1c    = s1.reshape(B, T, self.c1, H // 2, W // 2)[:, self.C]
        fused  = self.align(hl)
        x, s2  = self.dn2(fused)
        x, s3  = self.dn3(x)
        x = self.mid(x)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1c)
        return self._out(x), None

    def _dcn_h8(self, spad):
        """DCN alignment at H/8 (after second downblock)."""
        raw, f0, B, T, H, W = self._enc_batch(spad)
        h1, s1 = self.dn1(f0)
        h2, s2 = self.dn2(h1)
        hl     = list(h2.reshape(B, T, self.c3, H // 8, W // 8).unbind(1))
        s1c    = s1.reshape(B, T, self.c1, H // 2, W // 2)[:, self.C]
        s2c    = s2.reshape(B, T, self.c2, H // 4, W // 4)[:, self.C]
        fused  = self.align(hl)
        x, s3  = self.dn3(fused)
        x = self.mid(x)
        x = self.up3(x, s3)
        x = self.up2(x, s2c)
        x = self.up1(x, s1c)
        return self._out(x), None

    def _dcn_h16(self, spad):
        """DCN alignment at H/16 (after third downblock)."""
        raw, f0, B, T, H, W = self._enc_batch(spad)
        h1, s1 = self.dn1(f0)
        h2, s2 = self.dn2(h1)
        h3, s3 = self.dn3(h2)
        hl     = list(h3.reshape(B, T, self.c4, H // 16, W // 16).unbind(1))
        s1c    = s1.reshape(B, T, self.c1, H // 2,  W // 2 )[:, self.C]
        s2c    = s2.reshape(B, T, self.c2, H // 4,  W // 4 )[:, self.C]
        s3c    = s3.reshape(B, T, self.c3, H // 8,  W // 8 )[:, self.C]
        fused  = self.align(hl)
        x = self.mid(fused)
        x = self.up3(x, s3c)
        x = self.up2(x, s2c)
        x = self.up1(x, s1c)
        return self._out(x), None

    def _shift_net(self, spad):
        """ShiftNet alignment at H/4 feature level."""
        raw, f0, B, T, H, W = self._enc_batch(spad)
        h1, s1 = self.dn1(f0)
        hl     = list(h1.reshape(B, T, self.c2, H // 4, W // 4).unbind(1))
        s1c    = s1.reshape(B, T, self.c1, H // 2, W // 2)[:, self.C]
        fused  = self.align(hl)
        x, s2  = self.dn2(fused)
        x, s3  = self.dn3(x)
        x = self.mid(x)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1c)
        return self._out(x), None

    def _spynet(self, spad):
        """SpyNet optical flow + flow-guided DCN at H/2."""
        raw, f0, B, T, H, W = self._enc_batch(spad)
        raw4 = raw.reshape(B, T, 4, H // 2, W // 2)
        fl   = list(f0.reshape(B, T, self.c1, H // 2, W // 2).unbind(1))
        dn   = [self.denoiser(raw4[:, t]).mean(1, keepdim=True).float()
                for t in range(T)]
        cg   = dn[self.C]
        fll  = [None] * T
        for t in range(T):
            if t != self.C:
                # Clamp to ±64px: prevents extreme offsets causing deform_conv2d overflow
                fll[t] = self.spynet(dn[t], cg).clamp(-64., 64.)
        fused = self.align(fl, fll)
        x, s1 = self.dn1(fused)
        x, s2 = self.dn2(x)
        x, s3 = self.dn3(x)
        return self._decode(x, s3, s2, s1), None

    def _cascade(self, spad):
        """
        Hierarchical cascade: H/8 → H/4 → H/2, each level refines flow prior.
        FIX-2b: cascade offset upsampling in float32.
        Cascade stops at H/8 (d8); H/16 alignment dropped for performance.
        """
        B, T, H, W = spad.shape
        s3s, s2s, s1s, vs = [], [], [], []
        for t in range(T):
            raw, f1 = self._enc(spad[:, t])
            vs.append(self.rexp(raw))
            f2, s1 = self.dn1(f1)
            f3, s2 = self.dn2(f2)
            s3s.append(f3)
            s2s.append(s2)
            s1s.append(s1)

        def fuse(cas, pbff, beta, fl, cl, pl=None):
            cf  = cl[self.C]
            al  = []; offs = []
            for t in range(T):
                if t == self.C:
                    al.append(cf); offs.append(None); continue
                o, a = cas(fl[t], cf, prior=pl[t] if pl else None)
                al.append(a); offs.append(o)
            return cf + pbff(torch.cat(al, 1) * beta), offs

        def _up(offs):
            # FIX-2b: float32 before interpolation
            return [None if o is None else
                    F.interpolate(o.float(), scale_factor=2,
                                  mode='bilinear', align_corners=False) * 2.
                    for o in offs]

        fu3, o3 = fuse(self.cas3, self.pf3, self.b3, s3s, s3s, None)
        fu2, o2 = fuse(self.cas2, self.pf2, self.b2, s2s, s2s, _up(o3))

        cf1 = s1s[self.C]; cv = vs[self.C]; al1 = []
        for t in range(T):
            if t == self.C:
                al1.append(cv)
                continue
            _, a = self.cas1(s1s[t], cf1, vn=vs[t], prior=_up(o2)[t])
            al1.append(a)
        fu1 = self.reg(cv + self.pf1(torch.cat(al1, 1) * self.b1), cf1)

        f4_ctr, s3_ctr = self.dn3(fu3)
        x = self.mid(f4_ctr)
        x = self.up3(x, s3_ctr)
        x = self.up2(x, fu2)
        x = self.up1(x, fu1)
        return self._out(x), None

    def _oracle(self, spad, clean_rgb):
        """
        Oracle optical flow from frozen RAFT on clean GT RGB.
        FIX-2c: RAFT runs inside autocast(enabled=False).
        """
        raw, f0, B, T, H, W = self._enc_batch(spad)
        fl      = list(f0.reshape(B, T, self.c1, H // 2, W // 2).unbind(1))
        ctr_rgb = clean_rgb[:, self.C]
        fll     = [None] * T
        for t in range(T):
            if t == self.C:
                continue
            nbr = clean_rgb[:, t]
            if self._raft_ok and self.raft is not None:
                with torch.no_grad(), autocast("cuda", enabled=False):
                    tr    = self.raft._raft_transforms
                    c_in, n_in = tr(ctr_rgb.float(), nbr.float())
                    c_in = c_in.to(spad.device)
                    n_in = n_in.to(spad.device)
                    flow = self.raft(c_in, n_in)[-1]
            else:
                flow = self.raft_fb(ctr_rgb, nbr)
            fll[t] = F.avg_pool2d(flow.float(), 2, 2) * 0.5
        fused = self.align(fl, fll)
        x, s1 = self.dn1(fused)
        x, s2 = self.dn2(x)
        x, s3 = self.dn3(x)
        return self._decode(x, s3, s2, s1), None
