"""
Utility functions: config loading, model complexity, checkpoint I/O.
"""

import os

import torch
import yaml


def load_config(path: str) -> dict:
    """
    Load YAML config with optional _base inheritance.

    If the config contains `_base: <filename>`, the base config is loaded
    first and then recursively merged with deep-merge semantics (leaf values
    in the child override those in the base).
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base_name = cfg.pop('_base', None)
    if base_name:
        base_path = os.path.join(os.path.dirname(os.path.abspath(path)), base_name)
        base = load_config(base_path)
        cfg  = _deep_merge(base, cfg)
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base`. Override wins on conflicts."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def model_complexity(model, H=256, W=256, T=11, device='cpu'):
    """
    Return (params_M, gflops) for the model.

    Requires fvcore for GFLOPs; falls back to nan if unavailable.
    """
    params = sum(p.numel() for p in model.parameters()) / 1e6
    try:
        from fvcore.nn import FlopCountAnalysis
        x     = torch.zeros(1, T, H, W, device=device)
        flops = FlopCountAnalysis(model, x).total() / 1e9
    except Exception:
        flops = float('nan')
    return params, flops


def save_checkpoint(state: dict, path: str):
    """Save a checkpoint dict to `path`, creating parent dirs as needed."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, device='cpu') -> dict:
    """Load a checkpoint saved with save_checkpoint / torch.save."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(path, map_location=device, weights_only=False)
