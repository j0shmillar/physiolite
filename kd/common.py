from __future__ import annotations

import importlib.util
import math
import os
import random
import sys
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def fp32(x: torch.Tensor) -> torch.Tensor:
    return x.float() if isinstance(x, torch.Tensor) and x.dtype != torch.float32 else x


def autocast_ctx(*, enabled: bool, dtype: torch.dtype):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", dtype=dtype)
    return torch.cuda.amp.autocast(dtype=dtype)


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def make_posenc_1d_concat(num_freq: int, T: int, device, dtype=torch.float32):
    num_freq = int(num_freq)
    if num_freq <= 0:
        return None

    ts = torch.linspace(0, 1, steps=T, device=device, dtype=dtype)
    chans = []
    for k in range(num_freq):
        f = 2.0 ** k
        s = torch.sin(2.0 * math.pi * f * ts)
        c = torch.cos(2.0 * math.pi * f * ts)
        chans.extend([s, c])
    return torch.stack(chans, dim=0)


def parse_kernel_set(s: str) -> tuple[int, ...]:
    if s is None:
        return (3, 5, 7)
    s = str(s).strip()
    if not s:
        return (3, 5, 7)
    parts = [p.strip() for p in s.split(",") if p.strip()]
    ks = tuple(int(p) for p in parts)
    if len(ks) == 0:
        raise ValueError("Empty --student_kernel_set. Use e.g. '3,5,7' or '7'.")
    for k in ks:
        if k <= 0 or (k % 2 == 0):
            raise ValueError(f"Invalid kernel size {k} in --student_kernel_set (must be positive odd).")
    return ks


def unwrap_ddp(m):
    return m.module if hasattr(m, "module") else m


def zscore_per_sample_channel(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mu = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True).clamp_min(eps)
    return (x - mu) / std


def load_repo_module(repo_root: str, module_relpath: str, module_name: str):
    module_path = os.path.join(repo_root, module_relpath)
    if not os.path.isfile(module_path):
        raise FileNotFoundError(f"Module not found: {module_path}")

    repo_root = os.path.abspath(repo_root)
    try:
        sys.path.insert(0, repo_root)
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        return mod
    finally:
        if sys.path and sys.path[0] == repo_root:
            sys.path.pop(0)


def make_student_input(
    x: torch.Tensor,
    *,
    arch: str,
    pe_cache: torch.Tensor | None,
    pe_scale: float,
):
    if arch == "physiowavenpu":
        if pe_cache is None:
            return x
        pe = pe_cache.unsqueeze(0).expand(x.size(0), -1, -1)
        pe = pe_scale * pe
        return torch.cat([x, pe], dim=1)
    if arch in ("waveformer", "tinymyo"):
        return x.unsqueeze(1)
    if arch == "otis":
        x4 = x.unsqueeze(1)
        pw = 32
        T = x4.size(-1)
        pad_t = (pw - (T % pw)) % pw
        if pad_t:
            x4 = F.pad(x4, (0, pad_t))
        return x4
    if arch in ("ecgfounder", "clef", "ecgfm"):
        return x
    if arch == "hubertecg":
        return x[:, 0, :]
    if arch == "physiowave":
        patch_t = 64
        T = x.size(-1)
        pad_t = (patch_t - (T % patch_t)) % patch_t
        if pad_t:
            x = F.pad(x, (0, pad_t))
        return x
    if arch in ("fcn", "tcn", "bilstm", "convnext1d", "ai85tcn1d"):
        return x

    raise ValueError(f"Unknown student arch: {arch}")


def patch_wavelet_modules_io(model: nn.Module, *, rank: int = 0) -> dict:
    counts = {"conv1d": 0, "linear": 0}

    def _pre_hook(mod, inputs):
        if not inputs:
            return
        x = inputs[0]
        if not torch.is_tensor(x):
            return
        p = next(mod.parameters(), None)
        if p is None:
            return
        dev = p.device
        dt = p.dtype
        if x.device != dev:
            x = x.to(dev, non_blocking=True)
        if x.dtype != dt:
            x = x.to(dt)
        return (x,) + tuple(inputs[1:])

    for name, m in model.named_modules():
        if "wavelet_decomp" not in name:
            continue
        if isinstance(m, nn.Conv1d):
            m.register_forward_pre_hook(_pre_hook)
            counts["conv1d"] += 1
        elif isinstance(m, nn.Linear):
            m.register_forward_pre_hook(_pre_hook)
            counts["linear"] += 1

    if rank == 0:
        print(f">> Patched wavelet IO: hooked Conv1d={counts['conv1d']} Linear={counts['linear']}")
    return counts
