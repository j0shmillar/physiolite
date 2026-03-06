# kd/runner.py
"""
Knowledge Distillation for ECG/sEMG (single-label OR multilabel):
  Teacher: BERTWaveletTransformer (checkpoint)
  Student: one of:
    - physiowavenpu  (your NPU-friendly model)
    - waveformer     (ForeverBlue816/WaveFormer)
    - otis           (oetu/otis)  [real init + pretrained load from main_finetune]
    - tinymyo        (pulp-bio/BioFoundation) [Hydra instantiate(cfg.task,cfg) + pretrained load]

Key benchmark-alignment:
  - Do NOT override max_length for test
  - For WaveFormer-like models: expect inputs [B, 1, V, T]
  - For PhysioWaveNPU: [B, V+PE, T]

Notes:
  - OTiS pos_embed_x interpolation is done (like their code).
  - OTiS pos_embed_y loading: supports domain-specific and domain-agnostic checkpoints.
  - TinyMyo: you provide hydra config dir/name; we instantiate cfg.task only (no datamodule).
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
import argparse
import random
import json
import importlib.util
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, WeightedRandomSampler
from tqdm import tqdm

import h5py

from torch.cuda.amp import GradScaler

from dataclasses import dataclass

from sklearn.metrics import (
    accuracy_score,
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    hamming_loss, jaccard_score
)

# Your project imports
from model import BERTWaveletTransformer
from models.vit_pw import PhysioWaveNPU
from dataset_multilabel import (
    SingleLabelTimeSeriesDataset,
    MultiLabelTimeSeriesDataset,
    collate_multilabel_fn,
    parse_file_paths,
    collate_singlelabel_fn,
)

try:
    from scipy.signal import iirnotch, butter, filtfilt
    from scipy.signal import medfilt
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# NEW - TEMP
def debug_hf_output(out, prefix="hubert_out"):
    import torch

    print(f"\n[{prefix}] type={type(out)}")

    # dict-like keys
    try:
        keys = list(out.keys())
        print(f"[{prefix}] keys: {keys}")
        for k in keys:
            v = out[k]
            if isinstance(v, torch.Tensor):
                print(f"  - {k}: Tensor shape={tuple(v.shape)} dtype={v.dtype} device={v.device}")
            elif isinstance(v, (list, tuple)) and len(v) and isinstance(v[0], torch.Tensor):
                print(f"  - {k}: list/tuple of Tensors len={len(v)} first_shape={tuple(v[0].shape)}")
            else:
                print(f"  - {k}: {type(v)}")
    except Exception as e:
        print(f"[{prefix}] out.keys() not available: {e}")

    # common attrs
    for attr in ("last_hidden_state", "extract_features", "hidden_states", "attentions"):
        if hasattr(out, attr):
            v = getattr(out, attr)
            if isinstance(v, torch.Tensor):
                print(f"[{prefix}] attr.{attr}: Tensor shape={tuple(v.shape)} dtype={v.dtype} device={v.device}")
            elif isinstance(v, (list, tuple)) and len(v) and isinstance(v[0], torch.Tensor):
                print(f"[{prefix}] attr.{attr}: list/tuple len={len(v)} first_shape={tuple(v[0].shape)}")
            else:
                print(f"[{prefix}] attr.{attr}: {type(v)}")


# ----------------------------
# Utils
# ----------------------------

def fp32(x: torch.Tensor) -> torch.Tensor:
    return x.float() if isinstance(x, torch.Tensor) and x.dtype != torch.float32 else x

def autocast_ctx(*, enabled: bool, dtype: torch.dtype):
    """
    Torch AMP compatibility:
      - PyTorch >= 2.0: torch.amp.autocast(device_type="cuda", dtype=...)
      - Older: torch.cuda.amp.autocast(dtype=...)
    """
    if not enabled:
        return nullcontext()
    # Prefer torch.amp.autocast if present
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", dtype=dtype)
    # Fallback: torch.cuda.amp.autocast (no device_type kw)
    return torch.cuda.amp.autocast(dtype=dtype)

def collate_singlelabel_with_tlogits(batch):
    # batch: list of (x,y,t)
    xs = [(x, y) for (x, y, _t) in batch]
    x, y = collate_singlelabel_fn(xs)
    t = torch.stack([_t for (_x, _y, _t) in batch], dim=0)  # [B,C]
    return x, y, t

def patch_wavelet_modules_io(model: nn.Module, *, rank: int = 0) -> dict:
    """
    Runtime patch (no model edits):
    For wavelet_decomp submodules, registers forward_pre_hooks on Conv1d + Linear
    to ensure input tensor is moved to the module's device and cast to module's dtype.

    Returns counts of hooked modules.
    """
    counts = {"conv1d": 0, "linear": 0}

    def _pre_hook(mod, inputs):
        if not inputs:
            return
        x = inputs[0]
        if not torch.is_tensor(x):
            return
        # target device/dtype from parameters
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

def collate_multilabel_with_tlogits(batch):
    xs = [(x, y) for (x, y, _t) in batch]
    x, y = collate_multilabel_fn(xs)
    t = torch.stack([_t for (_x, _y, _t) in batch], dim=0)  # [B,C]
    return x, y, t

def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def make_posenc_1d_concat(num_freq: int, T: int, device, dtype=torch.float32):
    """
    Returns:
      - None if num_freq <= 0 (i.e., PosEnc disabled)
      - Tensor [2F, T] otherwise
    """
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
    return torch.stack(chans, dim=0)  # [2F, T]

def parse_kernel_set(s: str) -> tuple[int, ...]:
    """
    Examples:
      "3,5,7" -> (3,5,7)
      "7"     -> (7,)
      "3"     -> (3,)
    """
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
    """
    x: [B, V, T]
    z-score over time for each (B,V) channel separately.
    """
    mu = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True).clamp_min(eps)
    return (x - mu) / std

# ----------------------------
# Generic repo module loader
# ----------------------------
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

# ----------------------------
# Student input formatting
# ----------------------------
def make_student_input(
    x: torch.Tensor,
    *,
    arch: str,
    pe_cache: torch.Tensor | None,
    pe_scale: float,
):
    """
    x from your datasets: [B, V, T]

    Returns:
      physiowavenpu: [B, V+2F, T]  (concat PE)
      waveformer/otis/tinymyo: [B, 1, V, T]
    """
    if arch == "physiowavenpu":
        if pe_cache is None:
            return x
        pe = pe_cache.unsqueeze(0).expand(x.size(0), -1, -1)  # [B, 2F, T]
        pe = pe_scale * pe
        return torch.cat([x, pe], dim=1)  # [B, V+2F, T]
    elif arch in ("waveformer", "tinymyo"):
        return x.unsqueeze(1)  # [B, 1, V, T]
    elif arch == "otis":
        # x is [B, V, T] -> OTiS expects [B, 1, V, T]
        x4 = x.unsqueeze(1)

        # Optional: pad T to multiple of otis_patch_width to avoid truncation inside patch embed
        pw = 32 # TODO: pass as arg
        T = x4.size(-1)
        pad_t = (pw - (T % pw)) % pw
        if pad_t:
            x4 = F.pad(x4, (0, pad_t))  # pad last dim (time)
        return x4
    # --- NEW: ECG FMs / foundation models ---
    elif arch in ("ecgfounder", "clef", "ecgfm"):
        # Most ECG backbones are Conv1D-style: [B, C, T] where C == #leads
        return x

    elif arch == "hubertecg":
        # Most wav2vec/HubERT-style models expect 1D sequences; pick one lead by default.
        # If you want a different behavior (avg over leads), adjust here.
        return x[:, 0, :]  # [B, T]
    elif arch == "physiowave":
        patch_t = 64  # must match the model's patch_size[1]
        T = x.size(-1)
        pad_t = (patch_t - (T % patch_t)) % patch_t
        if pad_t:
            x = F.pad(x, (0, pad_t))
        return x
    elif arch in ("fcn", "tcn", "bilstm", "convnext1d", "ai85tcn1d"):
        # Baseline conv/RNN models expect [B, V, T]
        return x
    
    raise ValueError(f"Unknown student arch: {arch}")

# ----------------------------
# Schedulers
# ----------------------------
class WarmupCosineSchedule(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = max(1, int(warmup_steps))
        self.total_steps = max(self.warmup_steps + 1, int(total_steps))
        super().__init__(optimizer, self.lr_lambda, last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(self.warmup_steps)
        progress = float(step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

# ----------------------------
# Teacher builder (your existing)
# ----------------------------
@torch.no_grad()
def build_teacher_for_kd(ckpt_path, kd_task_type, num_outputs_for_kd, rank=0):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get('model_state_dict') or ckpt.get('state_dict') or ckpt
    targs = ckpt.get('args', {}) or {}

    in_channels       = int(targs.get('in_channels', 12))
    max_level         = int(targs.get('max_level', 3))
    wave_kernel_size  = int(targs.get('wave_kernel_size', 16))
    wavelet_names     = targs.get('wavelet_names', ['db6'])
    use_sep_ch        = bool(targs.get('use_separate_channel', True))
    patch_size_t      = int(targs.get('patch_size', 40))
    embed_dim         = int(targs.get('embed_dim', 256))
    depth             = int(targs.get('depth', 6))
    num_heads         = int(targs.get('num_heads', 8))
    mlp_ratio         = float(targs.get('mlp_ratio', 4.0))
    dropout           = float(targs.get('dropout', 0.1))
    use_pos_embed     = bool(targs.get('use_pos_embed', True))
    pos_embed_type    = targs.get('pos_embed_type', '2d')
    pooling           = targs.get('pooling', 'mean')

    head_hidden_dim   = targs.get("head_hidden_dim", None)
    head_dropout      = float(targs.get("head_dropout", 0.1))
    pooling           = str(targs.get("pooling", "mean"))
    hidden_factor     = int(targs.get("hidden_factor", 2))

    if head_hidden_dim is None:
        k = "task_heads.classification.head.1.weight"
        if k in sd and sd[k].ndim == 2:
            head_hidden_dim = int(sd[k].shape[0])

    head_config = {
        "hidden_dims": [int(head_hidden_dim)] if head_hidden_dim else None,
        "dropout": head_dropout,
        "pooling": pooling,
        "hidden_factor": 1 if head_hidden_dim else hidden_factor,
    }

    if kd_task_type == 'multilabel':
        teacher = BERTWaveletTransformer(
            in_channels=in_channels,
            max_level=max_level,
            wave_kernel_size=wave_kernel_size,
            wavelet_names=wavelet_names,
            use_separate_channel=use_sep_ch,
            patch_size=(1, patch_size_t),
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            use_pos_embed=use_pos_embed,
            pos_embed_type=pos_embed_type,
            task_type='multilabel',
            num_labels=num_outputs_for_kd,
            head_config=head_config,
            pooling=pooling
        )
        want_prefix = 'task_heads.multilabel.'
        alt_prefix  = 'task_heads.classification.'
    else:
        teacher = BERTWaveletTransformer(
            in_channels=in_channels,
            max_level=max_level,
            wave_kernel_size=wave_kernel_size,
            wavelet_names=wavelet_names,
            use_separate_channel=use_sep_ch,
            patch_size=(1, patch_size_t),
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            use_pos_embed=use_pos_embed,
            pos_embed_type=pos_embed_type,
            task_type='classification',
            num_classes=num_outputs_for_kd,
            head_config=head_config,
            pooling=pooling
        )
        want_prefix = 'task_heads.classification.'
        alt_prefix  = 'task_heads.multilabel.'

    missing, unexpected = teacher.load_state_dict(sd, strict=False)
    has_wanted_head_sd = any(k.startswith(want_prefix) for k in sd)

    if not has_wanted_head_sd:
        if rank == 0:
            print(f">> Teacher checkpoint lacks '{want_prefix[:-1]}' head. Loading backbone only.")
        backbone_sd = {k: v for k, v in sd.items()
                       if not k.startswith(want_prefix) and not k.startswith(alt_prefix)}
        missing, unexpected = teacher.load_state_dict(backbone_sd, strict=False)
        if rank == 0:
            print(f"   Loaded backbone. Missing={len(missing)}, Unexpected={len(unexpected)}")
        return teacher, False

    head_loaded = not any(k.startswith(want_prefix) for k in missing)
    if rank == 0:
        print(f"   Loaded teacher SD. Missing={len(missing)}, Unexpected={len(unexpected)} "
              f"(matching head loaded: {head_loaded})")
    return teacher, head_loaded

# ----------------------------
# KD losses
# ----------------------------
def kd_ce(student_logits, teacher_logits, T=1.0):
    s = F.log_softmax(student_logits / T, dim=1)
    t = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(s, t, reduction='batchmean') * (T * T)

def kd_bce_with_logits(student_logits, teacher_logits, T=1.0):
    if T <= 0:
        T = 1.0
    with torch.no_grad():
        soft_targets = torch.sigmoid(teacher_logits / T)
    return F.binary_cross_entropy_with_logits(student_logits / T, soft_targets, reduction='mean') * (T * T)

# ----------------------------
# Eval
# ----------------------------

@torch.no_grad()
def eval_unified(epoch, rank, model, loader, device, criterion,
                 threshold=0.5, desc_prefix="Val", task_type=None,
                 pe_cache=None, student_arch="physiowavenpu", pe_scale=0.1, waveformer_patch_width=64, amp_enabled=False, amp_dtype=torch.float16):

    model.eval()
    total_loss, total_samples = 0.0, 0
    all_logits, all_labels = [], []

    disp = loader if rank != 0 else tqdm(loader, desc=f"{desc_prefix} Epoch {epoch}", ncols=120)
    for batch in disp:
        x, y = batch[0], batch[1]

        # ✅ MOVE DATA TO GPU (this is what was missing)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if student_arch in ("ecgfm","ecgfounder","clef","hubertecg","fcn", "bilstm", "ai85tcn1d"):
            x = zscore_per_sample_channel(x)

        with autocast_ctx(enabled=(amp_enabled and device.type == "cuda"), dtype=amp_dtype):
            x_in = make_student_input(x, arch=student_arch, pe_cache=pe_cache, pe_scale=pe_scale)

            if student_arch == "waveformer":
                # x_in: [B,1,V,T]
                x_in, pos_embed_y = make_waveformer_pos_embed_y(
                    x_in, patch_height=1, patch_width=waveformer_patch_width, domain_offset=0
                )
                logits = model(x_in, pos_embed_y)
            elif student_arch == "physiowave":
                logits = model(x_in, task='downstream', task_name='classification')
                if isinstance(logits, (tuple, list)): logits = logits[0]
                assert logits.dim() == 2, f"PhysioWave should output [B,C], got {tuple(logits.shape)}"
            else:
                logits = model(x_in)

        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if isinstance(logits, dict):
            logits = logits.get("logits", None) or logits.get("preds", None) or logits.get("y_hat", None) or next(iter(logits.values()))

        if logits.dim() == 3:
            logits = logits.mean(dim=-1)
        elif logits.dim() == 4:
            logits = logits.flatten(2).mean(-1)

        # loss = criterion(logits, y.float() if (task_type == "multilabel") else y)
        logits_f = fp32(logits)
        if task_type == "multilabel":
            loss = criterion(logits_f, y.float())
        else:
            loss = criterion(logits_f, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_samples += bs
        all_logits.append(logits.detach().cpu())
        all_labels.append(y.detach().cpu())

        if student_arch == "waveformer":
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            if isinstance(logits, dict):
                logits = logits.get("logits", None) or logits.get("preds", None) or logits.get("y_hat", None) or next(iter(logits.values()))
            assert logits.dim() == 2, f"WaveFormer should return [B,C] logits; got {tuple(logits.shape)}"

        if rank == 0:
            disp.set_postfix({"loss": f"{total_loss/max(1,total_samples):.4f}"})

    avg_loss = total_loss / max(1, total_samples)
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    if task_type is None:
        if labels.ndim == 1:
            task_type = "classification"
        elif labels.ndim == 2 and (labels.sum(dim=1) == 1).all():
            task_type = "classification"
        else:
            task_type = "multilabel"

    metrics = {"loss": avg_loss}

    if task_type == "classification":
        if labels.ndim == 2:
            y_true = labels.argmax(dim=1).numpy()
        else:
            y_true = labels.numpy()

        probs = torch.softmax(logits, dim=1).numpy()
        y_pred = probs.argmax(axis=1)

        num_classes = probs.shape[1]
        y_true_1h = np.eye(num_classes, dtype=np.float32)[y_true]
        y_pred_1h = np.eye(num_classes, dtype=np.float32)[y_pred]

        acc = accuracy_score(y_true, y_pred)
        prec_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
        prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec_micro  = recall_score(y_true, y_pred, average='micro', zero_division=0)
        rec_macro  = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_micro   = f1_score(y_true, y_pred, average='micro', zero_division=0)
        f1_macro   = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro_no_rest = f1_score(y_true, y_pred, average="macro", labels=list(range(1, 53)), zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        try:
            auroc_micro = roc_auc_score(y_true_1h, probs, average='micro', multi_class='ovr')
            auroc_macro = roc_auc_score(y_true_1h, probs, average='macro', multi_class='ovr')
        except Exception:
            auroc_micro = float('nan'); auroc_macro = float('nan')

        try:
            ap_micro = average_precision_score(y_true_1h, probs, average='micro')
            ap_macro = average_precision_score(y_true_1h, probs, average='macro')
        except Exception:
            ap_micro = float('nan'); ap_macro = float('nan')

        ham = hamming_loss(y_true_1h, y_pred_1h)
        jac_micro = jaccard_score(y_true_1h, y_pred_1h, average='micro', zero_division=0)
        jac_macro = jaccard_score(y_true_1h, y_pred_1h, average='macro', zero_division=0)

        if rank == 0:
            print(f"[{desc_prefix}] Epoch {epoch}: Loss={avg_loss:.4f}")
            print("  -- Single-label view --")
            print(f"  Accuracy:               {acc:.4f}")
            print(f"  Precision(micro/macro): {prec_micro:.4f}/{prec_macro:.4f}")
            print(f"  Recall(micro/macro):    {rec_micro:.4f}/{rec_macro:.4f}")
            print(f"  F1(micro/macro):        {f1_micro:.4f}/{f1_macro:.4f}")
            print(f"  F1(weighted):           {f1_weighted:.4f}")
            print(f"  F1(no rest, macro):     {f1_macro_no_rest:.4f}")
            print(f"  AUROC(micro/macro):     {auroc_micro:.4f}/{auroc_macro:.4f}")
            print(f"  AP(micro/macro):        {ap_micro:.4f}/{ap_macro:.4f}")
            print("  -- Multilabel-style on one-hot --")
            print(f"  Hamming Loss:           {ham:.4f}")
            print(f"  Jaccard(micro/macro):   {jac_micro:.4f}/{jac_macro:.4f}")

        metrics.update(dict(
            accuracy=acc,
            precision_micro=prec_micro, precision_macro=prec_macro,
            recall_micro=rec_micro,     recall_macro=rec_macro,
            f1_micro=f1_micro,          f1_macro=f1_macro,
            f1_weighted=f1_weighted,
            auroc_micro=auroc_micro,    auroc_macro=auroc_macro,
            ap_micro=ap_micro,          ap_macro=ap_macro,
            hamming_loss=ham,
            jaccard_micro=jac_micro,    jaccard_macro=jac_macro
        ))
        return metrics

    # multilabel
    y_true = labels.numpy().astype(np.float32)
    probs = torch.sigmoid(logits).numpy()

    y_pred = (probs >= threshold).astype(np.float32)

    y_true_arg = y_true.argmax(axis=1)
    y_pred_arg = probs.argmax(axis=1)

    acc = accuracy_score(y_true_arg, y_pred_arg)
    prec_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_micro  = recall_score(y_true, y_pred, average='micro', zero_division=0)
    rec_macro  = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_micro   = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_macro   = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    try:
        auroc_micro = roc_auc_score(y_true, probs, average='micro')
        auroc_macro = roc_auc_score(y_true, probs, average='macro')
    except Exception:
        auroc_micro = float('nan'); auroc_macro = float('nan')

    try:
        ap_micro = average_precision_score(y_true, probs, average='micro')
        ap_macro = average_precision_score(y_true, probs, average='macro')
    except Exception:
        ap_micro = float('nan'); ap_macro = float('nan')

    ham = hamming_loss(y_true, y_pred)
    jac_micro = jaccard_score(y_true, y_pred, average='micro', zero_division=0)
    jac_macro = jaccard_score(y_true, y_pred, average='macro', zero_division=0)

    if rank == 0:
        print(f"[{desc_prefix}] Epoch {epoch}: Loss={avg_loss:.4f}")
        print("  -- True multilabel view (thresholded) --")
        print(f"  Acc:                    {acc:.4f}")
        print(f"  Precision(micro/macro): {prec_micro:.4f}/{prec_macro:.4f}")
        print(f"  Recall(micro/macro):    {rec_micro:.4f}/{rec_macro:.4f}")
        print(f"  F1(micro/macro):        {f1_micro:.4f}/{f1_macro:.4f}")
        print(f"  F1(weighted):           {f1_weighted:.4f}")
        print(f"  AUROC(micro/macro):     {auroc_micro:.4f}/{auroc_macro:.4f}")
        print(f"  AP(micro/macro):        {ap_micro:.4f}/{ap_macro:.4f}")
        print(f"  Hamming Loss:           {ham:.4f}")
        print(f"  Jaccard(micro/macro):   {jac_micro:.4f}/{jac_macro:.4f}")

    metrics.update(dict(
        accuracy=acc,
        precision_micro=prec_micro, precision_macro=prec_macro,
        recall_micro=rec_micro,     recall_macro=rec_macro,
        f1_micro=f1_micro,          f1_macro=f1_macro,
        f1_weighed=f1_weighted,
        auroc_micro=auroc_micro,    auroc_macro=auroc_macro,
        ap_micro=ap_micro,          ap_macro=ap_macro,
        hamming_loss=ham,
        jaccard_micro=jac_micro,    jaccard_macro=jac_macro
    ))
    return metrics

# ----------------------------
# OTiS: build + load pretrained (based on their main_finetune)
# ----------------------------
def build_otis_student(args, num_classes: int, device: torch.device, rank: int):
    if not args.otis_repo_root:
        raise ValueError("--otis_repo_root is required for --student_arch otis")

    # Load OTiS modules
    models_vit = load_repo_module(args.otis_repo_root, "models_vit.py", "otis_models_vit")
    pos_embed = load_repo_module(args.otis_repo_root, "util/pos_embed.py", "otis_pos_embed")

    # Domains: emulate a single-domain dataset like OTiS expects
    domain = args.domain_name
    domains = {domain: (args.otis_input_channels, args.otis_input_variates, args.otis_time_steps)}
    img_size = (args.otis_input_channels, args.otis_input_variates, args.otis_time_steps)
    patch_size = (args.otis_patch_height, args.otis_patch_width)

    if args.otis_model not in models_vit.__dict__:
        raise AttributeError(f"OTiS models_vit has no model '{args.otis_model}'")

    model = models_vit.__dict__[args.otis_model](
        domains=domains,
        img_size=img_size,
        patch_size=patch_size,
        num_classes=num_classes,
        drop_path_rate=args.otis_drop_path,
        global_pool=args.otis_global_pool,
        attention_pool=args.otis_attention_pool,
        masking_blockwise=False,
        mask_ratio=0.0,
        mask_c_ratio=0.0,
        mask_t_ratio=0.0,
    ).to(device)

    if not args.otis_finetune:
        if rank == 0:
            print(">> OTiS: no --otis_finetune provided; training from scratch.")
        return model

    ckpt = torch.load(args.otis_finetune, map_location="cpu", weights_only=False)
    checkpoint_model = ckpt.get("model", ckpt)

    # Strip head if mismatch
    state_dict = model.state_dict()
    for k in ["head.weight", "head.bias"]:
        if k in checkpoint_model and k in state_dict and checkpoint_model[k].shape != state_dict[k].shape:
            if rank == 0:
                print(f">> OTiS: removing mismatched key {k} from checkpoint")
            del checkpoint_model[k]

    # Patch_embed mismatch => drop patch_embed weights (like their code)
    new_patch_size = False
    try:
        nb_channels_ckpt = checkpoint_model["patch_embed.proj.weight"].shape[-3]
        nb_channels_model = img_size[0]
        checkpoint_patch_size = checkpoint_model["patch_embed.proj.weight"].shape[-2:]
        ph_ckpt, pw_ckpt = int(checkpoint_patch_size[0]), int(checkpoint_patch_size[1])
        ph_model, pw_model = int(patch_size[0]), int(patch_size[1])

        if nb_channels_ckpt != nb_channels_model or ph_ckpt != ph_model or pw_ckpt != pw_model:
            new_patch_size = True
            for key in [
                "patch_embed.proj.weight", "patch_embed.proj.bias",
                "patch_embed.norm.weight", "patch_embed.norm.bias"
            ]:
                if key in checkpoint_model:
                    if rank == 0:
                        print(f">> OTiS: removing key {key} (new patch_embed required)")
                    del checkpoint_model[key]
    except Exception as e:
        if rank == 0:
            print(f">> OTiS: patch_embed check skipped ({e})")

    # interpolate pos_embed_x
    try:
        pos_embed.interpolate_pos_embed_x(model, checkpoint_model)
        if "pos_embed_x" in checkpoint_model:
            del checkpoint_model["pos_embed_x"]
    except Exception as e:
        if rank == 0:
            print(f">> OTiS: interpolate_pos_embed_x failed/skipped: {e}")

    # pos_embed_y handling:
    # If checkpoint includes pos_embed_y.weight, load it if shapes permit.
    if "pos_embed_y.weight" in checkpoint_model:
        try:
            if rank == 0:
                print(f">> OTiS: loading pos_embed_y from checkpoint: {checkpoint_model['pos_embed_y.weight'].shape}")
            model.pos_embed_y = None
            model.pos_embed_y = torch.nn.Embedding.from_pretrained(checkpoint_model["pos_embed_y.weight"])
            del checkpoint_model["pos_embed_y.weight"]
        except Exception as e:
            if rank == 0:
                print(f">> OTiS: pos_embed_y load failed; using fresh init. Reason: {e}")
            # keep model's initialized pos_embed_y
            del checkpoint_model["pos_embed_y.weight"]

    msg = model.load_state_dict(checkpoint_model, strict=False)
    if rank == 0:
        print(f">> OTiS: loaded checkpoint. missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
        if new_patch_size:
            print(">> OTiS: patch_embed was re-initialized due to patch/channel mismatch")

    # IMPORTANT: some params (e.g., pos_embed_y from_pretrained) are created on CPU
    # Ensure *all* params/buffers are on the target GPU before wrapping with DDP.
    model = model.to(device)

    return model

class OTiSWrapper(nn.Module):
    """
    OTiS forward expects: model(x, pos_embed_y)
      - x: [B, 1, V, T]
      - pos_embed_y: LongTensor indices shaped [B, grid_h, grid_w]
    """
    def __init__(self, otis_model: nn.Module, *, patch_height: int, patch_width: int, domain_offset: int = 0):
        super().__init__()
        self.model = otis_model
        self.patch_height = int(patch_height)
        self.patch_width = int(patch_width)
        self.domain_offset = int(domain_offset)

    def forward(self, x):
        # x: [B, 1, V, T]
        B, C, V, T = x.shape

        ph, pw = self.patch_height, self.patch_width
        grid_h = V // ph
        grid_w = T // pw

        y = torch.arange(grid_h, device=x.device, dtype=torch.long) + self.domain_offset  # [grid_h]
        pos_embed_y = y.view(1, grid_h, 1).expand(B, grid_h, grid_w)                      # [B, grid_h, grid_w]

        return self.model(x, pos_embed_y)

# ----------------------------
# TinyMyo: instantiate cfg.task via Hydra and load pretrained
# ----------------------------
def build_tinymyo_student(args, device: torch.device, rank: int):
    if not args.tinymyo_repo_root:
        raise ValueError("--tinymyo_repo_root is required for --student_arch tinymyo")
    if not args.tinymyo_config_dir or not args.tinymyo_config_name:
        raise ValueError("--tinymyo_config_dir and --tinymyo_config_name are required for --student_arch tinymyo")

    # Import Hydra & OmegaConf from user's environment
    import hydra
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    # Make repo importable (task classes typically live there)
    repo_root = os.path.abspath(args.tinymyo_repo_root)
    sys.path.insert(0, repo_root)

    # Compose config
    with initialize_config_dir(config_dir=os.path.abspath(args.tinymyo_config_dir), version_base="1.1"):
        overrides = []
        # You can inject any overrides you want here via --tinymyo_overrides
        if args.tinymyo_overrides:
            overrides.extend(args.tinymyo_overrides)
        cfg = compose(config_name=args.tinymyo_config_name, overrides=overrides)

    if rank == 0:
        print(">> TinyMyo cfg (resolved):")
        print(OmegaConf.to_yaml(cfg, resolve=True))

    # Instantiate LightningModule (task)
    model = hydra.utils.instantiate(cfg.task, cfg)

    # Load pretrained checkpoint (matching their run_train)
    if args.tinymyo_pretrained_safetensors:
        if hasattr(model, "load_safetensors_checkpoint"):
            if rank == 0:
                print(f">> TinyMyo: loading safetensors from {args.tinymyo_pretrained_safetensors}")
            model.load_safetensors_checkpoint(args.tinymyo_pretrained_safetensors)
        else:
            raise AttributeError("TinyMyo task has no load_safetensors_checkpoint()")
    elif args.tinymyo_pretrained_ckpt:
        if hasattr(model, "load_pretrained_checkpoint"):
            if rank == 0:
                print(f">> TinyMyo: loading checkpoint from {args.tinymyo_pretrained_ckpt}")
            model.load_pretrained_checkpoint(args.tinymyo_pretrained_ckpt)
        else:
            raise AttributeError("TinyMyo task has no load_pretrained_checkpoint()")
    else:
        if rank == 0:
            print(">> TinyMyo: no pretrained path given; training from scratch.")

    # We want a pure nn.Module forward returning logits; LightningModule is nn.Module already.
    model.to(device)
    return model

# ----------------------------
# WaveFormer: build (as before)
# ----------------------------
def build_waveformer_student(args, num_classes: int, device: torch.device):
    if not args.waveformer_repo_root:
        raise ValueError("--waveformer_repo_root is required for --student_arch waveformer")

    wf = load_repo_module(args.waveformer_repo_root, "model.py", "waveformer_ext_model")
    if not hasattr(wf, args.waveformer_model):
        raise AttributeError(f"WaveFormer model.py has no attribute '{args.waveformer_model}'")

    wf_args = type("WFArgs", (), {"nb_classes": num_classes})()

    student = getattr(wf, args.waveformer_model)(
        downstream="classification",
        args=wf_args,
        input_channels=1,
        time_steps=args.max_length,
        patch_size=(1, args.waveformer_patch_width),
        domains={args.domain_name: (1, args.in_channels, args.max_length)},
        domain_weights={args.domain_name: 1.0},
        domain_agnostic=True,
    ).to(device)

    return student

################################################################################################################
# NEW

def ecgfounder_preprocess_np(
    x_ct: np.ndarray,
    *,
    fs: float,
    notch_freq: float = 50.0,
    notch_q: float = 30.0,
    band_low: float = 0.67,
    band_high: float = 40.0,
    band_order: int = 4,
    baseline_kernel_sec: float = 0.4,
    do_zscore: bool = True,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    x_ct: numpy array [C, T]
    Returns: numpy array [C, T]
    """
    if not _HAS_SCIPY:
        raise RuntimeError("ECGFounder preprocessing requested but scipy is not installed.")

    x = np.asarray(x_ct, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # 1) Notch
    b, a = iirnotch(notch_freq, notch_q, fs)
    y = np.zeros_like(x)
    for c in range(x.shape[0]):
        y[c] = filtfilt(b, a, x[c])

    # 2) Bandpass
    b, a = butter(N=band_order, Wn=[band_low, band_high], btype="bandpass", fs=fs)
    for c in range(y.shape[0]):
        y[c] = filtfilt(b, a, y[c])

    # 3) Baseline wander removal via median filter
    k = int(baseline_kernel_sec * fs) + 1
    if k % 2 == 0:
        k += 1
    baseline = np.zeros_like(y)
    for c in range(y.shape[0]):
        baseline[c] = medfilt(y[c], kernel_size=k)
    y = y - baseline

    # 4) Z-score (global per-sample, like their snippet)
    if do_zscore:
        mu = y.mean()
        sig = y.std()
        y = (y - mu) / (sig + eps)

    return y

class PreprocessWrapperDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds, *, mode: str, fs: float, args):
        self.base_ds = base_ds
        self.mode = mode
        self.fs = float(fs)
        self.args = args

        # Preserve common dataset attributes used elsewhere
        if hasattr(base_ds, "num_classes"):
            self.num_classes = base_ds.num_classes
        if hasattr(base_ds, "num_labels"):
            self.num_labels = base_ds.num_labels

    def __len__(self):
        return len(self.base_ds)

    def __getattr__(self, name):
        """
        Delegate unknown attributes to the wrapped dataset.
        Important for things like num_classes, class_names, etc.
        """
        if name in ("base_ds", "mode", "fs", "args"):
            raise AttributeError(name)
        return getattr(self.base_ds, name)

    def __getitem__(self, idx):
        x, y = self.base_ds[idx]  # x: torch [V,T] on CPU
        if self.mode == "none":
            return x, y

        x_np = x.detach().cpu().numpy()  # [V,T]
        x_np = ecgfounder_preprocess_np(
            x_np,
            fs=self.fs,
            notch_freq=self.args.pp_notch_freq,
            notch_q=self.args.pp_notch_q,
            band_low=self.args.pp_band_low,
            band_high=self.args.pp_band_high,
            band_order=self.args.pp_band_order,
            baseline_kernel_sec=self.args.pp_baseline_kernel_sec,
            do_zscore=self.args.pp_zscore,
        )
        x = torch.from_numpy(x_np).to(dtype=torch.float32)
        return x, y

################################################################################################################
# NEW

# ----------------------------
# Generic "frozen FM backbone + linear head" wrapper
# ----------------------------
class FrozenBackboneWithHead(nn.Module):
    """
    Wrap an arbitrary backbone that returns either:
      - logits [B,C]
      - embeddings [B,D]
      - dict/tuple with hidden states / features
    and attach a trainable linear head.
    """
    def __init__(self, backbone: nn.Module, num_classes: int, *, rank: int = 0):
        super().__init__()
        self.backbone = backbone
        self.num_classes = int(num_classes)
        self.head = None  # lazily created after we see D
        self._feat_dim = None
        if rank == 0:
            print(f">> FM wrapper: backbone={type(backbone)} num_classes={self.num_classes}")

    @staticmethod
    def _to_feat_tensor(out: Any) -> torch.Tensor:
        # common dict keys / attrs
        if isinstance(out, dict):
            for k in ("last_hidden_state", "encoder_out", "extract_features", "features", "embeddings", "embedding", "hidden_states", "repr", "representations"):
                if k in out and isinstance(out[k], torch.Tensor):
                    return out[k]
        for attr in ("features", "feats", "embeddings", "embedding", "last_hidden_state", "hidden_states"):
            if hasattr(out, attr):
                v = getattr(out, attr)
                if isinstance(v, torch.Tensor):
                    return v
        # tuple/list
        if isinstance(out, (tuple, list)):
            for v in out:
                if isinstance(v, torch.Tensor):
                    return v
                if isinstance(v, dict):
                    try:
                        return FrozenBackboneWithHead._to_feat_tensor(v)
                    except Exception:
                        pass
        # already tensor
        if isinstance(out, torch.Tensor):
            return out
        raise RuntimeError(f"Could not extract tensor features from output type={type(out)}")

    @staticmethod
    def _pool_to_bxd(t: torch.Tensor) -> torch.Tensor:
        # normalize any shape to [B, D]
        if t.dim() == 2:
            return t

        if t.dim() == 3:
            # Common cases:
            #  - HF transformers: [B, T, D]  -> mean over T
            #  - CNN-ish features: [B, D, T] -> mean over T
            B, A, C = t.shape
            # Heuristic: feature dim is usually larger than token length
            # If last dim is "big", assume [B, T, D]
            if C >= A:
                return t.mean(dim=1)  # [B, D]
            else:
                return t.mean(dim=2)  # [B, D]

        if t.dim() == 4:
            B = t.size(0)
            t = t.view(B, -1, t.size(-1))
            return t.mean(dim=1)

        B = t.size(0)
        t = t.view(B, -1)
        return t

    def _ensure_head(self, feat_bxd: torch.Tensor):
        if self.head is not None:
            return
        if feat_bxd.dim() != 2:
            raise RuntimeError(f"Expected pooled feat [B,D], got {tuple(feat_bxd.shape)}")
        D = int(feat_bxd.size(1))
        self._feat_dim = D
        self.head = nn.Linear(D, self.num_classes).to(feat_bxd.device)

    def forward(self, x):
        # backbone frozen by default
        with torch.no_grad():
            out = self.backbone(x)
            feat = self._to_feat_tensor(out)
            feat = self._pool_to_bxd(feat)

        self._ensure_head(feat)
        return self.head(feat)


# ----------------------------
# ECGFounder
# ----------------------------
def build_ecgfounder_student(args, num_classes: int, device: torch.device, rank: int):
    if not args.ecgfounder_repo_root:
        raise ValueError("--ecgfounder_repo_root is required for --student_arch ecgfounder")
    if not args.ecgfounder_ckpt:
        raise ValueError("--ecgfounder_ckpt is required for --student_arch ecgfounder (path to .pth)")

    ft = load_repo_module(args.ecgfounder_repo_root, "finetune_model.py", "ecgfounder_finetune")

    lead_cfg = str(args.ecgfounder_lead_config)
    if lead_cfg == "12lead":
        model = ft.ft_12lead_ECGFounder(device, args.ecgfounder_ckpt, num_classes, linear_prob=args.ecgfounder_linear_probe)
    elif lead_cfg == "1lead":
        model = ft.ft_1lead_ECGFounder(device, args.ecgfounder_ckpt, num_classes, linear_prob=args.ecgfounder_linear_probe)
    else:
        raise ValueError("--ecgfounder_lead_config must be 1lead or 12lead")

    model = model.to(device)
    return model


# ----------------------------
# CLEF (Nokia-Bell-Labs/ecg-foundation-model)
# ----------------------------
def build_clef_student(args, num_classes: int, device: torch.device, rank: int):
    if not args.clef_repo_root:
        raise ValueError("--clef_repo_root is required for --student_arch clef")
    if not args.clef_ckpt:
        raise ValueError("--clef_ckpt is required for --student_arch clef (path to .ckpt/.pth from Zenodo)")

    clef_mod = load_repo_module(args.clef_repo_root, "clef/baselines/models/CLEF.py", "clef_baseline_clef")

    # Their helper returns a backbone with dense=Identity (i.e., embeddings) and optional weight loading.
    backbone = clef_mod.create_net1d_by_size(
        device=device,
        model_size=str(args.clef_model_size),
        n_classes=None,               # head is Identity anyway
        linear_prob=False,            # we handle freezing ourselves
        # pth=str(args.clef_ckpt),
        in_channels=int(args.clef_in_channels),
    ).to(device)

    # NEW - for 12 lead
    sd = torch.load(args.clef_ckpt, map_location="cpu")
    state = sd["state_dict"] if "state_dict" in sd else sd
    k = "first_conv.conv.weight"
    if k in state and state[k].shape[1] == 1 and args.clef_in_channels == 12:
        w = state[k]  # [32,1,16]
        state[k] = w.repeat(1, 12, 1) / 12.0   # average-energy preserving
        # (or repeat without /12 if you want larger initial magnitude)
    msg = backbone.load_state_dict(state, strict=False)
    print(msg)

    # freeze backbone; train linear head
    for p in backbone.parameters():
        p.requires_grad = True

    student = FrozenBackboneWithHead(backbone, num_classes, rank=rank).to(device)
    # make head trainable
    for p in student.parameters():
        p.requires_grad = True
    # head created lazily; we’ll unfreeze after a warmup forward in main_worker
    return student


# ----------------------------
# HuBERT-ECG (Hugging Face Transformers)
# ----------------------------
class HuBERTECGBackbone(nn.Module):
    def __init__(self, model_id: str, device: torch.device, *, rank: int = 0,
                 trust_remote_code: bool = False, revision: str | None = None):
        super().__init__()
        try:
            from transformers import AutoModel
        except Exception as e:
            raise RuntimeError("student_arch=hubertecg requires `transformers` installed.") from e

        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=bool(trust_remote_code),
            revision=revision,
        )
        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        if rank == 0:
            print(f">> HuBERT-ECG backbone loaded: {model_id} (trust_remote_code={trust_remote_code}, revision={revision})")

    def forward(self, x_bt: torch.Tensor):
        return self.model(input_values=x_bt)



def build_hubertecg_student(args, num_classes: int, device: torch.device, rank: int):
    backbone = HuBERTECGBackbone(
        args.hubertecg_model_id,
        device,
        rank=rank,
        trust_remote_code=args.hubertecg_trust_remote_code,
        revision=(args.hubertecg_revision or None),
    )
    student = FrozenBackboneWithHead(backbone, num_classes, rank=rank).to(device)
    for p in student.parameters():
        p.requires_grad = False
    return student


# ----------------------------
# ECG-FM (bowang-lab/ecg-fm; fairseq-signals)
# ----------------------------
class ECGFMBackbone(nn.Module):
    def __init__(self, checkpoint_path: str, device: torch.device, *, rank: int = 0):
        super().__init__()
        self.device = device

        build_fn = None

        # Prefer fairseq-signals if present
        try:
            from fairseq_signals.models import build_model_from_checkpoint as build_fn  # type: ignore
        except Exception:
            pass

        # Fallback: fairseq
        if build_fn is None:
            try:
                from fairseq.models import build_model_from_checkpoint as build_fn  # type: ignore
            except Exception:
                build_fn = None

        self.model = None
        if build_fn is not None:
            self.model = build_fn(checkpoint_path)
        else:
            # Final fallback: checkpoint_utils style
            try:
                from fairseq_signals import checkpoint_utils as cu  # type: ignore
            except Exception:
                try:
                    from fairseq import checkpoint_utils as cu  # type: ignore
                except Exception as e:
                    raise RuntimeError(
                        "ECG-FM requires fairseq-signals or fairseq installed. "
                        "Could not import build_model_from_checkpoint or checkpoint_utils."
                    ) from e

            models, _args, _task = cu.load_model_ensemble_and_task([checkpoint_path])
            self.model = models[0]

        assert self.model is not None
        self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

        if rank == 0:
            print(f">> ECG-FM backbone loaded from checkpoint: {checkpoint_path}")

    def forward(self, x_bct: torch.Tensor):
        # fairseq models sometimes accept source=... or x=...
        # We'll try a couple of common calling conventions.
        try:
            return self.model(x_bct)
        except Exception:
            try:
                return self.model(source=x_bct)
            except Exception:
                return self.model(x=x_bct)


def build_ecgfm_student(args, num_classes: int, device: torch.device, rank: int):
    if not args.ecgfm_ckpt:
        raise ValueError("--ecgfm_ckpt is required for --student_arch ecgfm (path to .pt)")

    backbone = ECGFMBackbone(args.ecgfm_ckpt, device, rank=rank)
    student = FrozenBackboneWithHead(backbone, num_classes, rank=rank).to(device)
    for p in student.parameters():
        p.requires_grad = False
    return student

################################################################################################################

# ----------------------------
# Student builder dispatch
# ----------------------------
def build_student(args, train_ds, device, rank):
    pe_cache = None

    if args.student_arch == "physiowavenpu":
        depth = int(args.student_depth)  # allow 0,1,3
        embed = int(args.student_embed_dim)

        kernel_set = parse_kernel_set(args.student_kernel_set)

        # If PosEnc disabled, pe_cache is None and C_total == in_channels.
        pos_freqs = int(args.student_pos_freqs)
        C_total = int(args.in_channels) + 2 * pos_freqs

        student = PhysioWaveNPU(
            input_length=args.max_length,
            in_channels=C_total,
            num_classes=train_ds.num_classes,
            bank_ch=args.student_bank_ch,
            kernel_set=kernel_set,          # <-- NEW (was fixed (3,5,7))
            patch_t=args.patch_size,
            embed_dim=embed,
            depth=depth,                    # <-- allow 0
            reduce=args.student_reduce,
            conv1d_k=3,
            head_residual=False,
            post_patch_pool_t=5
        ).to(device)

        pe_cache = make_posenc_1d_concat(pos_freqs, args.max_length, device=device)
        return student, pe_cache

    if args.student_arch == "waveformer":
        return build_waveformer_student(args, train_ds.num_classes, device), None

    if args.student_arch == "otis":
        # Ensure otis params are consistent with our data:
        # input_channels=1, input_variates=args.in_channels, time_steps=args.max_length
        args.otis_input_channels = 1
        args.otis_input_variates = args.in_channels
        args.otis_time_steps = args.max_length
        return build_otis_student(args, train_ds.num_classes, device, rank), None

    if args.student_arch == "tinymyo":
        return build_tinymyo_student(args, device, rank), None

    # --- NEW ---
    if args.student_arch == "ecgfounder":
        return build_ecgfounder_student(args, train_ds.num_classes, device, rank), None

    if args.student_arch == "clef":
        return build_clef_student(args, train_ds.num_classes, device, rank), None

    if args.student_arch == "hubertecg":
        return build_hubertecg_student(args, train_ds.num_classes, device, rank), None

    if args.student_arch == "ecgfm":
        return build_ecgfm_student(args, train_ds.num_classes, device, rank), None

    # --- NEW baselines ---
    if args.student_arch == "fcn":
        m = FCN1D(
            in_ch=args.in_channels,
            num_classes=train_ds.num_classes,
            width=int(args.baseline_width),
            dropout=float(args.baseline_dropout),
        ).to(device)
        return m, None

    if args.student_arch == "tcn":
        m = TCN1D(
            in_ch=args.in_channels,
            num_classes=train_ds.num_classes,
            width=int(args.baseline_width),
            depth=int(args.baseline_depth),
            kernel=int(args.tcn_kernel),
            dropout=float(args.baseline_dropout),
            dilated=bool(args.tcn_dilated),
        ).to(device)
        return m, None

    if args.student_arch == "bilstm":
        m = BiLSTM1D(
            in_ch=args.in_channels,
            num_classes=train_ds.num_classes,
            hidden=int(args.baseline_width),
            layers=int(args.bilstm_layers),
            dropout=float(args.baseline_dropout),
        ).to(device)
        return m, None

    if args.student_arch == "convnext1d":
        m = ConvNeXt1D(
            in_ch=args.in_channels,
            num_classes=train_ds.num_classes,
            width=int(args.baseline_width),
            depth=int(args.baseline_depth),
            dropout=float(args.baseline_dropout),
        ).to(device)
        return m, None

    if args.student_arch == "ai85tcn1d":
        m = AI85ECGTCN1D(
            in_ch=args.in_channels,
            num_classes=train_ds.num_classes,
            width=int(args.baseline_width),        # reuse your knobs
            stem_depth=max(1, int(args.baseline_depth // 2)),
            tcn_depth=max(2, int(args.baseline_depth)),
            kernel=int(args.tcn_kernel),
            dropout=float(args.baseline_dropout),
            downsample_every=2,                    # tweakable; keeps compute sane
            dilated=bool(args.tcn_dilated),
            max_dilation=64,
        ).to(device)
        return m, None

    raise ValueError(f"Unknown student_arch: {args.student_arch}")

# ----------------------------
# AI8X/MAX78000-friendly 1D CNN + TCN baseline
# ----------------------------
class _ResDilatedTCNBlock(nn.Module):
    """
    Residual dilated Conv1d block (non-causal), "same" padded.
    Two conv layers + BN + ReLU with dropout.
    """
    def __init__(self, ch: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        assert kernel % 2 == 1, "Use odd kernel for symmetric 'same' padding."
        pad = (kernel // 2) * dilation

        self.conv1 = nn.Conv1d(ch, ch, kernel_size=kernel, padding=pad, dilation=dilation, bias=False)
        self.bn1   = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, kernel_size=kernel, padding=pad, dilation=dilation, bias=False)
        self.bn2   = nn.BatchNorm1d(ch)

        self.act  = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        r = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.drop(x)
        x = self.bn2(self.conv2(x))
        x = self.drop(x)
        return self.act(x + r)


class AI85ECGTCN1D(nn.Module):
    """
    Pure-torch baseline: 1D Conv stem + dilated residual TCN + GAP + Linear.

    Input:  [B, V, T]
    Output: [B, C]

    MAX78000-friendly design choices (conceptually):
      - Conv1d + BN + ReLU throughout
      - Optional strided downsampling in the stem to reduce temporal length early
      - Dilated convs in the TCN head for long receptive field without heavy depth
      - Global average pooling for fixed-size head

    This uses only torch.nn modules (no ai8x wrappers).
    """
    def __init__(
        self,
        in_ch: int,
        num_classes: int,
        width: int = 64,
        stem_depth: int = 4,          # number of conv blocks in stem after 1x1 lift
        tcn_depth: int = 6,           # number of residual dilated blocks
        kernel: int = 3,
        dropout: float = 0.1,
        downsample_every: int = 2,    # stride-2 every N stem blocks (0 disables)
        dilated: bool = True,         # if False: dilation=1 everywhere
        max_dilation: int | None = 64 # cap dilation growth (keeps things sane)
    ):
        super().__init__()
        assert kernel % 2 == 1, "Use odd kernel for symmetric padding."
        self.num_classes = int(num_classes)

        # --- Stem ---
        stem = []
        # 1x1 lift to width
        stem += [
            nn.Conv1d(in_ch, width, kernel_size=1, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
        ]

        for i in range(stem_depth):
            stride = 1
            if downsample_every and downsample_every > 0 and ((i + 1) % downsample_every == 0):
                stride = 2

            pad = kernel // 2
            stem += [
                nn.Conv1d(width, width, kernel_size=kernel, padding=pad, stride=stride, bias=False),
                nn.BatchNorm1d(width),
                nn.ReLU(inplace=True),
            ]

        self.stem = nn.Sequential(*stem)

        # --- TCN head ---
        blocks = []
        for i in range(tcn_depth):
            d = (2 ** i) if dilated else 1
            if max_dilation is not None:
                d = min(d, int(max_dilation))
            blocks.append(_ResDilatedTCNBlock(width, kernel=kernel, dilation=d, dropout=dropout))
        self.tcn = nn.Sequential(*blocks)

        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(width, num_classes)

    def forward(self, x):
        # x: [B,V,T]
        x = self.stem(x)    # [B,width,T']
        x = self.tcn(x)     # [B,width,T']
        x = x.mean(dim=-1)  # GAP -> [B,width]
        x = self.drop(x)
        return self.head(x) # [B,C]

class FCN1D(nn.Module):
    """
    Classic TimeSeries FCN: 3x Conv1D + BN + ReLU, then GAP + Linear.
    Input:  [B, V, T]
    Output: [B, C]
    """
    def __init__(self, in_ch: int, num_classes: int, width: int = 128, dropout: float = 0.1):
        super().__init__()
        w1, w2, w3 = width, width * 2, width * 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, w1, kernel_size=8, padding=4, bias=False),
            nn.BatchNorm1d(w1),
            nn.ReLU(inplace=True),

            nn.Conv1d(w1, w2, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(w2),
            nn.ReLU(inplace=True),

            nn.Conv1d(w2, w3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(w3),
            nn.ReLU(inplace=True),
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(w3, num_classes)

    def forward(self, x):
        x = self.net(x)                 # [B, w3, T]
        x = x.mean(dim=-1)              # GAP -> [B, w3]
        x = self.drop(x)
        return self.head(x)             # [B, C]


class _TCNBlock(nn.Module):
    """
    Residual dilated conv block.
    """
    def __init__(self, ch: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel - 1) * dilation // 2  # non-causal "same" padding
        self.conv1 = nn.Conv1d(ch, ch, kernel_size=kernel, padding=pad, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, kernel_size=kernel, padding=pad, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(ch)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        r = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.drop(x)
        x = self.bn2(self.conv2(x))
        x = self.drop(x)
        return self.act(x + r)


class TCN1D(nn.Module):
    """
    Temporal Convolutional Network (non-causal) with residual dilated blocks.
    Input:  [B, V, T]
    Output: [B, C]
    """
    def __init__(
        self,
        in_ch: int,
        num_classes: int,
        width: int = 128,
        depth: int = 6,
        kernel: int = 3,
        dropout: float = 0.1,
        dilated: bool = True,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, width, kernel_size=1, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
        )
        blocks = []
        for i in range(depth):
            d = (2 ** i) if dilated else 1
            blocks.append(_TCNBlock(width, kernel=kernel, dilation=d, dropout=dropout))
        self.blocks = nn.Sequential(*blocks)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(width, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = x.mean(dim=-1)   # GAP
        x = self.drop(x)
        return self.head(x)


class BiLSTM1D(nn.Module):
    """
    BiLSTM classifier. We treat time as sequence length.
    Input:  [B, V, T]
    Intern: [B, T, V]
    Output: [B, C]
    """
    def __init__(
        self,
        in_ch: int,
        num_classes: int,
        hidden: int = 128,
        layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_ch,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=(dropout if layers > 1 else 0.0),
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)     # [B, T, V]
        out, _ = self.lstm(x)     # [B, T, 2H]
        feat = out.mean(dim=1)    # mean-pool over time
        feat = self.drop(feat)
        return self.head(feat)


class _ConvNeXt1DBlock(nn.Module):
    """
    ConvNeXt-style 1D block: depthwise conv + pointwise MLP.
    Uses GroupNorm(1, C) which behaves like LayerNorm over channels.
    """
    def __init__(self, ch: int, kernel: int = 7, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        pad = kernel // 2
        self.dw = nn.Conv1d(ch, ch, kernel_size=kernel, padding=pad, groups=ch, bias=False)
        self.norm = nn.GroupNorm(1, ch)  # LN-ish
        self.pw1 = nn.Conv1d(ch, ch * mlp_ratio, kernel_size=1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv1d(ch * mlp_ratio, ch, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        r = x
        x = self.dw(x)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.pw2(x)
        x = self.drop(x)
        return x + r


class ConvNeXt1D(nn.Module):
    """
    Simple ConvNeXt1D stack + GAP + Linear.
    Input:  [B, V, T]
    Output: [B, C]
    """
    def __init__(self, in_ch: int, num_classes: int, width: int = 128, depth: int = 6, dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(width),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[
            _ConvNeXt1DBlock(width, kernel=7, mlp_ratio=4, dropout=dropout)
            for _ in range(depth)
        ])
        self.head_norm = nn.LayerNorm(width)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(width, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = x.mean(dim=-1)          # [B, width]
        x = self.head_norm(x)
        x = self.drop(x)
        return self.head(x)

def make_waveformer_pos_embed_y(
    x4: torch.Tensor,
    *,
    patch_height: int = 1,
    patch_width: int = 64,
    domain_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build WaveFormer-style pos_embed_y from the current tensor shape.

    Inputs:
      x4: [B, 1, V, T]
    Returns:
      x4_pad: [B, 1, V_pad, T_pad]  (we only pad T to match patch_width; V assumed divisible by patch_height)
      pos_embed_y: LongTensor [B, grid_h, grid_w]
    """
    assert x4.dim() == 4 and x4.size(1) == 1, f"Expected [B,1,V,T], got {tuple(x4.shape)}"

    B, C, V, T = x4.shape
    ph, pw = int(patch_height), int(patch_width)

    # --- pad T to multiple of pw (WaveFormer patching along time) ---
    pad_t = (pw - (T % pw)) % pw
    if pad_t:
        x4 = F.pad(x4, (0, pad_t))  # pad last dim (time)
        T = T + pad_t

    # --- (optional) pad V to multiple of ph if you ever use ph>1 ---
    pad_v = (ph - (V % ph)) % ph
    if pad_v:
        # pad along V dimension: x is [B,1,V,T] => pad dim=2
        # torch.nn.functional.pad pads last dims first; so we permute to pad V easily
        x4 = x4.permute(0, 1, 3, 2)      # [B,1,T,V]
        x4 = F.pad(x4, (0, pad_v))       # pad last dim (V)
        x4 = x4.permute(0, 1, 3, 2)      # back to [B,1,V,T]
        V = V + pad_v

    grid_h = V // ph
    grid_w = T // pw

    # WaveFormer expects y-indices per patch row
    y = torch.arange(grid_h, device=x4.device, dtype=torch.long) + int(domain_offset)  # [grid_h]
    pos_embed_y = y.view(1, grid_h, 1).expand(B, grid_h, grid_w).contiguous()          # [B, grid_h, grid_w]

    return x4, pos_embed_y

def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid width
    return:
    pos_embed: [grid_size, embed_dim] or [1+grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_w = np.arange(grid_size, dtype=np.float32)

    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid_w)

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)

    return pos_embed
    

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

@torch.no_grad()
def validate_waveformer_forward_equivalence(model: nn.Module, x4: torch.Tensor, pos_embed_y: torch.Tensor, *, rtol=1e-3, atol=1e-3, rank=0):
    """
    Checks that:
      model(x, pos) ~= model.forward_head(model.forward_features(x, pos))
    If forward_features/forward_head are missing, we skip.
    """
    base = unwrap_ddp(model)

    if not (hasattr(base, "forward_features") and hasattr(base, "forward_head")):
        if rank == 0:
            print(">> WaveFormer check: forward_features/forward_head not found; skipping equivalence check.")
        return True

    base.eval()
    with torch.no_grad():
        out_fwd = base(x4, pos_embed_y)
        feat = base.forward_features(x4, pos_embed_y)
        out_path = base.forward_head(feat)

    ok = True

    if not isinstance(out_fwd, torch.Tensor) or not isinstance(out_path, torch.Tensor):
        if rank == 0:
            print(f">> WaveFormer check: non-tensor outputs: forward={type(out_fwd)} path={type(out_path)}")
        return False

    if out_fwd.shape != out_path.shape:
        ok = False
        if rank == 0:
            print(f">> WaveFormer check FAILED: shape mismatch forward{tuple(out_fwd.shape)} vs path{tuple(out_path.shape)}")

    # Numeric check (best-effort)
    if ok:
        diff = (out_fwd - out_path).abs().max().item()
        rel = diff / (out_path.abs().max().item() + 1e-8)
        if rank == 0:
            print(f">> WaveFormer check: max_abs_diff={diff:.4e}, max_rel_diff={rel:.4e}")
        if not torch.allclose(out_fwd, out_path, rtol=rtol, atol=atol):
            ok = False
            if rank == 0:
                print(">> WaveFormer check FAILED: forward != forward_head(forward_features) within tolerance.")
    return ok

class TeacherLogitsWrapperDataset(torch.utils.data.Dataset):
    """
    Wraps an existing dataset so __getitem__ returns (x, y, t_logits),
    where t_logits are read from an HDF5 dataset (default key 'teacher_logits').

    IMPORTANT: Opens HDF5 lazily per-worker to avoid multiprocessing issues.
    """
    def __init__(self, base_ds, logits_h5_path: str, logits_key: str = "teacher_logits"):
        self.base_ds = base_ds
        self.logits_h5_path = logits_h5_path
        self.logits_key = logits_key
        self._h5 = None
        self._dset = None

        # pass-through common attrs
        for attr in ("num_classes", "num_labels"):
            if hasattr(base_ds, attr):
                setattr(self, attr, getattr(base_ds, attr))

    def __len__(self):
        return len(self.base_ds)

    def __getattr__(self, name):
        if name in ("base_ds", "logits_h5_path", "logits_key", "_h5", "_dset"):
            raise AttributeError(name)
        return getattr(self.base_ds, name)

    def _lazy_open(self):
        if self._h5 is None:
            # read-only; SWMR isn't required if file isn't being written
            self._h5 = h5py.File(self.logits_h5_path, "r")
            self._dset = self._h5[self.logits_key]

    def __getitem__(self, idx):
        x, y = self.base_ds[idx]
        self._lazy_open()
        t = self._dset[idx]  # numpy array [C]
        t = torch.from_numpy(t).to(dtype=torch.float16)
        return x, y, t

class CEPlusSoftF1Macro(nn.Module):
    """
    Single-label multiclass:
      total_loss = CrossEntropyLoss + lam * (1 - SoftF1_macro_ovr)

    SoftF1 is computed per class in an OvR manner using softmax probabilities,
    then macro-averaged across classes.
    """
    def __init__(
        self,
        num_classes: int,
        lam: float = 0.25,
        weight: torch.Tensor | None = None,   # class weights for CE, shape [C]
        ignore_index: int = -100,
        eps: float = 1e-8,
        ce_reduction: str = "mean",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lam = lam
        self.ignore_index = ignore_index
        self.eps = eps
        self.ce = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_index,
            reduction=ce_reduction,
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, C]
        target: [B] (int64)
        """
        # --- CE term ---
        ce_loss = self.ce(logits, target)

        # --- mask ignore_index for SoftF1 as well ---
        if self.ignore_index is not None and self.ignore_index >= 0:
            valid = target != self.ignore_index
            logits = logits[valid]
            target = target[valid]
            if target.numel() == 0:
                return ce_loss  # nothing to compute for soft-F1

        # --- SoftF1 macro (OvR) ---
        probs = F.softmax(logits, dim=1)  # [B, C]
        y = F.one_hot(target, num_classes=self.num_classes).float()  # [B, C]

        # Soft counts per class
        tp = (probs * y).sum(dim=0)                 # [C]
        fp = (probs * (1.0 - y)).sum(dim=0)         # [C]
        fn = ((1.0 - probs) * y).sum(dim=0)         # [C]

        f1_per_class = (2.0 * tp) / (2.0 * tp + fp + fn + self.eps)  # [C]
        soft_f1_macro = f1_per_class.mean()

        soft_f1_loss = 1.0 - soft_f1_macro
        total = ce_loss + (self.lam * soft_f1_loss)
        return total
    
def make_balanced_sampler_singlelabel(ds, num_classes: int, pow_: float = 1.0):
    """
    ds[i] -> (x, y) where y is int class (or one-hot)
    Returns a WeightedRandomSampler over indices.
    """
    ys = []
    for i in range(len(ds)):
        _x, y = ds[i]
        if isinstance(y, torch.Tensor) and y.ndim == 0:
            yi = int(y.item())
        elif isinstance(y, torch.Tensor) and y.ndim == 1 and y.numel() == num_classes:
            yi = int(y.argmax().item())
        else:
            yi = int(y)
        ys.append(yi)

    ys = torch.tensor(ys, dtype=torch.long)
    counts = torch.bincount(ys, minlength=num_classes).float().clamp_min(1.0)

    # class weight ~ 1 / count^pow
    class_w = 1.0 / (counts ** pow_)
    sample_w = class_w[ys]

    sampler = WeightedRandomSampler(
        weights=sample_w,
        num_samples=len(ds),   # draw len(ds) samples per epoch
        replacement=True
    )
    return sampler, counts.cpu().numpy().astype(int).tolist()

def load_physiowave():
    rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{rank}")
    head_hidden_dim = 512
    head_dropout = 0.1
    pooling = "mean"
    in_channels = 8
    max_level = 3
    wave_kernel_size = 16
    wavelet_names = ["sym4", "sym5", "db6", "coif3", "bior4.4"]
    use_separate_channel = True
    patch_size = 64
    embed_dim = 256
    depth = 4
    num_heads = 8
    mlp_ratio = 4.0
    dropout = 0.1
    use_pos_embed = True
    pos_embed_type = "2d"
    num_classes = 53
    pretrained_path = "emg.pth"
    freeze_encoder = False

    head_config = {
        'hidden_dims': [head_hidden_dim],
        'dropout': head_dropout,
        'pooling': pooling
    }
    
    model = BERTWaveletTransformer(
        in_channels=in_channels,
        max_level=max_level,
        wave_kernel_size=wave_kernel_size,
        wavelet_names=wavelet_names,
        use_separate_channel=use_separate_channel,
        patch_size=(1, patch_size),
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        use_pos_embed=use_pos_embed,
        pos_embed_type=pos_embed_type,
        task_type='classification',
        num_classes=num_classes,
        head_config=head_config,
        pooling=pooling
    ).to(device)
    
    # Initialize weights
    if hasattr(model, 'initialize_weights'):
        model.initialize_weights()
        print("Initialized model weights")
    
    # Load pretrained weights
    # if pretrained_path:
    #     load_pretrained_feature_extractor(model, pretrained_path, rank)
    
    if freeze_encoder:
        for name, param in model.named_parameters():
            if 'task_heads' not in name:
                param.requires_grad = False
        print("Frozen encoder parameters (excluding task heads)")

    return model, None

@torch.no_grad()
def load_teacher_model_for_eval(args, device: torch.device, num_outputs: int, rank: int):
    """
    Loads teacher model for sanity evaluation on test data.

    - teacher_arch=bertwavelet: expects a BERTWaveletTransformer checkpoint compatible with build_teacher_for_kd()
    - teacher_arch=physiowave : expects a checkpoint saved by this script (best_student.pth) and loads via load_physiowave()
    """
    #if args.teacher_arch == "physiowave":
    #    # Build the exact PhysioWave config you hardcoded in load_physiowave()
    #    teacher, _ = load_physiowave()
    #
    #    ckpt = torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=False)
    #    sd = ckpt.get("student_state_dict") or ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    #
    #    # strict=True is good here: we want to know if something is wrong
    #    teacher.load_state_dict(sd, strict=True)
    #    teacher = teacher.to(device).eval()
    #    for p in teacher.parameters():
    #        p.requires_grad = False
    #    patch_wavelet_modules_io(teacher, rank=rank)
    #    return teacher, "physiowave"

    # default: original builder (BERTWaveletTransformer teacher checkpoint)
    teacher, _ = build_teacher_for_kd(args.teacher_checkpoint, args.task_type, num_outputs, rank=rank)
    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    patch_wavelet_modules_io(teacher, rank=rank)
    return teacher, "bertwavelet"

# ----------------------------
# Main worker
# ----------------------------
from .common import (
    autocast_ctx,
    fp32,
    load_repo_module,
    make_posenc_1d_concat,
    make_student_input,
    parse_kernel_set,
    patch_wavelet_modules_io,
    set_random_seed,
    unwrap_ddp,
    zscore_per_sample_channel,
)
from .data import (
    PreprocessWrapperDataset,
    TeacherLogitsWrapperDataset,
    collate_multilabel_with_tlogits,
    collate_singlelabel_with_tlogits,
    ecgfounder_preprocess_np,
    make_balanced_sampler_singlelabel,
)
from .losses import CEPlusSoftF1Macro, kd_bce_with_logits, kd_ce
from .schedulers import WarmupCosineSchedule
from .teacher import build_teacher_for_kd, load_physiowave, load_teacher_model_for_eval


def main_worker(rank, world_size, args):
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print("=" * 60)
        print("KD TRAINING (Teacher -> Multi-Backbone Student)")
        print("=" * 60)
        print(f"Task: {args.task_type} | max_length={args.max_length}")
        print(f"Student arch: {args.student_arch}")
        if args.student_arch == "waveformer":
            print(f"WaveFormer patch_width: {args.waveformer_patch_width}")
        if args.student_arch == "otis":
            print(f"OTiS model: {args.otis_model} | patch_width: {args.otis_patch_width}")
        print(f"Output dir: {args.output_dir}")

    # Datasets
    train_files = parse_file_paths(args.train_file)
    val_files   = parse_file_paths(args.val_file)
    test_files  = parse_file_paths(args.test_file) if args.test_file else []

    if args.task_type == 'multilabel':
        train_ds = MultiLabelTimeSeriesDataset(train_files, max_length=args.max_length,
                                              data_key=args.data_key, label_key=args.label_key)
        val_ds   = MultiLabelTimeSeriesDataset(val_files,   max_length=args.max_length,
                                              data_key=args.data_key, label_key=args.label_key)
        test_ds  = MultiLabelTimeSeriesDataset(test_files,  max_length=args.max_length,
                                              data_key=args.data_key, label_key=args.label_key) if test_files else None
        base_collate_fn = collate_multilabel_fn
        kd_outputs = train_ds.num_classes
    else:
        train_ds = SingleLabelTimeSeriesDataset(train_files, max_length=args.max_length,
                                                data_key=args.data_key, label_key=args.label_key)
        val_ds   = SingleLabelTimeSeriesDataset(val_files,   max_length=args.max_length, data_key=args.data_key, label_key=args.label_key) 
        test_ds  = SingleLabelTimeSeriesDataset(test_files,  max_length=args.max_length, data_key=args.data_key, label_key=args.label_key) if test_files else None
        base_collate_fn = collate_singlelabel_fn
        kd_outputs = train_ds.num_classes

    print(f"Num classes = {train_ds.num_classes}")

    use_cached_teacher = bool(args.teacher_logits_h5)

    train_collate_fn = base_collate_fn  # default

    if use_cached_teacher:
        if rank == 0:
            print(f">> Using cached teacher logits: {args.teacher_logits_h5} key={args.teacher_logits_key}")

        train_ds = TeacherLogitsWrapperDataset(train_ds, args.teacher_logits_h5, args.teacher_logits_key)

        if args.task_type == "multilabel":
            train_collate_fn = collate_multilabel_with_tlogits
        else:
            train_collate_fn = collate_singlelabel_with_tlogits

    # NEW - preprocessing
    def _pp_enabled_for_arch(pp_apply_to: str, arch: str) -> bool:
        if pp_apply_to in ("none",):
            return False
        if pp_apply_to == "all":
            return True
        return pp_apply_to == arch
    
    # NEW - preprocessing
    if args.pp_mode != "none" and _pp_enabled_for_arch(args.pp_apply_to, args.student_arch):
        if rank == 0:
            print(f">> Preprocessing enabled: mode={args.pp_mode} apply_to={args.pp_apply_to} fs={args.pp_fs}")
        train_ds = PreprocessWrapperDataset(train_ds, mode=args.pp_mode, fs=args.pp_fs, args=args)
        val_ds   = PreprocessWrapperDataset(val_ds,   mode=args.pp_mode, fs=args.pp_fs, args=args)
        if test_ds is not None:
            test_ds = PreprocessWrapperDataset(test_ds, mode=args.pp_mode, fs=args.pp_fs, args=args)

    if rank == 0:
        print(f"Train/Val/Test: {len(train_ds)}/{len(val_ds)}/{len(test_ds) if test_ds else 0}")
        print(f"Num classes: {train_ds.num_classes}")

    # Loaders
    train_sampler = None
    if args.balanced_sampler:
        if args.task_type != "classification":
            raise ValueError("--balanced_sampler only supports task_type=classification (single-label).")
        if world_size > 1:
            raise ValueError("--balanced_sampler not implemented for DDP > 1 in this script (needs per-rank sampling).")

        sampler, cls_counts = make_balanced_sampler_singlelabel(train_ds, train_ds.num_classes, pow_=args.sampler_pow)
        if rank == 0:
            print(f">> Using balanced sampler: pow={args.sampler_pow}")
            print(f">> Class counts: {cls_counts}")
        train_sampler = sampler
    else:
        # original behavior
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)

    val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler  = DistributedSampler(test_ds,  num_replicas=world_size, rank=rank, shuffle=False) if test_ds else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=False,  # IMPORTANT: don't shuffle when sampler is provided
        collate_fn=train_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=4 if args.num_workers > 0 else None,
        persistent_workers=(args.num_workers > 0)
    )
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, sampler=val_sampler,
                              collate_fn=base_collate_fn, num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, sampler=test_sampler,
                              collate_fn=base_collate_fn, num_workers=args.num_workers, pin_memory=True) if test_ds else None
    
    if rank == 0:
        item0 = train_ds[0]
        if isinstance(item0, (tuple, list)) and len(item0) == 3:
            x0, y0, t0 = item0
            print("SANITY y0.shape:", tuple(y0.shape), "t0.shape:", tuple(t0.shape),
                "num_classes:", train_ds.num_classes)
        else:
            x0, y0 = item0
            print("SANITY y0.shape:", tuple(y0.shape), "num_classes:", train_ds.num_classes)

    # Teacher
    teacher = None
    if (args.alpha_kd > 0) and (not use_cached_teacher):
        # TODO, only if dataset = db5 should we use load_physiowave
        teacher, _ = build_teacher_for_kd(args.teacher_checkpoint, args.task_type, kd_outputs, rank=rank)
        teacher = teacher.to(device).eval()
        for p in teacher.parameters():
            p.requires_grad = False
        #teacher, _ = load_physiowave()  # builds the exact PhysioWave architecture you used
        #ckpt = torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=False)
        #sd = ckpt.get("student_state_dict") or ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
        #teacher.load_state_dict(sd, strict=True)
        #teacher = teacher.to(device).eval()
        #for p in teacher.parameters():
        #    p.requires_grad = False
    elif use_cached_teacher and rank == 0:
        print(">> Skipping online teacher forward (cached logits will be used).")

    # Student
    if args.student_arch == "physiowave":
        student, pe_cache = load_physiowave() # TODO; save parameters for each dataset as if clause
    else:
        student, pe_cache = build_student(args, train_ds, device, rank)

    # NEW
    # Warmup forward to instantiate lazy heads for FrozenBackboneWithHead wrappers
    if args.student_arch in ("clef", "hubertecg", "ecgfm", "ecgfounder"):
        student.eval()
        with torch.no_grad():
            x0, _ = train_ds[0]
            x0 = x0.unsqueeze(0).to(device)  # [1,V,T]
            x0 = zscore_per_sample_channel(x0)
            x_in = make_student_input(x0, arch=args.student_arch, pe_cache=pe_cache, pe_scale=args.student_pe_scale)
            _ = student(x_in)
    
        # Freeze everything then unfreeze only the linear head
        base = unwrap_ddp(student)
        
        # NEW - TEMP
        out = base.backbone(x_in)       
        debug_hf_output(out, prefix="hubert_ecg")
    
        for p in base.parameters():
            p.requires_grad = True
        assert hasattr(base, "head") and base.head is not None
        for p in base.head.parameters():
            p.requires_grad = True
        if rank == 0:
            print(f">> {args.student_arch}: instantiated head; feat_dim={getattr(base, '_feat_dim', None)}")
        student.train()

    if rank == 0:
        total_params = sum(p.numel() for p in student.parameters())
        print(f"Student params: {total_params/1e6:.2f}M")

    # DDP: external repos often have unused params depending on forward path
    find_unused = args.student_arch in ("waveformer", "otis", "tinymyo")

    if args.student_arch == "otis":
        student = OTiSWrapper(student, patch_height=1, patch_width=args.otis_patch_width, domain_offset=0).to(device)
    
    use_ddp = world_size > 1
    if use_ddp:
        student = DDP(student, device_ids=[rank], find_unused_parameters=find_unused)

    def compute_class_counts_from_loader(train_loader, num_classes, device="cpu", use_cached_teacher=False):
        counts = torch.zeros(num_classes, dtype=torch.long, device=device)
        print("Computing clas counts")
        for batch in tqdm(train_loader):
            # adjust these two lines to match your batch structure
            if use_cached_teacher:
                x, y, _ = batch  # y: [B]
            else:
                x, y = batch
            y = y.to(device)
            counts += torch.bincount(y, minlength=num_classes)
        return counts

    # example usage
    #num_classes = train_ds.num_classes
    #counts = compute_class_counts_from_loader(train_loader, num_classes, device="cpu", use_cached_teacher = use_cached_teacher)
    #print(f"Counts = {counts}")

    # inverse-frequency weights (normalized)
    #weights = 1.0 / (counts.float().clamp_min(1.0) ** 0.5)
    #weights = weights / weights.mean()
    
    # Loss
    hard_criterion = nn.CrossEntropyLoss() if args.task_type == "classification" else nn.BCEWithLogitsLoss()
    #hard_criterion = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.0)
    #weights = None  # or your computed weights tensor on device
    #hard_criterion = CEPlusSoftF1Macro(
    #    num_classes=num_classes,
    #    lam=float(getattr(args, "softf1_lambda", 0.3)),
    #    weight=(weights.to(device) if weights is not None else None),
    #    ignore_index=getattr(args, "ignore_index", -100),
    #).to(device)
    # hard_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Optim + scheduler
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, student.parameters()),
                            lr=args.lr, weight_decay=args.weight_decay)

    n_trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print("trainable params:", n_trainable)
    for name, p in unwrap_ddp(student).named_parameters():
        if p.requires_grad:
            print("trainable:", name, p.shape)

    amp_enabled = bool(args.amp) and device.type == "cuda"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    scaler = GradScaler(enabled=(amp_enabled and amp_dtype == torch.float16))
    if rank == 0:
        print(f">> AMP: enabled={amp_enabled} dtype={args.amp_dtype} (GradScaler={'on' if scaler.is_enabled() else 'off'})")

    scheduler = None
    scheduler_per_batch = False
    if args.scheduler == 'cosine':
        micro_batches = len(train_loader)
        eff_steps_per_epoch = math.ceil(micro_batches / max(1, args.accum_steps))
        total_steps = args.epochs * eff_steps_per_epoch
        warmup_steps = (args.warmup_epochs * eff_steps_per_epoch) if args.warmup_epochs > 0 else int(0.1 * total_steps)
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps, total_steps)
        scheduler_per_batch = True
        if rank == 0:
            print(f"WarmupCosine: warmup_steps={warmup_steps}, total_steps={total_steps}")

    best_val = float('inf')
    best_f1_macro = 0.0

    if args.student_arch == "waveformer" and rank == 0:
        student.eval()
        with torch.no_grad():
            x0, _ = train_ds[0]
            x0 = x0.unsqueeze(0).to(device)  # [1,V,T]
            x_in = make_student_input(x0, arch="waveformer", pe_cache=None, pe_scale=args.student_pe_scale)  # [1,1,V,T]
            x_in, pos_embed_y = make_waveformer_pos_embed_y(
                x_in, patch_height=1, patch_width=args.waveformer_patch_width, domain_offset=0
            )
            _ = validate_waveformer_forward_equivalence(student, x_in, pos_embed_y, rank=rank)

    # ---- Teacher sanity check on TEST ----
    if rank == 0 and args.sanity_teacher_test and test_loader is not None:
        print("\n===> SANITY: Evaluating TEACHER on TEST split <===")

        teacher_model, teacher_kind = load_teacher_model_for_eval(
            args, device=device, num_outputs=kd_outputs, rank=rank
        )

        # simple criterion for reporting loss (doesn't affect metrics)
        if args.task_type == "classification":
            teacher_crit = nn.CrossEntropyLoss().to(device)
        else:
            teacher_crit = nn.BCEWithLogitsLoss().to(device)

        # Use eval_unified to print the full metric block
        _ = eval_unified(
            epoch="TeacherTest",
            rank=rank,
            model=teacher_model,
            loader=test_loader,
            device=device,
            criterion=teacher_crit,
            threshold=args.threshold,
            desc_prefix=f"Teacher({teacher_kind}) TEST",
            task_type=args.task_type,
            pe_cache=None,
            student_arch=("physiowave" if teacher_kind == "physiowave" else "physiowave"),  # see note below
            pe_scale=0.0,
            waveformer_patch_width=args.waveformer_patch_width,
            amp_enabled=False,
            amp_dtype=(torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16),
        )

        # free memory explicitly
        del teacher_model
        torch.cuda.empty_cache()
        print("===> SANITY: Teacher test eval done <===\n")

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        student.train()

        pbar = tqdm(total=len(train_loader), desc=f"Train Epoch {epoch}", ncols=120) if rank == 0 else None

        total_loss, total_samples = 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            if use_cached_teacher:
                x, y, t_logits_cached = batch
            else:
                x, y = batch
                t_logits_cached = None
            x = x.to(device)  # [B,V,T]
            y = y.to(device)

            # if args.student_arch in ("ecgfm", "ecgfounder", "clef", "hubertecg"):
            if args.student_arch in ("ecgfm", "ecgfounder", "clef", "hubertecg", "fcn", "bilstm", "ai85tcn1d"):
                x = zscore_per_sample_channel(x)

            x_in = make_student_input(x, arch=args.student_arch, pe_cache=pe_cache, pe_scale=args.student_pe_scale)

            is_accum = ((step + 1) % args.accum_steps == 0)
            sync_ctx = (student.no_sync() if isinstance(student, DDP) and not is_accum else nullcontext())

            with sync_ctx:
                x_used = x
                lam = 1.0
                y_a = y
                y_b = y

                # Teacher logits (use same mixed input if mixup enabled)
                t_logits = None
                if use_cached_teacher:
                    # cached teacher logits come from CPU as float16
                    t_logits = t_logits_cached.to(device, non_blocking=True)
                elif teacher is not None:
                    with torch.inference_mode():
                        with autocast_ctx(enabled=(amp_enabled and device.type == "cuda"), dtype=amp_dtype):
                            t_logits = teacher(x_used, task="downstream", task_name=args.task_type, return_logits=True)

                # Student input formatting
                x_in = make_student_input(x_used, arch=args.student_arch, pe_cache=pe_cache, pe_scale=args.student_pe_scale)

                with autocast_ctx(enabled=(amp_enabled and device.type == "cuda"), dtype=amp_dtype):
                    # Forward
                    if args.student_arch == "waveformer":
                        x_in, pos_embed_y = make_waveformer_pos_embed_y(
                            x_in, patch_height=1, patch_width=args.waveformer_patch_width, domain_offset=0
                        )
                        s_logits = student(x_in, pos_embed_y)
                    elif args.student_arch == "physiowave":
                        s_logits = student(x_in, task='downstream', task_name='classification')
                        if isinstance(s_logits, (tuple, list)): s_logits = s_logits[0]
                        assert s_logits.dim() == 2, f"PhysioWave should output [B,C], got {tuple(s_logits.shape)}"
                    else:
                        s_logits = student(x_in)

                    if isinstance(s_logits, (tuple, list)):
                        s_logits = s_logits[0]
                    if isinstance(s_logits, dict):
                        s_logits = s_logits.get("logits", None) or s_logits.get("preds", None) or s_logits.get("y_hat", None) or next(iter(s_logits.values()))

                    # Prevent silent token-mean on WaveFormer (should be [B,C])
                    if args.student_arch == "waveformer":
                        assert s_logits.dim() == 2, f"WaveFormer logits must be [B,C], got {tuple(s_logits.shape)}"
                    else:
                        if s_logits.dim() == 3:
                            s_logits = s_logits.mean(dim=-1)
                        elif s_logits.dim() == 4:
                            s_logits = s_logits.flatten(2).mean(-1)

                    s_logits_f = fp32(s_logits)
                    t_logits_f = fp32(t_logits) if (t_logits is not None) else None

                    # Hard loss (with mixup support)
                    if args.task_type == "classification":
                        hard_loss = hard_criterion(s_logits_f, y)   # or your soft-target CE if using mixup
                        soft_loss = kd_ce(s_logits_f, t_logits_f, T=args.temperature) if (t_logits_f is not None) else 0.0
                    else:
                        hard_loss = hard_criterion(s_logits_f, y.float())
                        soft_loss = kd_bce_with_logits(s_logits_f, t_logits_f, T=args.temperature) if (t_logits_f is not None) else 0.0

                    alpha = float(args.alpha_kd)
                    loss = (1.0 - alpha) * hard_loss + (alpha * soft_loss if (t_logits_f is not None) else 0.0)

                # Backward (AMP-aware)
                loss_to_backprop = loss / args.accum_steps
                if scaler.is_enabled():
                    scaler.scale(loss_to_backprop).backward()
                else:
                    loss_to_backprop.backward()

            bs = x.size(0)
            total_loss += float(loss.item()) * bs
            total_samples += bs

            if is_accum:
                if args.grad_clip > 0.0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

                if scheduler is not None and scheduler_per_batch:
                    scheduler.step()

            if rank == 0:
                curr_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix({"loss": f"{total_loss/max(1,total_samples):.4f}", "lr": f"{curr_lr:.6f}"})
                pbar.update(1)

        if rank == 0:
            pbar.close()

        val_metrics = eval_unified(
            epoch, rank, student, val_loader, device, hard_criterion,
            threshold=args.threshold, desc_prefix="Val",
            task_type=args.task_type, pe_cache=pe_cache,
            student_arch=args.student_arch, pe_scale=args.student_pe_scale,
            waveformer_patch_width=args.waveformer_patch_width,
            amp_enabled=False, amp_dtype=amp_dtype,
        )
        val_loss = val_metrics["loss"]
        val_f1_macro = val_metrics["f1_macro"]

        #if rank == 0 and val_loss < best_val:
        if rank == 0  and val_f1_macro > best_f1_macro:
            best_val = val_loss
            best_f1_macro = val_f1_macro
            save_path = os.path.join(args.output_dir, "best_student.pth")

            base_student = unwrap_ddp(student)

            torch.save({
                "epoch": epoch,
                "student_state_dict": base_student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "val_metrics": val_metrics,
                "args": vars(args),
            }, save_path)

            print(f"Saved best student @ epoch {epoch} -> {save_path}")

    if rank == 0 and test_loader is not None:
        print("\n===> Testing best student <===")
        ckpt = torch.load(os.path.join(args.output_dir, "best_student.pth"), map_location="cpu", weights_only=False)
        unwrap_ddp(student).load_state_dict(ckpt["student_state_dict"])
        #student.module.load_state_dict(ckpt["student_state_dict"])
        test_metrics = eval_unified(
            "Test", rank, student, test_loader, device, hard_criterion,
            threshold=args.threshold, desc_prefix="Test",
            task_type=args.task_type, pe_cache=pe_cache,
            student_arch=args.student_arch, pe_scale=args.student_pe_scale,
            waveformer_patch_width=args.waveformer_patch_width,
            amp_enabled=False, amp_dtype=amp_dtype
        )
        with open(os.path.join(args.output_dir, "test_results_kd.json"), "w") as f:
            json.dump(test_metrics, f, indent=4)

    dist.destroy_process_group()

# ----------------------------
# Argparse
# ----------------------------
def _load_config_defaults(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        if config_path.endswith(".json"):
            cfg = json.load(f)
        elif config_path.endswith((".yml", ".yaml")):
            try:
                import yaml  # type: ignore
            except Exception as exc:
                raise RuntimeError(
                    "YAML config requested but PyYAML is not installed. "
                    "Install pyyaml or use JSON config files."
                ) from exc
            cfg = yaml.safe_load(f)
        else:
            raise ValueError("Unsupported --config format. Use .json, .yml, or .yaml")

    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping/object, got {type(cfg)}")
    return cfg


def main(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default="")
    pre_args, _ = pre.parse_known_args(argv)

    p = argparse.ArgumentParser(description="KD: Teacher -> Multi-backbone Student")
    p.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to JSON/YAML config file. Values are used as defaults; explicit CLI flags override them.",
    )

    # Data
    p.add_argument("--train_file", type=str, default="")
    p.add_argument("--val_file",   type=str, default="")
    p.add_argument("--test_file",  type=str, default="")
    p.add_argument("--data_key",   type=str, default="data")
    p.add_argument("--label_key",  type=str, default="label")
    p.add_argument("--max_length", type=int, default=1024)

    # Task
    p.add_argument("--task_type", type=str, default="classification",
                   choices=["multilabel", "classification"])
    p.add_argument("--threshold", type=float, default=0.5)

    # Dist / I/O
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs",     type=int, default=50)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--num_workers",  type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grad_clip", type=float, default=0.0)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--world_size", type=int, default=1)
    p.add_argument("--output_dir", type=str, default="./kd_output")
    p.add_argument("--accum_steps", type=int, default=1)
    p.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "none"])

    # Teacher
    p.add_argument("--teacher_checkpoint", type=str, default="")

    # KD
    p.add_argument("--alpha_kd", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=2.0)

    # Shared signal shape
    p.add_argument("--in_channels", type=int, default=8, help="Number of variates/electrodes V (EPN-612=8).")
    p.add_argument("--domain_name", type=str, default="emg612")

    # Student selector
    # p.add_argument("--student_arch", type=str, default="waveformer", choices=["physiowavenpu", "waveformer", "otis", "tinymyo","ecgfm", "ecgfounder", "clef", "hubertecg"])
    p.add_argument(
        "--student_arch",
        type=str,
        default="physiowavenpu",
        choices=[
            "physiowavenpu", "waveformer", "otis", "tinymyo",
            "ecgfm", "ecgfounder", "clef", "hubertecg",
            "fcn", "tcn", "bilstm", "convnext1d", "ai85tcn1d"
        ],
    )

    # PhysioWaveNPU
    p.add_argument("--patch_size", type=int, default=40)
    p.add_argument("--student_embed_dim", type=int, default=256)
    p.add_argument("--student_depth", type=int, default=3)
    p.add_argument("--student_bank_ch", type=int, default=16)
    p.add_argument("--student_reduce", type=int, default=4)
    p.add_argument("--student_pos_freqs", type=int, default=8)
    p.add_argument("--student_pe_scale", type=float, default=0.1)

    p.add_argument(
        "--student_kernel_set",
        type=str,
        default="3,5,7",
        help="Comma-separated odd kernel sizes for wavelet-like front-end branches. "
            "Examples: '3,5,7' (baseline), '7' (single-branch), '3' (single-branch)."
    )

    # WaveFormer
    p.add_argument("--waveformer_repo_root", type=str, default="")
    p.add_argument("--waveformer_model", type=str, default="Waveformer_base")
    p.add_argument("--waveformer_patch_width", type=int, default=64)

    # OTiS
    p.add_argument("--otis_repo_root", type=str, default="")
    p.add_argument("--otis_finetune", type=str, default="", help="Path to OTiS pretrained checkpoint (.pth)")
    p.add_argument("--otis_model", type=str, default="vit_baseDeep_patchX")
    p.add_argument("--otis_drop_path", type=float, default=0.1)
    p.add_argument("--otis_global_pool", action="store_true", default=False)
    p.add_argument("--otis_attention_pool", action="store_true", default=False)

    # OTiS patching (set to WaveFormer-aligned defaults)
    p.add_argument("--otis_input_channels", type=int, default=1)
    p.add_argument("--otis_input_variates", type=int, default=8)
    p.add_argument("--otis_time_steps", type=int, default=1024)
    p.add_argument("--otis_patch_height", type=int, default=1)
    p.add_argument("--otis_patch_width", type=int, default=64)

    # TinyMyo
    p.add_argument("--tinymyo_repo_root", type=str, default="")
    p.add_argument("--tinymyo_config_dir", type=str, default="", help="Path to BioFoundation config dir (the one with defaults.yaml etc.)")
    p.add_argument("--tinymyo_config_name", type=str, default="", help="Hydra config name, e.g. 'defaults'")
    p.add_argument("--tinymyo_overrides", nargs="*", default=[], help="Hydra overrides, e.g. task.model.num_classes=6")
    p.add_argument("--tinymyo_pretrained_ckpt", type=str, default="")
    p.add_argument("--tinymyo_pretrained_safetensors", type=str, default="")

    # NEW
    # ECG-FM
    p.add_argument("--ecgfm_ckpt", type=str, default="", help="Path to ECG-FM checkpoint .pt (e.g., mimic_iv_ecg_physionet_pretrained.pt)")
    # ECGFounder
    p.add_argument("--ecgfounder_repo_root", type=str, default="", help="Path to cloned PKUDigitalHealth/ECGFounder repo")
    p.add_argument("--ecgfounder_ckpt", type=str, default="", help="Path to ECGFounder checkpoint .pth (1-lead or 12-lead)")
    p.add_argument("--ecgfounder_lead_config", type=str, default="12lead", choices=["1lead", "12lead"])
    p.add_argument("--ecgfounder_linear_probe", action="store_true", default=False, help="Freeze backbone, train only final dense")
    # CLEF
    p.add_argument("--clef_repo_root", type=str, default="", help="Path to cloned Nokia-Bell-Labs/ecg-foundation-model repo")
    p.add_argument("--clef_ckpt", type=str, default="", help="Path to CLEF checkpoint (downloaded from Zenodo)")
    p.add_argument("--clef_model_size", type=str, default="medium", choices=["small", "medium", "large"])
    p.add_argument("--clef_in_channels", type=int, default=1, help="Input channels to CLEF backbone (typically 1 for single-lead, 12 for 12-lead)")
    # HuBERT-ECG
    p.add_argument("--hubertecg_model_id", type=str, default="Edoardo-BS/HuBERT-ECG", help="HF model id for HuBERT-ECG")
    p.add_argument("--hubertecg_trust_remote_code", action="store_true", default=False)
    p.add_argument("--hubertecg_revision", type=str, default="")

    # NEW - Preprocessing
    p.add_argument("--pp_mode", type=str, default="none", choices=["none", "ecgfounder"])
    p.add_argument("--pp_apply_to", type=str, default="all",
                choices=["all", "hubertecg", "ecgfm", "ecgfounder", "clef", "none"])
    p.add_argument("--pp_fs", type=float, default=500.0, help="Sampling rate used for preprocessing filters.")
    p.add_argument("--pp_notch_freq", type=float, default=50.0)
    p.add_argument("--pp_notch_q", type=float, default=30.0)
    p.add_argument("--pp_band_low", type=float, default=0.67)
    p.add_argument("--pp_band_high", type=float, default=40.0)
    p.add_argument("--pp_band_order", type=int, default=4)
    p.add_argument("--pp_baseline_kernel_sec", type=float, default=0.4)
    p.add_argument("--pp_zscore", action="store_true", default=False,
                help="Apply ECGFounder-style global zscore after filtering.")
    p.add_argument("--pp_zscore_per_channel", action="store_true", default=False,
                help="Additionally apply your zscore_per_sample_channel() in the training loop (torch).")

    # NEW baselines: common sizing knobs
    p.add_argument("--baseline_width", type=int, default=128, help="Base channel/hidden width for baseline models.")
    p.add_argument("--baseline_depth", type=int, default=6, help="Depth for TCN/ConvNeXt1D blocks.")
    p.add_argument("--baseline_dropout", type=float, default=0.1, help="Dropout for baseline models.")
    p.add_argument("--tcn_kernel", type=int, default=3, help="Kernel size for TCN.")
    p.add_argument("--tcn_dilated", action="store_true", default=True, help="Use dilations in TCN blocks.")
    p.add_argument("--bilstm_layers", type=int, default=2, help="Number of BiLSTM layers.")

    p.add_argument("--balanced_sampler", action="store_true",
                           help="Use WeightedRandomSampler to balance classes in TRAIN loader (single-label only).")

    p.add_argument("--sampler_pow", type=float, default=1.0,
                           help="Sampler weight exponent. 1.0=inv_freq, 0.5=sqrt inv_freq (less aggressive).")

    p.add_argument("--amp", action="store_true", default=False, help="Enable torch.cuda.amp autocast + GradScaler.")
    p.add_argument("--use_amp", action="store_true", default=False, help="Alias for --amp.")
    p.add_argument("--amp_dtype", type=str, default="fp16", choices=["fp16", "bf16"], help="AMP dtype for autocast.")

    p.add_argument("--softf1_start_epoch", type=int, default=6,
                help="Epoch to start CEPlusSoftF1Macro (0-indexed). Before this: CE only.")
    p.add_argument("--softf1_lambda", type=float, default=0.3,
                help="Lambda for CEPlusSoftF1Macro once enabled.")

    p.add_argument("--teacher_logits_h5", type=str, default="",
                help="Optional HDF5 file containing precomputed teacher logits aligned with train dataset order.")
    p.add_argument("--teacher_logits_key", type=str, default="teacher_logits",
                help="Dataset key inside HDF5 (default: teacher_logits).")
    p.add_argument("--sanity_teacher_test", action="store_true", default=False,
                help="Run a one-off teacher evaluation on the TEST split before training.")
    p.add_argument("--teacher_arch", type=str, default="bertwavelet",
                choices=["bertwavelet", "physiowave"],
                help="How to load --teacher_checkpoint for the sanity test.")
    
    if pre_args.config:
        config_defaults = _load_config_defaults(pre_args.config)
        valid_keys = {a.dest for a in p._actions}
        unknown_keys = sorted(k for k in config_defaults.keys() if k not in valid_keys)
        if unknown_keys:
            raise ValueError(
                f"Unknown keys in config '{pre_args.config}': {unknown_keys}. "
                "Use 'python run_kd.py --help' for valid argument names."
            )
        p.set_defaults(**config_defaults)

    args = p.parse_args(argv)
    if args.use_amp:
        args.amp = True
    if not args.train_file:
        p.error("--train_file is required (either CLI or via --config).")
    if not args.val_file:
        p.error("--val_file is required (either CLI or via --config).")
    if not args.teacher_checkpoint:
        p.error("--teacher_checkpoint is required (either CLI or via --config).")

    set_random_seed(args.seed)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
    main_worker(local_rank, world_size, args)

if __name__ == "__main__":
    main()
