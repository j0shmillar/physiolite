import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
import argparse
import random
import json
import importlib.util
from copy import deepcopy
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

from sklearn.metrics import (
    accuracy_score,
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    hamming_loss, jaccard_score)

from models.physiowave import BERTWaveletTransformer
from models.vit_pw import PhysioWaveNPU
from timeseries_ds import (
    SingleLabelTimeSeriesDataset,
    MultiLabelTimeSeriesDataset,
    collate_multilabel_fn,
    parse_file_paths,
    collate_singlelabel_fn)

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
)
from .losses import kd_bce_with_logits, kd_ce
from .schedulers import WarmupCosineSchedule
from .teacher import build_teacher_for_kd, load_physiowave

try:
    from scipy.signal import iirnotch, butter, filtfilt
    from scipy.signal import medfilt
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False



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
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", dtype=dtype)
    return torch.cuda.amp.autocast(dtype=dtype)

def collate_singlelabel_with_tlogits(batch):
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


def configure_reproducibility(seed: int, deterministic: bool = False):
    set_random_seed(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def build_hard_criterion(args, train_ds, device):
    if args.task_type == "multilabel":
        if args.hard_loss != "bce":
            raise ValueError("--hard_loss must be 'bce' for task_type=multilabel.")
        return nn.BCEWithLogitsLoss().to(device)

    if args.hard_loss == "ce":
        return nn.CrossEntropyLoss().to(device)
    if args.hard_loss == "ce_softf1":
        return CEPlusSoftF1Macro(
            num_classes=train_ds.num_classes,
            lam=float(args.softf1_lambda),
        ).to(device)
    raise ValueError(f"Unsupported --hard_loss={args.hard_loss}")


def get_save_metric(metrics: dict[str, Any], criterion_name: str) -> tuple[float, bool]:
    if criterion_name == "loss":
        return float(metrics["loss"]), False
    if criterion_name == "f1_macro":
        return float(metrics["f1_macro"]), True
    if criterion_name == "f1_weighted":
        return float(metrics["f1_weighted"]), True
    if criterion_name == "accuracy":
        return float(metrics["accuracy"]), True
    raise ValueError(f"Unsupported save criterion: {criterion_name}")

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


STUDENT_DATASET_PROFILES = {
    # "uci": {"patch_t": 4, "front_pool_k": 3, "post_patch_pool_t": 5, "student_pos_freqs": 8},
    "uci": {"patch_t": 4, "front_pool_k": 4, "post_patch_pool_t": 4, "student_pos_freqs": 8},
    "db5": {"patch_t": 8, "front_pool_k": 4, "post_patch_pool_t": 2, "student_pos_freqs": 8},
    "epn612": {"patch_t": 4, "front_pool_k": 4, "post_patch_pool_t": 4, "student_pos_freqs": 8},
    "ptb": {"patch_t": 8, "front_pool_k": 4, "post_patch_pool_t": 4, "student_pos_freqs": 8},
    "cpsc": {"patch_t": 8, "front_pool_k": 4, "post_patch_pool_t": 4, "student_pos_freqs": 8},
    "chapman": {"patch_t": 16, "front_pool_k": 4, "post_patch_pool_t": 4, "student_pos_freqs": 8},
}


def infer_student_dataset_profile(train_file: str) -> str:
    s = str(train_file).lower()
    if "uci_emg" in s:
        return "uci"
    if "db5" in s or "ninapro" in s:
        return "db5"
    if "epn612" in s:
        return "epn612"
    if "ptb" in s or "ptb-xl" in s:
        return "ptb"
    if "cpsc" in s:
        return "cpsc"
    if "chapman" in s or "shaoxing" in s:
        return "chapman"
    return "none"


def apply_student_dataset_profile(args, rank: int):
    profile = args.student_dataset_profile
    if profile == "auto":
        profile = infer_student_dataset_profile(args.train_file)
        if rank == 0:
            print(f">> Auto-selected student dataset profile: {profile}")
    if profile == "none":
        if rank == 0:
            print(
                "!! WARNING: student dataset profile is 'none'. "
                "Using raw CLI values for patch/front/post-pool/pos-freqs."
            )
        return

    cfg = STUDENT_DATASET_PROFILES[profile]
    args.patch_size = int(cfg["patch_t"])
    args.student_front_pool_k = int(cfg["front_pool_k"])
    args.student_post_patch_pool_t = int(cfg["post_patch_pool_t"])
    args.student_pos_freqs = int(cfg["student_pos_freqs"])
    if rank == 0:
        print(
            ">> Applied student profile "
            f"'{profile}': patch_t={args.patch_size}, "
            f"front_pool_k={args.student_front_pool_k}, "
            f"post_patch_pool_t={args.student_post_patch_pool_t}, "
            f"student_pos_freqs={args.student_pos_freqs}"
        )

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
        pe = pe_cache.unsqueeze(0).expand(x.size(0), -1, -1)  # [B, 2F, T]
        pe = pe_scale * pe
        return torch.cat([x, pe], dim=1)  # [B, V+2F, T]
    elif arch == "physiowave":
        patch_t = 64  # must match the model's patch_size[1]
        T = x.size(-1)
        pad_t = (patch_t - (T % patch_t)) % patch_t
        if pad_t:
            x = F.pad(x, (0, pad_t))
        return x
    raise ValueError(f"Unknown student arch: {arch}")

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


@torch.no_grad()
def eval_unified(epoch, rank, model, loader, device, criterion,
                 threshold=0.5, desc_prefix="Val", task_type=None,
                 pe_cache=None, student_arch="physiowavenpu", pe_scale=0.1, amp_enabled=False, amp_dtype=torch.float16):

    model.eval()
    total_loss, total_samples = 0.0, 0
    all_logits, all_labels = [], []

    disp = loader if rank != 0 else tqdm(loader, desc=f"{desc_prefix} Epoch {epoch}", ncols=120)
    for batch in disp:
        x, y = batch[0], batch[1]

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if student_arch in ("ecgfm","ecgfounder","clef","hubertecg","fcn", "bilstm", "ai85tcn1d"):
            x = zscore_per_sample_channel(x)

        with autocast_ctx(enabled=(amp_enabled and device.type == "cuda"), dtype=amp_dtype):
            x_in = make_student_input(x, arch=student_arch, pe_cache=pe_cache, pe_scale=pe_scale)

            if student_arch == "physiowave":
                pw_task = "multilabel" if task_type == "multilabel" else "classification"
                logits = model(
                    x_in,
                    task="downstream",
                    task_name=pw_task,
                    return_logits=(pw_task == "multilabel"),
                )
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
        f1_macro_no_rest = None
        if num_classes == 53:
            f1_macro_no_rest = f1_score(
                y_true, y_pred, average="macro", labels=list(range(1, 53)), zero_division=0
            )
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
            if f1_macro_no_rest is not None:
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
            f1_macro_no_rest=f1_macro_no_rest,
            auroc_micro=auroc_micro,    auroc_macro=auroc_macro,
            ap_micro=ap_micro,          ap_macro=ap_macro,
            hamming_loss=ham,
            jaccard_micro=jac_micro,    jaccard_macro=jac_macro
        ))
        return metrics

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
        if isinstance(out, dict):
            for k in ("last_hidden_state", "encoder_out", "extract_features", "features", "embeddings", "embedding", "hidden_states", "repr", "representations"):
                if k in out and isinstance(out[k], torch.Tensor):
                    return out[k]
        for attr in ("features", "feats", "embeddings", "embedding", "last_hidden_state", "hidden_states"):
            if hasattr(out, attr):
                v = getattr(out, attr)
                if isinstance(v, torch.Tensor):
                    return v
        if isinstance(out, (tuple, list)):
            for v in out:
                if isinstance(v, torch.Tensor):
                    return v
                if isinstance(v, dict):
                    try:
                        return FrozenBackboneWithHead._to_feat_tensor(v)
                    except Exception:
                        pass
        if isinstance(out, torch.Tensor):
            return out
        raise RuntimeError(f"Could not extract tensor features from output type={type(out)}")

    @staticmethod
    def _pool_to_bxd(t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 2:
            return t

        if t.dim() == 3:
            B, A, C = t.shape
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
        with torch.no_grad():
            out = self.backbone(x)
            feat = self._to_feat_tensor(out)
            feat = self._pool_to_bxd(feat)

        self._ensure_head(feat)
        return self.head(feat)

def build_student(args, train_ds, device, rank):
    pe_cache = None

    if args.student_arch == "physiowavenpu":
        apply_student_dataset_profile(args, rank)

        depth = int(args.student_depth)  # allow 0,1,3
        embed = int(args.student_embed_dim)

        kernel_set = parse_kernel_set(args.student_kernel_set)

        pos_freqs = int(args.student_pos_freqs)
        C_total = int(args.in_channels) + 2 * pos_freqs
        if args.print_student_config and rank == 0:
            print(">> Resolved student config:")
            print(f"   student_arch={args.student_arch}")
            print(f"   student_dataset_profile={args.student_dataset_profile}")
            print(f"   in_channels_raw={args.in_channels}")
            print(f"   student_pos_freqs={pos_freqs}")
            print(f"   C_total={C_total}")
            print(f"   input_length={args.max_length}")
            print(f"   patch_t={args.patch_size}")
            print(f"   front_pool_k={args.student_front_pool_k}")
            print(f"   post_patch_pool_t={args.student_post_patch_pool_t}")
            print(f"   student_bank_ch={args.student_bank_ch}")
            print(f"   student_kernel_set={args.student_kernel_set}")
            print(f"   student_embed_dim={args.student_embed_dim}")
            print(f"   student_depth={args.student_depth}")
            print(f"   student_reduce={args.student_reduce}")

        student = PhysioWaveNPU(
            input_length=args.max_length,
            in_channels=C_total,
            num_classes=train_ds.num_classes,
            bank_ch=args.student_bank_ch,
            kernel_set=kernel_set,     
            patch_t=args.patch_size,
            embed_dim=embed,
            depth=depth,                 
            reduce=args.student_reduce,
            conv1d_k=3,
            head_residual=False,
            front_pool_k=args.student_front_pool_k,
            post_patch_pool_t=args.student_post_patch_pool_t,
        ).to(device)

        pe_cache = make_posenc_1d_concat(pos_freqs, args.max_length, device=device)
        return student, pe_cache

    raise ValueError(f"Unknown student_arch: {args.student_arch}")


def _extract_checkpoint_state_dict(ckpt_obj):
    return ckpt_obj.get("student_state_dict") or ckpt_obj.get("model_state_dict") or ckpt_obj.get("state_dict") or ckpt_obj


def _extract_teacher_checkpoint_state_dict(ckpt_obj):
    return ckpt_obj.get("model_state_dict") or ckpt_obj.get("state_dict") or ckpt_obj


def build_teacher_model(args, train_ds, device, rank, kd_outputs):
    """
    Build teacher for online KD logits.
    - teacher_model=physiowave: loads BERTWavelet teacher via --teacher_checkpoint
    - teacher_model=<other backbone>: reuses student builder for that backbone, then loads --teacher_checkpoint state dict.
    """
    teacher_pe_cache = None

    if args.teacher_model == "physiowave":
        teacher_profile = args.teacher_dataset_profile
        if teacher_profile == "none" and args.student_dataset_profile != "none":
            teacher_profile = args.student_dataset_profile
        if teacher_profile in ("none", "auto"):
            teacher_profile = infer_student_dataset_profile(args.train_file)
        if rank == 0:
            print(f">> Resolved teacher profile: {teacher_profile}")

        teacher, head_loaded = build_teacher_for_kd(args.teacher_checkpoint, args.task_type, kd_outputs, rank=rank)
        teacher = teacher.to(device)
        if teacher_profile == "db5" and not head_loaded:
            if rank == 0:
                print(">> Falling back to DB5 PhysioWave template because checkpoint head metadata is incomplete.")
            teacher, _ = load_physiowave()
            ckpt = torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=False)
            sd = _extract_teacher_checkpoint_state_dict(ckpt)
            if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
                sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}

            model_sd = teacher.state_dict()
            shape_mismatch = []
            filtered_sd = {}
            for k, v in sd.items():
                if k not in model_sd:
                    continue
                if tuple(v.shape) != tuple(model_sd[k].shape):
                    shape_mismatch.append((k, tuple(model_sd[k].shape), tuple(v.shape)))
                    continue
                filtered_sd[k] = v

            msg = teacher.load_state_dict(filtered_sd, strict=False)
            teacher = teacher.to(device)
            if rank == 0:
                print(">> Loaded DB5 PhysioWave teacher checkpoint with DB5 fallback config.")
                print(
                    f"   matched={len(filtered_sd)} "
                    f"missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)} "
                    f"shape_mismatch={len(shape_mismatch)}"
                )
        patch_wavelet_modules_io(teacher, rank=rank)
    else:
        t_args = deepcopy(args)
        t_args.student_arch = args.teacher_model
        if args.teacher_dataset_profile != "none":
            t_args.student_dataset_profile = args.teacher_dataset_profile
        teacher, teacher_pe_cache = build_student(t_args, train_ds, device, rank)

        ckpt = torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=False)
        sd = _extract_checkpoint_state_dict(ckpt)
        missing, unexpected = unwrap_ddp(teacher).load_state_dict(sd, strict=False)
        if rank == 0:
            print(
                f">> Loaded teacher ({args.teacher_model}) from checkpoint. "
                f"Missing={len(missing)}, Unexpected={len(unexpected)}"
            )

    teacher = teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher, teacher_pe_cache


def forward_model_logits(model, x, *, arch, args, pe_cache=None, pe_scale=0.1):
    x_used = x
    if arch in ("ecgfm", "ecgfounder", "clef", "hubertecg", "fcn", "bilstm", "ai85tcn1d"):
        x_used = zscore_per_sample_channel(x_used)

    x_in = make_student_input(x_used, arch=arch, pe_cache=pe_cache, pe_scale=pe_scale)
    if arch == "physiowave":
        logits = model(x_in, task="downstream", task_name=args.task_type, return_logits=True)
    else:
        logits = model(x_in)

    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    if isinstance(logits, dict):
        logits = logits.get("logits", None) or logits.get("preds", None) or logits.get("y_hat", None) or next(iter(logits.values()))
    return logits


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
        ce_loss = self.ce(logits, target)

        if self.ignore_index is not None and self.ignore_index >= 0:
            valid = target != self.ignore_index
            logits = logits[valid]
            target = target[valid]
            if target.numel() == 0:
                return ce_loss  # nothing to compute for soft-F1

        probs = F.softmax(logits, dim=1)  # [B, C]
        y = F.one_hot(target, num_classes=self.num_classes).float()  # [B, C]

        tp = (probs * y).sum(dim=0)                 # [C]
        fp = (probs * (1.0 - y)).sum(dim=0)         # [C]
        fn = ((1.0 - probs) * y).sum(dim=0)         # [C]

        f1_per_class = (2.0 * tp) / (2.0 * tp + fp + fn + self.eps)  # [C]
        soft_f1_macro = f1_per_class.mean()

        soft_f1_loss = 1.0 - soft_f1_macro
        total = ce_loss + (self.lam * soft_f1_loss)
        return total
    
def load_physiowave(num_classes: int, in_channels: int = 8):
    rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{rank}")

    head_hidden_dim = 512
    head_dropout = 0.1
    pooling = "mean"
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

    if hasattr(model, 'initialize_weights'):
        model.initialize_weights()

    return model, None


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
        print(f"Output dir: {args.output_dir}")

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

    def _pp_enabled_for_arch(pp_apply_to: str, arch: str) -> bool:
        if pp_apply_to in ("none",):
            return False
        if pp_apply_to == "all":
            return True
        return pp_apply_to == arch
    
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

    teacher = None
    teacher_pe_cache = None
    if (args.alpha_kd > 0) and (not use_cached_teacher):
        teacher, teacher_pe_cache = build_teacher_model(args, train_ds, device, rank, kd_outputs)
    elif use_cached_teacher and rank == 0:
        print(">> Skipping online teacher forward (cached logits will be used).")

    if args.student_arch == "physiowave":
        student, pe_cache = load_physiowave(
            num_classes=train_ds.num_classes,
            in_channels=args.in_channels,
        )
    else:
        student, pe_cache = build_student(args, train_ds, device, rank)

    if args.student_arch in ("clef", "hubertecg", "ecgfm", "ecgfounder"):
        student.eval()
        with torch.no_grad():
            x0, _ = train_ds[0]
            x0 = x0.unsqueeze(0).to(device)  # [1,V,T]
            x0 = zscore_per_sample_channel(x0)
            x_in = make_student_input(x0, arch=args.student_arch, pe_cache=pe_cache, pe_scale=args.student_pe_scale)
            _ = student(x_in)
    
        base = unwrap_ddp(student)
        
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
    
    use_ddp = world_size > 1
    if use_ddp:
        student = DDP(student, device_ids=[rank])

    hard_criterion = build_hard_criterion(args, train_ds, device)

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

    best_metric = None

    if rank == 0 and args.sanity_teacher_test and test_loader is not None:
        print("\n===> SANITY: Evaluating TEACHER on TEST split <===")

        teacher_model, teacher_eval_pe_cache = build_teacher_model(
            args, train_ds, device, rank, kd_outputs
        )

        if args.task_type == "classification":
            teacher_crit = nn.CrossEntropyLoss().to(device)
        else:
            teacher_crit = nn.BCEWithLogitsLoss().to(device)

        _ = eval_unified(
            epoch="TeacherTest",
            rank=rank,
            model=teacher_model,
            loader=test_loader,
            device=device,
            criterion=teacher_crit,
            threshold=args.threshold,
            desc_prefix=f"Teacher({args.teacher_model}) TEST",
            task_type=args.task_type,
            pe_cache=teacher_eval_pe_cache,
            student_arch=args.teacher_model,
            pe_scale=args.teacher_pe_scale,
            amp_enabled=amp_enabled,
            amp_dtype=(torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16),
        )

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

                t_logits = None
                if use_cached_teacher:
                    t_logits = t_logits_cached.to(device, non_blocking=True)
                elif teacher is not None:
                    with torch.inference_mode():
                        with autocast_ctx(enabled=(amp_enabled and device.type == "cuda"), dtype=amp_dtype):
                            t_logits = forward_model_logits(
                                teacher,
                                x_used,
                                arch=args.teacher_model,
                                args=args,
                                pe_cache=teacher_pe_cache,
                                pe_scale=args.teacher_pe_scale,
                            )

                x_in = make_student_input(x_used, arch=args.student_arch, pe_cache=pe_cache, pe_scale=args.student_pe_scale)

                with autocast_ctx(enabled=(amp_enabled and device.type == "cuda"), dtype=amp_dtype):
                    if args.student_arch == "physiowave":
                        pw_task = "multilabel" if args.task_type == "multilabel" else "classification"
                        s_logits = student(
                            x_in,
                            task="downstream",
                            task_name=pw_task,
                            return_logits=(pw_task == "multilabel"),
                        )
                        if isinstance(s_logits, (tuple, list)): s_logits = s_logits[0]
                        assert s_logits.dim() == 2, f"PhysioWave should output [B,C], got {tuple(s_logits.shape)}"
                    else:
                        s_logits = student(x_in)

                    if isinstance(s_logits, (tuple, list)):
                        s_logits = s_logits[0]
                    if isinstance(s_logits, dict):
                        s_logits = s_logits.get("logits", None) or s_logits.get("preds", None) or s_logits.get("y_hat", None) or next(iter(s_logits.values()))

                    if s_logits.dim() == 3:
                        s_logits = s_logits.mean(dim=-1)
                    elif s_logits.dim() == 4:
                        s_logits = s_logits.flatten(2).mean(-1)

                    s_logits_f = fp32(s_logits)
                    t_logits_f = fp32(t_logits) if (t_logits is not None) else None

                    if args.task_type == "classification":
                        hard_loss = hard_criterion(s_logits_f, y)   # or your soft-target CE if using mixup
                        soft_loss = kd_ce(s_logits_f, t_logits_f, T=args.temperature) if (t_logits_f is not None) else 0.0
                    else:
                        hard_loss = hard_criterion(s_logits_f, y.float())
                        soft_loss = kd_bce_with_logits(s_logits_f, t_logits_f, T=args.temperature) if (t_logits_f is not None) else 0.0

                    alpha = float(args.alpha_kd)
                    loss = (1.0 - alpha) * hard_loss + (alpha * soft_loss if (t_logits_f is not None) else 0.0)

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
            amp_enabled=amp_enabled, amp_dtype=amp_dtype,
        )
        val_metric, higher_is_better = get_save_metric(val_metrics, args.save_criterion)
        should_save = False
        if best_metric is None:
            should_save = True
        elif higher_is_better:
            should_save = val_metric > best_metric
        else:
            should_save = val_metric < best_metric

        if rank == 0 and should_save:
            best_metric = val_metric
            save_path = os.path.join(args.output_dir, "best_student.pth")

            base_student = unwrap_ddp(student)

            torch.save({
                "epoch": epoch,
                "student_state_dict": base_student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "val_metrics": val_metrics,
                "save_criterion": args.save_criterion,
                "save_metric_value": val_metric,
                "args": vars(args),
            }, save_path)

            print(
                f"Saved best student @ epoch {epoch} -> {save_path} "
                f"({args.save_criterion}={val_metric:.4f})"
            )

    if rank == 0 and test_loader is not None:
        print("\n===> Testing best student <===")
        ckpt = torch.load(os.path.join(args.output_dir, "best_student.pth"), map_location="cpu", weights_only=False)
        unwrap_ddp(student).load_state_dict(ckpt["student_state_dict"])
        test_metrics = eval_unified(
            "Test", rank, student, test_loader, device, hard_criterion,
            threshold=args.threshold, desc_prefix="Test",
            task_type=args.task_type, pe_cache=pe_cache,
            student_arch=args.student_arch, pe_scale=args.student_pe_scale,
            amp_enabled=amp_enabled, amp_dtype=amp_dtype
        )
        with open(os.path.join(args.output_dir, "test_results_kd.json"), "w") as f:
            json.dump(test_metrics, f, indent=4)

    dist.destroy_process_group()

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

    p.add_argument("--train_file", type=str, default="")
    p.add_argument("--val_file",   type=str, default="")
    p.add_argument("--test_file",  type=str, default="")
    p.add_argument("--data_key",   type=str, default="data")
    p.add_argument("--label_key",  type=str, default="label")
    p.add_argument("--max_length", type=int, default=1024)

    p.add_argument("--task_type", type=str, default="classification",
                   choices=["multilabel", "classification"])
    p.add_argument("--threshold", type=float, default=0.5)

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs",     type=int, default=50)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--num_workers",  type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true", default=False,
                   help="Enable deterministic PyTorch execution for more reproducible runs.")
    p.add_argument("--grad_clip", type=float, default=0.0)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--world_size", type=int, default=1)
    p.add_argument("--output_dir", type=str, default="./kd_output")
    p.add_argument("--accum_steps", type=int, default=1)
    p.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "none"])

    p.add_argument("--teacher_checkpoint", type=str, default="")

    p.add_argument("--alpha_kd", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--hard_loss", type=str, default="ce",
                   choices=["ce", "ce_softf1", "bce"],
                   help="Hard-label loss. Use 'ce_softf1' for imbalanced single-label runs such as DB5.")
    p.add_argument("--save_criterion", type=str, default="f1_macro",
                   choices=["loss", "f1_macro", "f1_weighted", "accuracy"],
                   help="Validation metric used to select and save the best checkpoint.")

    p.add_argument("--in_channels", type=int, default=8, help="Number of variates/electrodes V (EPN-612=8).")
    p.add_argument("--domain_name", type=str, default="emg612")

    p.add_argument(
        "--student_arch",
        type=str,
        default="physiowavenpu",
        choices=[
            "physiowavenpu",
            "physiowave",
        ],
    )
    p.add_argument(
        "--teacher_model",
        type=str,
        default="physiowave",
        choices=[
            "physiowave",
            "physiowavenpu", 
        ],
        help="Teacher backbone used for KD logits. 'physiowave' uses the original BERTWavelet teacher loader.",
    )
    p.add_argument(
        "--student_dataset_profile",
        type=str,
        default="auto",
        choices=["none", "auto", "uci", "db5", "epn612", "ptb", "cpsc", "chapman"],
        help="Dataset-tuned PhysioWaveNPU settings. When set, overrides patch/front-pool/post-pool/pos-freqs.",
    )
    p.add_argument("--patch_size", type=int, default=40)
    p.add_argument("--student_embed_dim", type=int, default=256)
    p.add_argument("--student_depth", type=int, default=3)
    p.add_argument("--student_bank_ch", type=int, default=16)
    p.add_argument("--student_reduce", type=int, default=4)
    p.add_argument("--student_pos_freqs", type=int, default=8)
    p.add_argument("--student_front_pool_k", type=int, default=4)
    p.add_argument("--student_post_patch_pool_t", type=int, default=5)
    p.add_argument("--student_pe_scale", type=float, default=0.1)
    p.add_argument("--print_student_config", action="store_true", default=False,
                   help="Print resolved student model config (after profile overrides) before build.")
    p.add_argument(
        "--teacher_dataset_profile",
        type=str,
        default="auto",
        choices=["none", "auto", "uci", "db5", "epn612", "ptb", "cpsc", "chapman"],
        help="Optional dataset profile override for teacher model shape selection (used for physiowave DB5 path).",
    )
    p.add_argument("--teacher_pe_scale", type=float, default=0.1)

    p.add_argument(
        "--student_kernel_set",
        type=str,
        default="3,5,7",
        help="Comma-separated odd kernel sizes for wavelet-like front-end branches. "
            "Examples: '3,5,7' (baseline), '7' (single-branch), '3' (single-branch)."
    )

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

    p.add_argument("--baseline_width", type=int, default=128, help="Base channel/hidden width for baseline models.")
    p.add_argument("--baseline_depth", type=int, default=6, help="Depth for TCN/ConvNeXt1D blocks.")
    p.add_argument("--baseline_dropout", type=float, default=0.1, help="Dropout for baseline models.")
    p.add_argument("--tcn_kernel", type=int, default=3, help="Kernel size for TCN.")
    p.add_argument("--tcn_dilated", action="store_true", default=True, help="Use dilations in TCN blocks.")
    p.add_argument("--bilstm_layers", type=int, default=2, help="Number of BiLSTM layers.")

    p.add_argument("--amp", action="store_true", default=False, help="Enable torch.cuda.amp autocast + GradScaler.")
    p.add_argument("--use_amp", action="store_true", default=False, help="Alias for --amp.")
    p.add_argument("--amp_dtype", type=str, default="fp16", choices=["fp16", "bf16"], help="AMP dtype for autocast.")

    p.add_argument("--softf1_lambda", type=float, default=0.3,
                help="Lambda for --hard_loss ce_softf1.")

    p.add_argument("--teacher_logits_h5", type=str, default="",
                help="Optional HDF5 file containing precomputed teacher logits aligned with train dataset order.")
    p.add_argument("--teacher_logits_key", type=str, default="teacher_logits",
                help="Dataset key inside HDF5 (default: teacher_logits).")
    p.add_argument("--sanity_teacher_test", action="store_true", default=False,
                help="Run a one-off teacher evaluation on the TEST split before training.")
    
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
    if args.alpha_kd > 0 and not args.teacher_checkpoint and not args.teacher_logits_h5:
        p.error("--teacher_checkpoint is required when alpha_kd > 0 and no cached teacher logits are provided.")

    configure_reproducibility(args.seed, deterministic=args.deterministic)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
    main_worker(local_rank, world_size, args)

if __name__ == "__main__":
    main()


