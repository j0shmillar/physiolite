from __future__ import annotations

import h5py
import numpy as np
import torch
from scipy.signal import iirnotch, butter, filtfilt, medfilt
from torch.utils.data import WeightedRandomSampler

from timeseries_ds import collate_multilabel_fn, collate_singlelabel_fn


def collate_singlelabel_with_tlogits(batch):
    xs = [(x, y) for (x, y, _t) in batch]
    x, y = collate_singlelabel_fn(xs)
    t = torch.stack([_t for (_x, _y, _t) in batch], dim=0)
    return x, y, t


def collate_multilabel_with_tlogits(batch):
    xs = [(x, y) for (x, y, _t) in batch]
    x, y = collate_multilabel_fn(xs)
    t = torch.stack([_t for (_x, _y, _t) in batch], dim=0)
    return x, y, t


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
    x = np.asarray(x_ct, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    b, a = iirnotch(notch_freq, notch_q, fs)
    y = np.zeros_like(x)
    for c in range(x.shape[0]):
        y[c] = filtfilt(b, a, x[c])

    b, a = butter(N=band_order, Wn=[band_low, band_high], btype="bandpass", fs=fs)
    for c in range(y.shape[0]):
        y[c] = filtfilt(b, a, y[c])

    k = int(baseline_kernel_sec * fs) + 1
    if k % 2 == 0:
        k += 1
    baseline = np.zeros_like(y)
    for c in range(y.shape[0]):
        baseline[c] = medfilt(y[c], kernel_size=k)
    y = y - baseline

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

        if hasattr(base_ds, "num_classes"):
            self.num_classes = base_ds.num_classes
        if hasattr(base_ds, "num_labels"):
            self.num_labels = base_ds.num_labels

    def __len__(self):
        return len(self.base_ds)

    def __getattr__(self, name):
        if name in ("base_ds", "mode", "fs", "args"):
            raise AttributeError(name)
        return getattr(self.base_ds, name)

    def __getitem__(self, idx):
        x, y = self.base_ds[idx]
        if self.mode == "none":
            return x, y

        x_np = x.detach().cpu().numpy()
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


class TeacherLogitsWrapperDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds, h5_path: str, key: str = "teacher_logits"):
        self.base_ds = base_ds
        self.h5_path = h5_path
        self.key = key

        if hasattr(base_ds, "num_classes"):
            self.num_classes = base_ds.num_classes
        if hasattr(base_ds, "num_labels"):
            self.num_labels = base_ds.num_labels

        with h5py.File(self.h5_path, "r") as f:
            if self.key not in f:
                raise KeyError(f"'{self.key}' not found in {self.h5_path}")
            n = f[self.key].shape[0]
        if n != len(self.base_ds):
            raise ValueError(f"teacher logits rows ({n}) != dataset length ({len(self.base_ds)})")

        self._h5 = None
        self._arr = None

    def __len__(self):
        return len(self.base_ds)

    def __getattr__(self, name):
        if name in ("base_ds", "h5_path", "key", "_h5", "_arr"):
            raise AttributeError(name)
        return getattr(self.base_ds, name)

    def _lazy_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
            self._arr = self._h5[self.key]

    def __getitem__(self, idx):
        x, y = self.base_ds[idx]
        self._lazy_open()
        t = torch.tensor(self._arr[idx], dtype=torch.float16)
        return x, y, t


def make_balanced_sampler_singlelabel(ds, num_classes: int, pow_: float = 1.0):
    y = ds._labels
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    y = np.asarray(y).reshape(-1)

    cls_counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    cls_counts[cls_counts == 0] = 1.0

    inv = 1.0 / cls_counts
    inv = inv ** float(pow_)
    sample_w = inv[y]
    sample_w = torch.as_tensor(sample_w, dtype=torch.double)

    sampler = WeightedRandomSampler(
        weights=sample_w,
        num_samples=len(sample_w),
        replacement=True,
    )
    return sampler, cls_counts.astype(int)
