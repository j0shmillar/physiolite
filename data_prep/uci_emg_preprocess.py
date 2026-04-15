#!/usr/bin/env python3
"""Preprocess UCI EMG for Gestures into HDF5 splits.

This script preserves the exact data parsing, windowing, normalization, label
formatting, and HDF5 output behavior of the known-good `uci_preprocess.py`
that reproduces UCI_EMG_processed_v2 exactly when filtering is disabled and
the matching arguments are used.

It additionally supports optional EMG filtering before window extraction and
normalization.

Exact-reproduction example (matches UCI_EMG_processed_v2):
python data_prep/uci_emg_preprocess.py \
  --root_dir datasets/EMG_data_for_gestures-master \
  --out_dir /tmp/UCI_EMG_processed_repro \
  --seq_len 600 \
  --min_len 300 \
  --edge_trim 120 \
  --stride 0 \
  --zscore_per_subject \
  --val_subjects "33-34" \
  --test_subjects "35-36" \
  --save_subject_ids

Optional filtering example:
python data_prep/uci_emg_preprocess.py \
  --root_dir datasets/EMG_data_for_gestures-master \
  --out_dir /tmp/UCI_EMG_processed_filtered \
  --seq_len 600 \
  --min_len 300 \
  --edge_trim 120 \
  --stride 0 \
  --zscore_per_subject \
  --enable_filtering \
  --fs 200 \
  --band_low 20 \
  --band_high 95 \
  --band_order 4 \
  --notch_freq 50 \
  --notch_q 30 \
  --val_subjects "33-34" \
  --test_subjects "35-36" \
  --save_subject_ids
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt
from tqdm.auto import tqdm

# Keep six core gestures (drop 0=unmarked, 7=not always present)
KEEP_LABELS = {1, 2, 3, 4, 5, 6}
LABEL_REMAP = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
N_CH = 8  # MYO has 8 sensors (columns 2..9 in files)


# ----------------------------
# I/O helpers
# ----------------------------
def has_header(p: Path) -> bool:
    """Return True if first row is non-numeric (e.g., 'time channel1 ...')."""
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        first = f.readline().strip().split()
    if not first:
        return False
    try:
        float(first[0])
        return False
    except Exception:
        return True


def load_txt_file(p: Path) -> np.ndarray:
    """
    Load a subject TXT file into float32 array of shape [T, 10]:
      col0=time(ms), col1..col8=8 EMG channels, col9=class label

    This intentionally preserves the exact parsing behavior of the known-good
    script that reproduces UCI_EMG_processed_v2.
    """
    skip = 1 if has_header(p) else 0

    # Try whitespace first
    try:
        arr = np.loadtxt(str(p), dtype=np.float32, skiprows=skip)
    except Exception:
        # Fallback to comma-delimited
        arr = np.loadtxt(str(p), dtype=np.float32, skiprows=skip, delimiter=",")

    if arr.ndim == 1 and arr.size == 10:
        arr = arr[None, :]
    if arr.shape[1] != 10:
        raise ValueError(f"Expected 10 columns in {p}, got shape {arr.shape}")
    return arr


def read_subject_txts(subject_dir: Path) -> np.ndarray:
    """Concatenate all recordings in a subject directory along time axis: [T_total, 10]."""
    txts = sorted(subject_dir.glob("*.txt"))
    if not txts:
        raise FileNotFoundError(f"No .txt files found in {subject_dir}")
    parts = [load_txt_file(p) for p in txts]
    return np.concatenate(parts, axis=0)


# ----------------------------
# Preprocessing helpers
# ----------------------------
def maxabs_normalize(x: np.ndarray) -> np.ndarray:
    """Per-channel max-abs normalization. x: [C, L]."""
    m = np.max(np.abs(x), axis=1, keepdims=True)
    m[m == 0] = 1.0
    return x / m


def zscore_per_channel(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Per-channel z-score using provided stats. x: [C, L], mean/std: [C, 1]."""
    return (x - mean) / (std + 1e-6)


def bandpass_sos(fs: float, low: float, high: float, order: int):
    """Construct bandpass SOS filter, clamping invalid cutoffs safely."""
    nyq = 0.5 * fs
    low_c = float(low)
    high_c = float(high)

    if high_c >= nyq:
        high_c = 0.95 * nyq
    if low_c <= 0.0:
        low_c = 0.01 * nyq
    if high_c <= low_c:
        raise ValueError(
            f"Invalid band after clamp: low={low_c:.3f}, high={high_c:.3f}, fs={fs}, nyq={nyq}"
        )

    sos = butter(
        N=int(order),
        Wn=[low_c, high_c],
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    return sos, low_c, high_c, nyq


def filter_emg_zero_phase_ct(
    x_ct: np.ndarray,
    *,
    fs: float,
    band_low: float,
    band_high: float,
    band_order: int,
    notch_freq: float,
    notch_q: float,
    warn_prefix: str = "",
) -> np.ndarray:
    """
    Zero-phase filter EMG signal shaped [C, T].

    Filtering is optional and only applied when --enable_filtering is set.
    """
    x = np.asarray(x_ct, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    sos, _low_eff, high_eff, nyq = bandpass_sos(fs, band_low, band_high, band_order)
    if abs(high_eff - band_high) > 1e-6:
        print(
            f"{warn_prefix}[warn] band_high clamped: requested={band_high} Hz, "
            f"effective={high_eff:.3f} Hz (Nyq={nyq:.1f})"
        )

    y = sosfiltfilt(sos, x, axis=1)

    if 0 < notch_freq < nyq:
        b, a = iirnotch(w0=float(notch_freq), Q=float(notch_q), fs=float(fs))
        y = filtfilt(b, a, y, axis=1)
    else:
        print(
            f"{warn_prefix}[warn] notch skipped: "
            f"notch_freq={notch_freq} not in (0, Nyq={nyq:.1f})"
        )

    y = y.astype(np.float32)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y


def extract_runs(labels: np.ndarray) -> List[Tuple[int, int, int]]:
    """
    Get contiguous (start, end, label) runs from 1D labels array. end is exclusive.
    """
    runs: List[Tuple[int, int, int]] = []
    if labels.size == 0:
        return runs
    start = 0
    cur = labels[0]
    for i in range(1, labels.size):
        if labels[i] != cur:
            runs.append((start, i, int(cur)))
            start = i
            cur = labels[i]
    runs.append((start, labels.size, int(cur)))
    return runs


def windows_from_run(
    emg: np.ndarray,
    start: int,
    end: int,
    label: int,
    seq_len: int,
    edge_trim: int,
    min_len: int,
    stride: Optional[int] = None,
    center_pure: bool = True,
) -> List[Tuple[np.ndarray, int]]:
    """
    From a single (start,end,label) run, trim edges, enforce min_len,
    and cut into windows. Returns list of (window[C,L], label).
    """
    if label not in KEEP_LABELS:
        return []

    # trim edges
    s = start + edge_trim
    e = end - edge_trim
    if e <= s:
        return []
    L = e - s
    if L < min_len:
        return []

    # stride default: non-overlapping
    if stride is None or stride <= 0:
        stride = seq_len

    out: List[Tuple[np.ndarray, int]] = []

    # Non-overlapping AND centered packing (avoids boundary contamination)
    if stride == seq_len and center_pure:
        n_win = 0 if seq_len <= 0 else (L // seq_len)
        if n_win <= 0:
            return out
        offset = s + (L - n_win * seq_len) // 2
        for k in range(n_win):
            a = offset + k * seq_len
            b = a + seq_len
            out.append((emg[:, a:b], LABEL_REMAP[label]))
        return out

    # Generic sliding windows
    for a in range(s, e - seq_len + 1, stride):
        b = a + seq_len
        out.append((emg[:, a:b], LABEL_REMAP[label]))
    return out


def process_subject(
    subject_dir: Path,
    seq_len: int,
    edge_trim: int,
    min_len: int,
    use_zscore_subject: bool = False,
    stride: Optional[int] = None,
    enable_filtering: bool = False,
    fs: float = 200.0,
    band_low: float = 20.0,
    band_high: float = 190.0,
    band_order: int = 4,
    notch_freq: float = 50.0,
    notch_q: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (X, y) where
      X: [N, C=8, L=seq_len] float32
      y: [N] int64 (0..5)

    Important:
    - When filtering is OFF, this preserves the exact normalization and data
      formatting behavior of the known-good `uci_preprocess.py`.
    - Subject-wise z-score stats are computed BEFORE windowing on the signal
      actually used for window extraction:
        * raw EMG if filtering is OFF
        * filtered EMG if filtering is ON
    """
    raw = read_subject_txts(subject_dir)  # [T, 10]
    emg = raw[:, 1 : 1 + N_CH].T.astype(np.float32)  # [C=8, T]
    labels = raw[:, 9].astype(np.int64)  # [T]

    if enable_filtering:
        emg_proc = filter_emg_zero_phase_ct(
            emg,
            fs=fs,
            band_low=band_low,
            band_high=band_high,
            band_order=band_order,
            notch_freq=notch_freq,
            notch_q=notch_q,
            warn_prefix=f"[{subject_dir.name}] ",
        )
    else:
        emg_proc = emg

    # Subject-wise stats BEFORE windowing
    if use_zscore_subject:
        ch_mean = emg_proc.mean(axis=1, keepdims=True)  # [C,1]
        ch_std = emg_proc.std(axis=1, keepdims=True)
    else:
        ch_mean = None
        ch_std = None

    # Build windows
    runs = extract_runs(labels)
    windows: List[Tuple[np.ndarray, int]] = []
    for s, e, lab in runs:
        windows.extend(
            windows_from_run(
                emg_proc,
                start=s,
                end=e,
                label=lab,
                seq_len=seq_len,
                edge_trim=edge_trim,
                min_len=min_len,
                stride=stride,
                center_pure=True,
            )
        )

    if not windows:
        return (
            np.zeros((0, N_CH, seq_len), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    # Normalize windows
    if use_zscore_subject:
        X = np.stack(
            [zscore_per_channel(w, ch_mean, ch_std) for (w, _) in windows]
        ).astype(np.float32)
    else:
        X = np.stack([maxabs_normalize(w) for (w, _) in windows]).astype(np.float32)

    y = np.array([lab for (_, lab) in windows], dtype=np.int64)
    return X, y


# ----------------------------
# H5 writer
# ----------------------------
def write_h5(path: Path, X: np.ndarray, y: np.ndarray, subjects: Optional[np.ndarray]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=X, compression="gzip")
        f.create_dataset("label", data=y, compression="gzip")
        if subjects is not None:
            f.create_dataset("subject", data=subjects, compression="gzip")


# ----------------------------
# CLI / main
# ----------------------------
def parse_split_arg(arg: str) -> List[int]:
    """
    Parse a split argument like "33-34" or "33,34" or "33-36,20"
    Returns sorted unique ints.
    """
    result = set()
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            a, b = int(a), int(b)
            lo, hi = min(a, b), max(a, b)
            result.update(range(lo, hi + 1))
        else:
            result.add(int(part))
    return sorted(result)


def main():
    ap = argparse.ArgumentParser(
        description="Preprocess UCI EMG for Gestures -> HDF5 (labels 0..5)"
    )
    ap.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Path to EMG_data_for_gestures-master (contains 01..36)",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for H5 files",
    )
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument(
        "--min_len",
        type=int,
        default=200,
        help="Minimum usable run length AFTER trimming",
    )
    ap.add_argument(
        "--edge_trim",
        type=int,
        default=20,
        help="Trim this many samples from both ends of a run",
    )
    ap.add_argument(
        "--stride",
        type=int,
        default=0,
        help="Window stride (0 or <=0 means non-overlapping; we also center the windows).",
    )
    ap.add_argument(
        "--zscore_per_subject",
        action="store_true",
        help="Use subject-wise per-channel z-score (recommended for cross-subject splits).",
    )

    # Optional filtering
    ap.add_argument(
        "--enable_filtering",
        action="store_true",
        help="Enable bandpass+notch filtering before windowing and normalization.",
    )
    ap.add_argument("--fs", type=float, default=200.0, help="Sampling rate for filtering.")
    ap.add_argument("--band_low", type=float, default=20.0, help="Bandpass low cutoff (Hz).")
    ap.add_argument("--band_high", type=float, default=190.0, help="Bandpass high cutoff (Hz).")
    ap.add_argument("--band_order", type=int, default=4, help="Bandpass Butterworth order.")
    ap.add_argument("--notch_freq", type=float, default=50.0, help="Notch frequency (Hz).")
    ap.add_argument("--notch_q", type=float, default=30.0, help="Notch filter Q.")

    ap.add_argument(
        "--val_subjects",
        type=str,
        default="33-34",
        help='Subjects for validation, e.g. "33-34" or "10,12,14"',
    )
    ap.add_argument(
        "--test_subjects",
        type=str,
        default="35-36",
        help='Subjects for test, e.g. "35-36" or "20,22"',
    )
    ap.add_argument("--save_subject_ids", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)

    root = Path(args.root_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    # Collect subjects (folders named 01..36)
    subs = sorted(
        [int(p.name) for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
    )
    if not subs:
        raise RuntimeError(f"No subject folders like '01', '02', ... found under {root}")

    print("==== UCI EMG for Gestures preprocessing ====")
    print(f"Root: {root}")
    print(f"Subjects found: {len(subs)}  -> {subs[:5]} ... {subs[-5:]}")
    print(
        f"seq_len={args.seq_len}, min_len={args.min_len}, "
        f"edge_trim={args.edge_trim}, stride={args.stride}"
    )
    print("Keeping labels:", sorted(KEEP_LABELS), "(0/7 discarded)")
    if args.zscore_per_subject:
        print("Normalization: subject-wise per-channel z-score")
    else:
        print("Normalization: per-window max-abs")
    print("Label remap:", ", ".join([f"{k}->{v}" for k, v in LABEL_REMAP.items()]))

    if args.enable_filtering:
        print(
            "Filtering: ON "
            f"(bandpass {args.band_low}-{args.band_high} Hz, "
            f"order={args.band_order}; notch {args.notch_freq} Hz, Q={args.notch_q})"
        )
    else:
        print("Filtering: OFF")

    val_subs = parse_split_arg(args.val_subjects)
    test_subs = parse_split_arg(args.test_subjects)
    train_subs = [s for s in subs if s not in set(val_subs) | set(test_subs)]

    print(f"Train subjects: {train_subs}")
    print(f"Val subjects:   {val_subs}")
    print(f"Test subjects:  {test_subs}")

    def collect(for_subjects: List[int]):
        X_all, y_all, sid_all = [], [], []
        for s in tqdm(for_subjects, desc="Subjects"):
            s_dir = root / f"{s:02d}"
            if not s_dir.exists():
                print(f"[WARN] Missing subject dir {s_dir}, skipping.")
                continue
            try:
                X, y = process_subject(
                    s_dir,
                    seq_len=args.seq_len,
                    edge_trim=args.edge_trim,
                    min_len=args.min_len,
                    use_zscore_subject=args.zscore_per_subject,
                    stride=args.stride,
                    enable_filtering=args.enable_filtering,
                    fs=args.fs,
                    band_low=args.band_low,
                    band_high=args.band_high,
                    band_order=args.band_order,
                    notch_freq=args.notch_freq,
                    notch_q=args.notch_q,
                )
            except Exception as e:
                print(f"[WARN] Failed subject {s:02d}: {e}")
                continue
            if X.size == 0:
                continue
            X_all.append(X)
            y_all.append(y)
            if args.save_subject_ids:
                sid_all.append(np.full((X.shape[0],), s, dtype=np.int32))

        if not X_all:
            return (
                np.zeros((0, N_CH, args.seq_len), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
                None if not args.save_subject_ids else np.zeros((0,), dtype=np.int32),
            )

        X_cat = np.concatenate(X_all, axis=0)
        y_cat = np.concatenate(y_all, axis=0)
        sid_cat = None
        if args.save_subject_ids:
            sid_cat = np.concatenate(sid_all, axis=0)
        return X_cat, y_cat, sid_cat

    X_tr, y_tr, sid_tr = collect(train_subs)
    X_va, y_va, sid_va = collect(val_subs)
    X_te, y_te, sid_te = collect(test_subs)

    def counts(y):
        if y.size == 0:
            return {i: 0 for i in range(6)}
        uniq, cnt = np.unique(y, return_counts=True)
        d = {int(u): int(c) for u, c in zip(uniq, cnt)}
        for k in range(6):
            d.setdefault(k, 0)
        return d

    print("\nSummary after preprocessing (labels are 0..5):")
    print(f"Train: X={X_tr.shape}, y={y_tr.shape}, per-class={counts(y_tr)}")
    print(f"Val:   X={X_va.shape}, y={y_va.shape}, per-class={counts(y_va)}")
    print(f"Test:  X={X_te.shape}, y={y_te.shape}, per-class={counts(y_te)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    write_h5(out_dir / "uci_emg_train.h5", X_tr, y_tr, sid_tr)
    write_h5(out_dir / "uci_emg_val.h5", X_va, y_va, sid_va)
    write_h5(out_dir / "uci_emg_test.h5", X_te, y_te, sid_te)

    print(f"\nWrote H5s to: {out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
