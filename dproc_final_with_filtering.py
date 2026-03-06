#!/usr/bin/env python3
"""
NinaPro DB5 data processor (leakage-safe, repetition-safe, numerically safe)

End-to-end script that:
- Selects N channels (default 8; first or last armband)
- Applies zero-phase bandpass + notch filtering on raw continuous sEMG
- Prevents normalization leakage by computing per-subject per-channel μ/σ from TRAIN repetitions only
- Windows strictly *within contiguous repetition runs* (no cross-repetition leakage/contamination)
- Standardizes each channel via per-channel z-score normalization
- Hardens normalization to avoid NaN/Inf (std clamp + clip + finite asserts)
- Writes label_map.json (raw -> consecutive)

Requested preprocessing:
  Raw sEMG recordings are first filtered and normalized.
  - zero-phase bandpass (e.g., 20–450 Hz, 4th order)
  - notch (50/60 Hz, Q=30)
  - per-channel z-score normalization

NOTE: if fs is too low for band_high (e.g. DB5 fs=200 -> Nyquist=100),
      band_high is clamped to 0.95*Nyquist automatically.
"""

import os
import json
import argparse
from collections import Counter

import h5py
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt

# ----------------------------
# Defaults
# ----------------------------
FS_DEFAULT = 200
WSIZE_DEFAULT = 50   # 250 ms @ 200 Hz
STEP_DEFAULT  = 25   # 125 ms @ 200 Hz

TRAIN_REPEATS = [1, 3, 4, 6]
VAL_REPEATS   = [2]
TEST_REPEATS  = [5]

EPS = 1e-8

# Numeric safety knobs
STD_FLOOR = 1e-3     # clamp tiny stds
CLIP_VALUE = 10.0    # clip normalized EMG to [-CLIP_VALUE, +CLIP_VALUE]

# ----------------------------
# IO helpers
# ----------------------------
def save_h5(path, X, y):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=X.astype(np.float32), compression="gzip")
        f.create_dataset("label", data=y.astype(np.int64),   compression="gzip")
    cnt = Counter(y.tolist())
    print(f"Saved: {path}")
    print(f"  data:  {X.shape} (N, C, W)")
    print(f"  label: {y.shape}   classes={len(cnt)} | head={cnt.most_common(10)}")

def sanity_norm_print(name, X):
    if X.size == 0:
        print(f"[{name}] empty")
        return
    mean_all = float(X.mean())
    std_per_ch = X.reshape(X.shape[0], X.shape[1], -1).std(axis=(0, 2))
    finite = bool(np.isfinite(X).all())
    print(f"[{name}] finite={finite} mean(all)≈{mean_all:+.4f} | "
          f"std/ch mean={std_per_ch.mean():.3f} min={std_per_ch.min():.3f} max={std_per_ch.max():.3f} | "
          f"min={float(np.min(X)):.3f} max={float(np.max(X)):.3f}")

# ----------------------------
# Core helpers
# ----------------------------
def contiguous_runs(mask: np.ndarray):
    """
    Given boolean mask length T, yield (start, end) for True-runs, end exclusive.
    """
    mask = mask.astype(bool)
    T = mask.shape[0]
    if T == 0:
        return
    diff = np.diff(mask.astype(np.int8))
    starts = np.where(diff == 1)[0] + 1
    ends   = np.where(diff == -1)[0] + 1
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, T]
    for s, e in zip(starts, ends):
        if e > s:
            yield int(s), int(e)

def bandpass_sos(fs: float, low: float, high: float, order: int):
    nyq = 0.5 * fs
    # clamp to valid digital filter region
    low_c  = max(0.0, float(low))
    high_c = float(high)
    if high_c >= nyq:
        high_c = 0.95 * nyq
    if low_c <= 0.0:
        low_c = 0.01 * nyq  # tiny >0 to avoid invalid
    if high_c <= low_c:
        raise ValueError(f"Invalid band: low={low} high={high} for fs={fs} (after clamp low={low_c}, high={high_c})")

    sos = butter(N=int(order), Wn=[low_c, high_c], btype="bandpass", fs=fs, output="sos")
    return sos, low_c, high_c, nyq

def filter_emg_zero_phase(
    emg_t_c: np.ndarray,
    *,
    fs: float,
    band_low: float,
    band_high: float,
    band_order: int,
    notch_freq: float,
    notch_q: float,
) -> np.ndarray:
    """
    emg_t_c: (T,C) float32/float64
    Returns filtered emg (T,C) float32 with:
      - zero-phase bandpass (sosfiltfilt)
      - zero-phase notch (filtfilt)
    """
    x = np.asarray(emg_t_c, dtype=np.float64)
    # harden input
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    sos, low_eff, high_eff, nyq = bandpass_sos(fs, band_low, band_high, band_order)
    if high_eff != band_high:
        print(f"  [warn] band_high clamped: requested={band_high} Hz, effective={high_eff:.3f} Hz (Nyquist={nyq:.1f} Hz)")

    # bandpass (zero-phase)
    y = sosfiltfilt(sos, x, axis=0)

    # notch (zero-phase)
    # if notch_freq >= Nyquist, skip
    if notch_freq > 0 and notch_freq < nyq:
        b, a = iirnotch(w0=float(notch_freq), Q=float(notch_q), fs=float(fs))
        y = filtfilt(b, a, y, axis=0)
    else:
        print(f"  [warn] notch skipped: notch_freq={notch_freq} Hz not in (0, Nyquist={nyq:.1f})")

    y = y.astype(np.float32)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y

def compute_mu_std_from_segments(segments):
    """
    segments: list of (T_i, C) arrays
    returns mu(C,), std(C,) with std clamped to avoid divide-by-zero
    """
    if not segments:
        raise ValueError("No segments provided to compute normalization stats.")

    C = segments[0].shape[1]
    n = 0
    mean = np.zeros((C,), dtype=np.float64)
    M2 = np.zeros((C,), dtype=np.float64)

    for seg in segments:
        if seg.ndim != 2 or seg.shape[1] != C:
            raise ValueError(f"Bad segment shape for mu/std: {seg.shape}, expected (*,{C})")
        x = seg.astype(np.float64)
        # Welford per-sample over time
        for i in range(x.shape[0]):
            n += 1
            delta = x[i] - mean
            mean += delta / n
            delta2 = x[i] - mean
            M2 += delta * delta2

    var = M2 / max(n - 1, 1)
    std = np.sqrt(var).astype(np.float32)
    mu = mean.astype(np.float32)

    std = np.maximum(std, STD_FLOOR)
    return mu, std

def zscore_apply_per_channel(emg_t_c: np.ndarray, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Per-channel z-score using provided μ/σ (C,). (Leakage-safe if μ/σ computed from TRAIN reps only.)
    """
    x = np.asarray(emg_t_c, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = (x - mu[None, :]) / (std[None, :] + EPS)
    x = np.clip(x, -CLIP_VALUE, CLIP_VALUE).astype(np.float32)
    return x

def segment_within_repetition_majority(
    emg_t_c: np.ndarray,
    labels_t: np.ndarray,
    reps_t: np.ndarray,
    wsize: int,
    step: int,
    min_majority: float = 0.9,
    drop_ambiguous: bool = False,
):
    """
    Segment within contiguous repetition runs only (no cross-repetition windows).
    Label by majority vote within window.

    Returns:
      X: (N, C, W) float32
      y: (N,) int64
      r: (N,) int64
    """
    T, C = emg_t_c.shape
    if T < wsize:
        return (np.empty((0, C, wsize), np.float32),
                np.empty((0,), np.int64),
                np.empty((0,), np.int64))

    Xs, Ys, Rs = [], [], []

    reps_unique = np.unique(reps_t)
    for r_id in reps_unique:
        if r_id <= 0:
            continue
        mask = (reps_t == r_id)
        for rs, re in contiguous_runs(mask):
            seg_len = re - rs
            if seg_len < wsize:
                continue
            for s in range(rs, re - wsize + 1, step):
                e = s + wsize
                seg_lab = labels_t[s:e]
                vals, cnts = np.unique(seg_lab, return_counts=True)
                maj = int(vals[np.argmax(cnts)])
                purity = float(cnts.max()) / float(wsize)
                if drop_ambiguous and purity < min_majority:
                    continue
                Xs.append(emg_t_c[s:e, :].T.astype(np.float32))  # (C,W)
                Ys.append(maj)
                Rs.append(int(r_id))

    if not Xs:
        return (np.empty((0, C, wsize), np.float32),
                np.empty((0,), np.int64),
                np.empty((0,), np.int64))
    return np.stack(Xs, axis=0), np.asarray(Ys, np.int64), np.asarray(Rs, np.int64)

def cap_rest_indices(y, max_rest_ratio):
    """
    Limit class 0 ('rest') to <= max_rest_ratio * #non-rest.
    Deterministic (keep earliest rest).
    """
    if max_rest_ratio < 0:
        return np.ones_like(y, dtype=bool)
    rest = np.where(y == 0)[0]
    gest = np.where(y != 0)[0]
    if len(rest) == 0 or len(gest) == 0:
        return np.ones_like(y, dtype=bool)
    max_rest = int(max_rest_ratio * len(gest))
    if len(rest) <= max_rest:
        return np.ones_like(y, dtype=bool)
    keep = np.zeros_like(y, dtype=bool)
    keep[gest] = True
    keep[rest[:max_rest]] = True
    return keep

# ----------------------------
# DB5 parsing
# ----------------------------
def parse_db5_file(mpath: str):
    """
    Returns:
      emg (T, 16), lab (T,), rep (T,)
    Applies DB5 exercise label offsets for E2/E3.
    """
    d = loadmat(mpath)
    if "emg" not in d or "restimulus" not in d or "rerepetition" not in d:
        raise KeyError("missing one of: emg, restimulus, rerepetition")

    emg = np.asarray(d["emg"], dtype=np.float32)
    if emg.ndim != 2:
        raise ValueError(f"emg must be 2D (T,C); got {emg.shape}")

    lab = np.asarray(d["restimulus"]).squeeze().astype(np.int64)
    rep = np.asarray(d["rerepetition"]).squeeze().astype(np.int64)

    if emg.shape[0] != lab.shape[0] or emg.shape[0] != rep.shape[0]:
        raise ValueError(f"length mismatch emg:{emg.shape} lab:{lab.shape} rep:{rep.shape}")

    fname = os.path.basename(mpath)
    # DB exercise label offsets (zeros preserved)
    if "E2" in fname:
        lab = np.where(lab == 0, 0, lab + 12)
    elif "E3" in fname:
        lab = np.where(lab == 0, 0, lab + 29)

    return emg, lab, rep

def select_n_channels(emg: np.ndarray, mode: str, channels: int) -> np.ndarray:
    C = emg.shape[1]
    if C < channels:
        raise ValueError(f"Need >= {channels} channels, got {C}")
    if mode == "first":
        return emg[:, :channels]
    elif mode == "last":
        return emg[:, -channels:]
    else:
        raise ValueError(f"Unknown channel_mode: {mode}")

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="DB5 EMG formatter (rep-safe windowing + zero-phase filtering + leakage-safe per-channel zscore)")
    ap.add_argument("--input_data", type=str, default="data/ninapro_db5/",
                    help="DB5 root folder containing subject subdirs")
    ap.add_argument("--output_h5", type=str, default="data/processed/db5_h5",
                    help="Output folder for H5 files")
    ap.add_argument("--fs", type=float, default=FS_DEFAULT, help="Sampling rate (DB5=200)")
    ap.add_argument("--window_size", type=int, default=WSIZE_DEFAULT,
                    help="Window size in samples")
    ap.add_argument("--stride", type=int, default=STEP_DEFAULT,
                    help="Stride in samples")
    ap.add_argument("--min_majority", type=float, default=0.9,
                    help="Min dominant-label fraction per window")
    ap.add_argument("--drop_ambiguous", action="store_true",
                    help="Drop windows whose dominant label purity < min_majority")
    ap.add_argument("--max_rest_ratio", type=float, default=-1.0,
                    help="If >=0, cap rest windows to r*#gesture per split (after splitting)")
    ap.add_argument("--split_type", type=str, default="repetition",
                    choices=["repetition", "random"],
                    help="Repetition split or random stratified split")
    ap.add_argument("--random_val", type=float, default=0.1)
    ap.add_argument("--random_test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--channels", type=int, default=8, help="Channels to keep (default 8)")
    ap.add_argument("--channel_mode", type=str, default="first", choices=["first", "last"],
                    help="Select first N or last N channels (armband choice)")

    # --- Filtering knobs (requested preprocessing) ---
    ap.add_argument("--band_low", type=float, default=20.0, help="Bandpass low cutoff (Hz)")
    ap.add_argument("--band_high", type=float, default=90.0, help="Bandpass high cutoff (Hz)")
    ap.add_argument("--band_order", type=int, default=4, help="Bandpass Butterworth order")
    ap.add_argument("--notch_freq", type=float, default=50.0, help="Notch frequency (Hz) - use 50 or 60")
    ap.add_argument("--notch_q", type=float, default=30.0, help="Notch quality factor Q")

    args = ap.parse_args()
    np.random.seed(args.seed)
    os.makedirs(args.output_h5, exist_ok=True)

    subjects = [d for d in sorted(os.listdir(args.input_data))
                if os.path.isdir(os.path.join(args.input_data, d))]
    if not subjects:
        print(f"No subject folders under {args.input_data}")
        return

    print(f"Found {len(subjects)} subject(s)")
    print(f"FS={args.fs} Hz | window={args.window_size} ({args.window_size/args.fs:.3f}s)"
          f" | stride={args.stride} ({args.stride/args.fs:.3f}s)")
    print(f"Channels: {args.channels} mode={args.channel_mode}")
    print(f"Filtering: bandpass {args.band_low}-{args.band_high} Hz (order={args.band_order}), "
          f"notch {args.notch_freq} Hz (Q={args.notch_q})")
    if args.split_type == "repetition":
        print("Repetition split: train {1,3,4,6}, val {2}, test {5}")

    # Aggregate across all subjects (then split)
    X_all, y_all, r_all = [], [], []
    label_values_seen = set()

    for subj in subjects:
        subj_path = os.path.join(args.input_data, subj)
        mat_files = [f for f in sorted(os.listdir(subj_path)) if f.endswith(".mat")]
        if not mat_files:
            print(f"  [{subj}] no .mat files; skip")
            continue

        files_data = []
        print(f"Subject {subj}: {len(mat_files)} files")
        for mf in mat_files:
            mpath = os.path.join(subj_path, mf)
            try:
                emg, lab, rep = parse_db5_file(mpath)
                emg = select_n_channels(emg, args.channel_mode, args.channels)
                files_data.append((mf, emg, lab, rep))
            except Exception as e:
                print(f"  ! {mf}: {e}")
                continue

        if not files_data:
            print(f"  [{subj}] no usable files; skip")
            continue

        # 1) Filter all files (continuous) first (zero-phase bandpass + notch)
        files_filt = []
        for (mf, emg, lab, rep) in files_data:
            emg_f = filter_emg_zero_phase(
                emg,
                fs=float(args.fs),
                band_low=float(args.band_low),
                band_high=float(args.band_high),
                band_order=int(args.band_order),
                notch_freq=float(args.notch_freq),
                notch_q=float(args.notch_q),
            )
            if not np.isfinite(emg_f).all():
                raise ValueError(f"Non-finite values after filtering in {subj}/{mf}")
            files_filt.append((mf, emg_f, lab, rep))

        # 2) Compute per-subject per-channel μ/σ using TRAIN repetitions only (leakage-safe)
        train_segments = []
        for (mf, emg_f, lab, rep) in files_filt:
            mask_train = np.isin(rep, TRAIN_REPEATS)
            for s, e in contiguous_runs(mask_train):
                seg = emg_f[s:e, :]
                if seg.shape[0] > 0:
                    train_segments.append(seg)

        if not train_segments:
            print(f"  [{subj}] no TRAIN repetition samples found; skip subject")
            continue

        mu, std = compute_mu_std_from_segments(train_segments)

        # 3) Normalize + window within repetitions
        Xs_subj, ys_subj, rs_subj = [], [], []
        for (mf, emg_f, lab, rep) in files_filt:
            emg_n = zscore_apply_per_channel(emg_f, mu, std)
            if not np.isfinite(emg_n).all():
                raise ValueError(f"Non-finite values after normalization in {subj}/{mf}")

            X, y, r = segment_within_repetition_majority(
                emg_t_c=emg_n,
                labels_t=lab,
                reps_t=rep,
                wsize=args.window_size,
                step=args.stride,
                min_majority=args.min_majority,
                drop_ambiguous=args.drop_ambiguous,
            )
            if X.shape[0] == 0:
                continue
            Xs_subj.append(X); ys_subj.append(y); rs_subj.append(r)
            label_values_seen.update(np.unique(y).tolist())

        if not Xs_subj:
            print(f"  [{subj}] no windows collected; skip subject")
            continue

        Xs_subj = np.concatenate(Xs_subj, axis=0)
        ys_subj = np.concatenate(ys_subj, axis=0)
        rs_subj = np.concatenate(rs_subj, axis=0)

        if not np.isfinite(Xs_subj).all():
            raise ValueError(f"Non-finite windows produced for subject {subj}")

        X_all.append(Xs_subj)
        y_all.append(ys_subj)
        r_all.append(rs_subj)

    if not X_all:
        print("No windows collected; abort.")
        return

    X_all = np.concatenate(X_all, axis=0)  # (N, C, W)
    y_all = np.concatenate(y_all, axis=0)  # (N,)
    r_all = np.concatenate(r_all, axis=0)  # (N,)

    # Global remap labels to consecutive 0..K-1 (stable)
    uniq = sorted({int(v) for v in set(y_all.tolist())})
    old2new = {int(o): i for i, o in enumerate(uniq)}
    new2old = {i: int(o) for i, o in enumerate(uniq)}
    y_all = np.vectorize(old2new.get, otypes=[np.int64])(y_all)

    with open(os.path.join(args.output_h5, "label_map.json"), "w") as f:
        json.dump({"old2new": old2new, "new2old": new2old}, f, indent=2)
    print(f"Saved label_map.json → {os.path.join(args.output_h5, 'label_map.json')}")
    print(f"Raw labels: {uniq}")
    print(f"Remap: {old2new}")

    # ----------------------------
    # Build splits
    # ----------------------------
    splits = {"train": {"X": None, "y": None},
              "val":   {"X": None, "y": None},
              "test":  {"X": None, "y": None}}

    if args.split_type == "repetition":
        for name, rep_set in (("train", TRAIN_REPEATS), ("val", VAL_REPEATS), ("test", TEST_REPEATS)):
            m = np.isin(r_all, rep_set)
            splits[name]["X"] = X_all[m]
            splits[name]["y"] = y_all[m]
    else:
        test_frac = float(args.random_test)
        val_frac  = float(args.random_val)
        assert 0.0 < test_frac < 1.0 and 0.0 < val_frac < 1.0 and test_frac + val_frac < 1.0
        N = y_all.shape[0]
        idx = np.arange(N)

        idx_rest, idx_test = train_test_split(
            idx, test_size=test_frac, random_state=args.seed, stratify=y_all
        )
        y_rest = y_all[idx_rest]
        val_size_rel = val_frac / (1.0 - test_frac)
        idx_train, idx_val = train_test_split(
            idx_rest, test_size=val_size_rel, random_state=args.seed, stratify=y_rest
        )

        splits["train"]["X"] = X_all[idx_train]; splits["train"]["y"] = y_all[idx_train]
        splits["val"]["X"]   = X_all[idx_val];   splits["val"]["y"]   = y_all[idx_val]
        splits["test"]["X"]  = X_all[idx_test];  splits["test"]["y"]  = y_all[idx_test]

    # Optionally cap rest ratio per split (post-window, safe)
    if args.max_rest_ratio >= 0:
        for s in ("train",):
            Xs, ys = splits[s]["X"], splits[s]["y"]
            if ys is None or ys.size == 0:
                continue
            keep = cap_rest_indices(ys, args.max_rest_ratio)
            splits[s]["X"] = Xs[keep]
            splits[s]["y"] = ys[keep]

    # Save + diagnostics
    for s in ("train", "val", "test"):
        Xs, ys = splits[s]["X"], splits[s]["y"]
        if ys is None or ys.size == 0:
            print(f"[{s}] empty; skip save.")
            continue

        cnt = Counter(ys.tolist())
        print(f"[{s}] label distribution head: {cnt.most_common(10)} | classes={len(cnt)}")

        sanity_norm_print(s, Xs)
        out_path = os.path.join(args.output_h5, f"db5_{s}_set.h5")
        save_h5(out_path, Xs, ys)

    print("\nDone.")

if __name__ == "__main__":
    main()
