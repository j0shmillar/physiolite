#!/usr/bin/env python3
"""Preprocess NinaPro DB5 into train/val/test HDF5 files."""

import argparse
import json
import os
from collections import Counter

import h5py
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt
from sklearn.model_selection import train_test_split


FS_DEFAULT = 200
WSIZE_DEFAULT = 50
STEP_DEFAULT = 25

TRAIN_REPEATS = [1, 3, 4, 6]
VAL_REPEATS = [2]
TEST_REPEATS = [5]

EPS = 1e-8
STD_FLOOR = 1e-3
CLIP_VALUE = 10.0


def save_h5(path, X, y):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=X.astype(np.float32), compression="gzip")
        f.create_dataset("label", data=y.astype(np.int64), compression="gzip")
    cnt = Counter(y.tolist())
    print(f"Saved: {path}")
    print(f"  data:  {X.shape} (N, C, W)")
    print(f"  label: {y.shape} classes={len(cnt)} | head={cnt.most_common(10)}")


def sanity_norm_print(name, X):
    if X.size == 0:
        print(f"[{name}] empty")
        return
    mean_all = float(X.mean())
    std_per_ch = X.reshape(X.shape[0], X.shape[1], -1).std(axis=(0, 2))
    finite = bool(np.isfinite(X).all())
    print(
        f"[{name}] finite={finite} mean(all)≈{mean_all:+.4f} | "
        f"std/ch mean={std_per_ch.mean():.3f} min={std_per_ch.min():.3f} max={std_per_ch.max():.3f} | "
        f"min={float(np.min(X)):.3f} max={float(np.max(X)):.3f}"
    )


def contiguous_runs(mask: np.ndarray):
    mask = mask.astype(bool)
    if mask.size == 0:
        return
    diff = np.diff(mask.astype(np.int8))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, mask.shape[0]]
    for s, e in zip(starts, ends):
        if e > s:
            yield int(s), int(e)


def bandpass_sos(fs: float, low: float, high: float, order: int):
    nyq = 0.5 * fs
    low_c = max(0.0, float(low))
    high_c = float(high)
    if high_c >= nyq:
        high_c = 0.95 * nyq
    if low_c <= 0.0:
        low_c = 0.01 * nyq
    if high_c <= low_c:
        raise ValueError(f"Invalid band after clamp: low={low_c:.3f}, high={high_c:.3f}, fs={fs}")
    sos = butter(N=int(order), Wn=[low_c, high_c], btype="bandpass", fs=fs, output="sos")
    return sos, high_c, nyq


def filter_emg_zero_phase(
    signal: np.ndarray,
    *,
    fs: float,
    band_low: float,
    band_high: float,
    band_order: int,
    notch_freq: float,
    notch_q: float,
) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    sos, high_eff, nyq = bandpass_sos(fs, band_low, band_high, band_order)
    if abs(high_eff - band_high) > 1e-6:
        print(
            f"  [warn] band_high clamped: requested={band_high} Hz, "
            f"effective={high_eff:.3f} Hz (Nyquist={nyq:.1f} Hz)"
        )
    y = sosfiltfilt(sos, x, axis=0)

    if 0 < notch_freq < nyq:
        b, a = iirnotch(w0=float(notch_freq), Q=float(notch_q), fs=float(fs))
        y = filtfilt(b, a, y, axis=0)
    else:
        print(f"  [warn] notch skipped: notch_freq={notch_freq} not in (0, Nyquist={nyq:.1f})")

    y = np.nan_to_num(y.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return y


def compute_mu_std_from_segments(segments):
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
        for i in range(x.shape[0]):
            n += 1
            delta = x[i] - mean
            mean += delta / n
            delta2 = x[i] - mean
            M2 += delta * delta2

    var = M2 / max(n - 1, 1)
    std = np.maximum(np.sqrt(var).astype(np.float32), STD_FLOOR)
    mu = mean.astype(np.float32)
    return mu, std


def normalize_zscore(signal: np.ndarray, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
    out = (signal - mu[None, :]) / (std[None, :] + EPS)
    out = np.clip(out, -CLIP_VALUE, CLIP_VALUE).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def normalize_maxabs(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float32)
    scale = np.max(np.abs(x), axis=0, keepdims=True)
    scale[scale == 0] = 1.0
    out = x / scale
    return np.nan_to_num(out.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def segment_within_repetition_majority(
    emg_t_c: np.ndarray,
    labels_t: np.ndarray,
    reps_t: np.ndarray,
    wsize: int,
    step: int,
    min_majority: float = 0.9,
    drop_ambiguous: bool = False,
):
    T, C = emg_t_c.shape
    if T < wsize:
        return (
            np.empty((0, C, wsize), np.float32),
            np.empty((0,), np.int64),
            np.empty((0,), np.int64),
        )

    Xs, Ys, Rs = [], [], []
    for r_id in np.unique(reps_t):
        if r_id <= 0:
            continue
        mask = reps_t == r_id
        for rs, re in contiguous_runs(mask):
            if re - rs < wsize:
                continue
            for s in range(rs, re - wsize + 1, step):
                e = s + wsize
                seg_lab = labels_t[s:e]
                vals, cnts = np.unique(seg_lab, return_counts=True)
                maj = int(vals[np.argmax(cnts)])
                purity = float(cnts.max()) / float(wsize)
                if drop_ambiguous and purity < min_majority:
                    continue
                Xs.append(emg_t_c[s:e, :].T.astype(np.float32))
                Ys.append(maj)
                Rs.append(int(r_id))

    if not Xs:
        return (
            np.empty((0, C, wsize), np.float32),
            np.empty((0,), np.int64),
            np.empty((0,), np.int64),
        )
    return np.stack(Xs, axis=0), np.asarray(Ys, np.int64), np.asarray(Rs, np.int64)


def cap_rest_indices(y, max_rest_ratio):
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


def parse_db5_file(path: str):
    d = loadmat(path)
    if "emg" not in d or "restimulus" not in d or "rerepetition" not in d:
        raise KeyError("missing one of: emg, restimulus, rerepetition")

    emg = np.asarray(d["emg"], dtype=np.float32)
    lab = np.asarray(d["restimulus"]).squeeze().astype(np.int64)
    rep = np.asarray(d["rerepetition"]).squeeze().astype(np.int64)

    if emg.ndim != 2:
        raise ValueError(f"emg must be 2D (T,C); got {emg.shape}")
    if emg.shape[0] != lab.shape[0] or emg.shape[0] != rep.shape[0]:
        raise ValueError(f"length mismatch emg:{emg.shape} lab:{lab.shape} rep:{rep.shape}")

    name = os.path.basename(path)
    if "E2" in name:
        lab = np.where(lab == 0, 0, lab + 12)
    elif "E3" in name:
        lab = np.where(lab == 0, 0, lab + 29)
    return emg, lab, rep


def select_n_channels(emg: np.ndarray, mode: str, channels: int = 8) -> np.ndarray:
    if emg.shape[1] < channels:
        raise ValueError(f"Need >= {channels} channels, got {emg.shape[1]}")
    if mode == "first":
        return emg[:, :channels]
    if mode == "last":
        return emg[:, -channels:]
    raise ValueError(f"Unknown channel_mode: {mode}")


def collect_subject_mat_files(subject_dir: str):
    mat_files = []
    for root, _dirs, files in os.walk(subject_dir):
        for name in sorted(files):
            if name.endswith(".mat"):
                mat_files.append(os.path.join(root, name))
    return mat_files


def main():
    ap = argparse.ArgumentParser(description="Preprocess NinaPro DB5 into HDF5 splits.")
    ap.add_argument("--input_data", type=str, default="data/ninapro_db5/")
    ap.add_argument("--output_h5", type=str, default="data/processed/db5_h5")
    ap.add_argument("--fs", type=float, default=FS_DEFAULT)
    ap.add_argument("--window_size", type=int, default=WSIZE_DEFAULT)
    ap.add_argument("--stride", type=int, default=STEP_DEFAULT)
    ap.add_argument("--min_majority", type=float, default=0.9)
    ap.add_argument("--drop_ambiguous", action="store_true")
    ap.add_argument("--max_rest_ratio", type=float, default=-1.0)
    ap.add_argument("--split_type", type=str, default="repetition", choices=["repetition", "random"])
    ap.add_argument("--random_val", type=float, default=0.1)
    ap.add_argument("--random_test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--channels", type=int, default=8)
    ap.add_argument("--channel_mode", type=str, default="first", choices=["first", "last"])
    ap.add_argument("--normalization", type=str, default="zscore", choices=["zscore", "maxabs"])
    ap.add_argument("--enable_filtering", action="store_true")
    ap.add_argument("--band_low", type=float, default=20.0)
    ap.add_argument("--band_high", type=float, default=95.0)
    ap.add_argument("--band_order", type=int, default=4)
    ap.add_argument("--notch_freq", type=float, default=50.0)
    ap.add_argument("--notch_q", type=float, default=30.0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.output_h5, exist_ok=True)

    subjects = [
        d for d in sorted(os.listdir(args.input_data))
        if os.path.isdir(os.path.join(args.input_data, d))
    ]
    if not subjects:
        print(f"No subject folders under {args.input_data}")
        return

    print(f"Found {len(subjects)} subject(s)")
    print(
        f"FS={args.fs} Hz | window={args.window_size} ({args.window_size/args.fs:.3f}s)"
        f" | stride={args.stride} ({args.stride/args.fs:.3f}s)"
    )
    print(f"Channels: {args.channels} mode={args.channel_mode}")
    print(f"Normalization: {args.normalization}")
    if args.enable_filtering:
        print(
            f"Filtering: ON (bandpass {args.band_low}-{args.band_high} Hz, "
            f"order={args.band_order}; notch {args.notch_freq} Hz, Q={args.notch_q})"
        )
    else:
        print("Filtering: OFF")
    if args.split_type == "repetition":
        print("Repetition split: train {1,3,4,6}, val {2}, test {5}")

    X_all, y_all, r_all = [], [], []

    for subj in subjects:
        subj_path = os.path.join(args.input_data, subj)
        mat_files = collect_subject_mat_files(subj_path)
        if not mat_files:
            print(f"  [{subj}] no .mat files; skip")
            continue

        files_data = []
        print(f"Subject {subj}: {len(mat_files)} files")
        for path in mat_files:
            rel = os.path.relpath(path, subj_path)
            try:
                emg, lab, rep = parse_db5_file(path)
                emg = select_n_channels(emg, args.channel_mode, args.channels)
                files_data.append((rel, emg, lab, rep))
            except Exception as exc:
                print(f"  ! {rel}: {exc}")
                continue

        if not files_data:
            print(f"  [{subj}] no usable files; skip")
            continue

        files_proc = []
        for rel, emg, lab, rep in files_data:
            emg_proc = emg
            if args.enable_filtering:
                emg_proc = filter_emg_zero_phase(
                    emg_proc,
                    fs=float(args.fs),
                    band_low=float(args.band_low),
                    band_high=float(args.band_high),
                    band_order=int(args.band_order),
                    notch_freq=float(args.notch_freq),
                    notch_q=float(args.notch_q),
                )
            if not np.isfinite(emg_proc).all():
                raise ValueError(f"Non-finite values after preprocessing in {subj}/{rel}")
            files_proc.append((rel, emg_proc, lab, rep))

        train_segments = []
        for _rel, emg_proc, _lab, rep in files_proc:
            mask_train = np.isin(rep, TRAIN_REPEATS)
            for s, e in contiguous_runs(mask_train):
                seg = emg_proc[s:e, :]
                if seg.shape[0] > 0:
                    train_segments.append(seg)

        if args.normalization == "zscore":
            if not train_segments:
                print(f"  [{subj}] no TRAIN repetition samples found; skip subject")
                continue
            mu, std = compute_mu_std_from_segments(train_segments)
        else:
            mu = std = None

        Xs_subj, ys_subj, rs_subj = [], [], []
        for rel, emg_proc, lab, rep in files_proc:
            if args.normalization == "zscore":
                emg_norm = normalize_zscore(emg_proc, mu, std)
            else:
                emg_norm = normalize_maxabs(emg_proc)
            if not np.isfinite(emg_norm).all():
                raise ValueError(f"Non-finite values after normalization in {subj}/{rel}")

            X, y, r = segment_within_repetition_majority(
                emg_t_c=emg_norm,
                labels_t=lab,
                reps_t=rep,
                wsize=args.window_size,
                step=args.stride,
                min_majority=args.min_majority,
                drop_ambiguous=args.drop_ambiguous,
            )
            if X.shape[0] == 0:
                continue
            Xs_subj.append(X)
            ys_subj.append(y)
            rs_subj.append(r)

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

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    r_all = np.concatenate(r_all, axis=0)

    uniq = sorted({int(v) for v in set(y_all.tolist())})
    old2new = {int(o): i for i, o in enumerate(uniq)}
    new2old = {i: int(o) for i, o in enumerate(uniq)}
    y_all = np.vectorize(old2new.get, otypes=[np.int64])(y_all)

    with open(os.path.join(args.output_h5, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump({"old2new": old2new, "new2old": new2old}, f, indent=2)
    print(f"Saved label_map.json -> {os.path.join(args.output_h5, 'label_map.json')}")
    print(f"Raw labels: {uniq}")
    print(f"Remap: {old2new}")

    splits = {
        "train": {"X": None, "y": None},
        "val": {"X": None, "y": None},
        "test": {"X": None, "y": None},
    }

    if args.split_type == "repetition":
        for name, rep_set in (("train", TRAIN_REPEATS), ("val", VAL_REPEATS), ("test", TEST_REPEATS)):
            mask = np.isin(r_all, rep_set)
            splits[name]["X"] = X_all[mask]
            splits[name]["y"] = y_all[mask]
    else:
        test_frac = float(args.random_test)
        val_frac = float(args.random_val)
        assert 0.0 < test_frac < 1.0 and 0.0 < val_frac < 1.0 and test_frac + val_frac < 1.0

        idx = np.arange(y_all.shape[0])
        idx_rest, idx_test = train_test_split(
            idx, test_size=test_frac, random_state=args.seed, stratify=y_all
        )
        val_size_rel = val_frac / (1.0 - test_frac)
        idx_train, idx_val = train_test_split(
            idx_rest, test_size=val_size_rel, random_state=args.seed, stratify=y_all[idx_rest]
        )
        splits["train"]["X"], splits["train"]["y"] = X_all[idx_train], y_all[idx_train]
        splits["val"]["X"], splits["val"]["y"] = X_all[idx_val], y_all[idx_val]
        splits["test"]["X"], splits["test"]["y"] = X_all[idx_test], y_all[idx_test]

    if args.max_rest_ratio >= 0:
        Xs, ys = splits["train"]["X"], splits["train"]["y"]
        if ys is not None and ys.size > 0:
            keep = cap_rest_indices(ys, args.max_rest_ratio)
            splits["train"]["X"] = Xs[keep]
            splits["train"]["y"] = ys[keep]

    for split in ("train", "val", "test"):
        Xs, ys = splits[split]["X"], splits[split]["y"]
        if ys is None or ys.size == 0:
            print(f"[{split}] empty; skip save.")
            continue
        cnt = Counter(ys.tolist())
        print(f"[{split}] label distribution head: {cnt.most_common(10)} | classes={len(cnt)}")
        sanity_norm_print(split, Xs)
        save_h5(os.path.join(args.output_h5, f"db5_{split}_set.h5"), Xs, ys)

    print("\nDone.")


if __name__ == "__main__":
    main()

