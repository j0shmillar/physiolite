#!/usr/bin/env python3

import argparse
import ast
import os

import h5py
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt, iirnotch, medfilt
from tqdm import tqdm


FS = 500
WINDOW_SIZE = 2048
STEP_SIZE = 1024
CHUNK_SIZE = 500
SUPERCLASS_MAP = {"NORM": 0, "MI": 1, "STTC": 2, "CD": 3, "HYP": 4}


def load_diag_map(path: str) -> dict[str, str]:
    df = pd.read_csv(path, index_col=0)
    df = df[df["diagnostic"] == 1]
    return {code: row["diagnostic_class"] for code, row in df.iterrows()}


def parse_scp_codes_single_label(scp_str: str, diag_map: dict[str, str], threshold: float) -> int:
    try:
        scp_dict = ast.literal_eval(scp_str)
    except Exception:
        return -1

    found = []
    for code, likelihood in scp_dict.items():
        if likelihood < threshold:
            continue
        cls_name = diag_map.get(code)
        if cls_name in SUPERCLASS_MAP:
            found.append(cls_name)
    if not found:
        return -1
    return SUPERCLASS_MAP[max(found, key=lambda name: SUPERCLASS_MAP[name])]


def read_ecg(record_path: str) -> np.ndarray:
    signal, _ = wfdb.rdsamp(record_path)
    return signal.T.astype(np.float32)


def zscore_normalize(ecg: np.ndarray) -> np.ndarray:
    mean = ecg.mean(axis=1, keepdims=True)
    std = ecg.std(axis=1, keepdims=True) + 1e-8
    return ((ecg - mean) / std).astype(np.float32)


def filter_ecg(ecg: np.ndarray, fs: float, band_low: float, band_high: float, band_order: int,
               notch_freq: float, notch_q: float, baseline_kernel_sec: float) -> np.ndarray:
    x = np.nan_to_num(ecg.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if 0 < notch_freq < 0.5 * fs:
        b, a = iirnotch(notch_freq, notch_q, fs)
        for idx in range(x.shape[0]):
            x[idx] = filtfilt(b, a, x[idx])
    b, a = butter(band_order, [band_low, min(band_high, 0.95 * fs * 0.5)], btype="bandpass", fs=fs)
    for idx in range(x.shape[0]):
        x[idx] = filtfilt(b, a, x[idx])
    kernel = int(baseline_kernel_sec * fs) + 1
    if kernel % 2 == 0:
        kernel += 1
    baseline = np.zeros_like(x)
    for idx in range(x.shape[0]):
        baseline[idx] = medfilt(x[idx], kernel_size=kernel)
    return (x - baseline).astype(np.float32)


def pad_leads(ecg: np.ndarray, num_leads: int = 12) -> np.ndarray:
    if ecg.shape[0] >= num_leads:
        return ecg[:num_leads]
    out = np.zeros((num_leads, ecg.shape[1]), dtype=ecg.dtype)
    out[:ecg.shape[0]] = ecg
    return out


def handle_missing_leads(ecg: np.ndarray, mode: str, num_leads: int = 12) -> np.ndarray | None:
    if ecg.shape[0] >= num_leads:
        return ecg[:num_leads]
    if mode == "drop":
        return None
    return pad_leads(ecg, num_leads)


def sliding_windows(ecg: np.ndarray, args) -> list[np.ndarray]:
    segments = []
    length = ecg.shape[1]
    for start in range(0, length - args.window_size + 1, args.step_size):
        seg = ecg[:, start:start + args.window_size]
        seg = filter_ecg(seg, args.fs, args.band_low, args.band_high, args.band_order,
                         args.notch_freq, args.notch_q, args.baseline_kernel_sec) if args.enable_filtering else seg
        seg = zscore_normalize(seg)
        segments.append(seg.astype(np.float32))
    if length % args.step_size != 0:
        padded = np.zeros((ecg.shape[0], args.window_size), dtype=np.float32)
        tail = ecg[:, (length // args.step_size) * args.step_size:]
        padded[:, :tail.shape[1]] = tail
        padded = filter_ecg(padded, args.fs, args.band_low, args.band_high, args.band_order,
                            args.notch_freq, args.notch_q, args.baseline_kernel_sec) if args.enable_filtering else padded
        padded = zscore_normalize(padded)
        segments.append(padded.astype(np.float32))
    return segments


def create_h5(path: str, data_shape: tuple[int, int]):
    h5f = h5py.File(path, "w")
    h5f.create_dataset("data", shape=(0,) + data_shape, maxshape=(None,) + data_shape,
                       chunks=(CHUNK_SIZE,) + data_shape, dtype=np.float32)
    h5f.create_dataset("label", shape=(0,), maxshape=(None,), chunks=(CHUNK_SIZE,), dtype=np.int64)
    return h5f


def append_h5(h5f: h5py.File, data_batch: np.ndarray, label_batch: np.ndarray):
    data_ds, label_ds = h5f["data"], h5f["label"]
    old = data_ds.shape[0]
    new = old + data_batch.shape[0]
    data_ds.resize((new,) + data_ds.shape[1:])
    label_ds.resize((new,))
    data_ds[old:new] = data_batch
    label_ds[old:new] = label_batch


def process_split(df: pd.DataFrame, root: str, out_file: str, diag_map: dict[str, str], args):
    h5f = create_h5(out_file, data_shape=(12, args.window_size))
    data_buffer, label_buffer = [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=os.path.basename(out_file)):
        label = parse_scp_codes_single_label(row["scp_codes"], diag_map, threshold=args.threshold)
        if label == -1:
            continue
        signal = handle_missing_leads(read_ecg(os.path.join(root, row["filename_hr"])), args.missing_leads, 12)
        if signal is None:
            continue
        if signal.shape[1] < args.window_size:
            continue
        for seg in sliding_windows(signal, args):
            data_buffer.append(seg)
            label_buffer.append(label)
        if len(data_buffer) >= CHUNK_SIZE:
            append_h5(h5f, np.stack(data_buffer), np.asarray(label_buffer, dtype=np.int64))
            data_buffer.clear()
            label_buffer.clear()

    if data_buffer:
        append_h5(h5f, np.stack(data_buffer), np.asarray(label_buffer, dtype=np.int64))
    h5f.close()


def main():
    ap = argparse.ArgumentParser(description="Preprocess PTB-XL into train/val/test HDF5 files.")
    ap.add_argument("--root", type=str, default="physionet.org/files/ptb-xl/1.0.3/")
    ap.add_argument("--threshold", type=float, default=80.0)
    ap.add_argument("--missing_leads", type=str, default="pad", choices=["pad", "drop"])
    ap.add_argument("--enable_filtering", action="store_true")
    ap.add_argument("--fs", type=float, default=FS)
    ap.add_argument("--window_size", type=int, default=WINDOW_SIZE)
    ap.add_argument("--step_size", type=int, default=STEP_SIZE)
    ap.add_argument("--band_low", type=float, default=0.67)
    ap.add_argument("--band_high", type=float, default=40.0)
    ap.add_argument("--band_order", type=int, default=4)
    ap.add_argument("--notch_freq", type=float, default=50.0)
    ap.add_argument("--notch_q", type=float, default=30.0)
    ap.add_argument("--baseline_kernel_sec", type=float, default=0.4)
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(args.root, "ptbxl_database.csv"))
    diag_map = load_diag_map(os.path.join(args.root, "scp_statements.csv"))
    splits = {
        "train": df[df["strat_fold"] <= 8].reset_index(drop=True),
        "val": df[df["strat_fold"] == 9].reset_index(drop=True),
        "test": df[df["strat_fold"] == 10].reset_index(drop=True),
    }

    for split_name, split_df in splits.items():
        process_split(split_df, args.root, os.path.join(args.root, f"{split_name}.h5"), diag_map, args)

    mode = "filtered" if args.enable_filtering else "unfiltered"
    print(f"Saved PTB-XL train/val/test HDF5 files under {args.root} ({mode}).")


if __name__ == "__main__":
    main()
