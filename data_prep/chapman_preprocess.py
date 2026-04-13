#!/usr/bin/env python3

import argparse
import glob
import json
import os

import h5py
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch, medfilt
from tqdm import tqdm


SAMPLING_RATE = 500
WINDOW_SIZE = 2048
STEP_SIZE = 1024
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
MERGE_MAP = {
    "SB": ["SB"],
    "AFIB": ["AFIB", "AF", "AFL"],
    "GSVT": ["SVT", "AT", "AVNRT", "AVRT", "SAAWR"],
    "SR": ["SR", "SI", "ST"],
}
LABEL_TO_ID = {cls: idx for idx, cls in enumerate(MERGE_MAP)}
NUM_CLASSES = len(LABEL_TO_ID)
CODE_TO_CLASS = {
    "426177001": "SB",
    "164889003": "AFIB",
    "164890007": "AFIB",
    "426761007": "GSVT",
    "713422000": "GSVT",
    "233896004": "GSVT",
    "233897008": "GSVT",
    "195101003": "GSVT",
    "427084000": "SR",
    "426783006": "SR",
    "427393009": "SR",
}


def parse_multihot_label(header_path: str) -> np.ndarray:
    labels = np.zeros(NUM_CLASSES, dtype=np.int8)
    if not os.path.exists(header_path):
        return labels
    with open(header_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("#Dx:"):
                continue
            items = [token.strip() for token in line.split(":", 1)[1].split(",")]
            for item in items:
                if item.isdigit() and item in CODE_TO_CLASS:
                    labels[LABEL_TO_ID[CODE_TO_CLASS[item]]] = 1
            for item in items:
                for cls, aliases in MERGE_MAP.items():
                    if item in aliases:
                        labels[LABEL_TO_ID[cls]] = 1
            break
    return labels


def zscore(signal: np.ndarray) -> np.ndarray:
    mean = signal.mean(axis=1, keepdims=True)
    std = signal.std(axis=1, keepdims=True) + 1e-8
    return (signal - mean) / std


def pad_leads(signal: np.ndarray, num_leads: int = 12) -> np.ndarray:
    if signal.shape[0] >= num_leads:
        return signal[:num_leads]
    padded = np.zeros((num_leads, signal.shape[1]), dtype=signal.dtype)
    padded[:signal.shape[0]] = signal
    return padded


def handle_missing_leads(signal: np.ndarray, mode: str, num_leads: int = 12) -> np.ndarray | None:
    if signal.shape[0] >= num_leads:
        return signal[:num_leads]
    if mode == "drop":
        return None
    return pad_leads(signal, num_leads)


def filter_ecg(signal: np.ndarray, args) -> np.ndarray:
    x = np.nan_to_num(signal.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if 0 < args.notch_freq < 0.5 * args.fs:
        b, a = iirnotch(args.notch_freq, args.notch_q, args.fs)
        for idx in range(x.shape[0]):
            x[idx] = filtfilt(b, a, x[idx])
    b, a = butter(args.band_order, [args.band_low, min(args.band_high, 0.95 * args.fs * 0.5)],
                  btype="bandpass", fs=args.fs)
    for idx in range(x.shape[0]):
        x[idx] = filtfilt(b, a, x[idx])
    kernel = int(args.baseline_kernel_sec * args.fs) + 1
    if kernel % 2 == 0:
        kernel += 1
    baseline = np.zeros_like(x)
    for idx in range(x.shape[0]):
        baseline[idx] = medfilt(x[idx], kernel_size=kernel)
    return (x - baseline).astype(np.float32)


def slide_and_cut(signal: np.ndarray, args) -> np.ndarray | None:
    segments = []
    start = 0
    while start + args.window_size <= signal.shape[1]:
        segments.append(signal[:, start:start + args.window_size])
        start += args.step_size
    if start < signal.shape[1]:
        padded = np.zeros((signal.shape[0], args.window_size), dtype=signal.dtype)
        tail = signal[:, start:]
        padded[:, :tail.shape[1]] = tail
        segments.append(padded)
    if not segments:
        return None
    return np.stack(segments, axis=0)


def process_record(mat_path: str, label: np.ndarray, args):
    try:
        mat = loadmat(mat_path)
    except Exception:
        return None, None
    signal = next((mat[key] for key in ("val", "ECG", "signal", "p_signals", "data") if key in mat), None)
    if signal is None or signal.ndim != 2:
        return None, None
    if signal.shape[0] != 12 and signal.shape[1] == 12:
        signal = signal.T
    signal = handle_missing_leads(signal, args.missing_leads)
    if signal is None:
        return None, None
    if signal.shape[1] < args.window_size:
        return None, None
    signal = signal.astype(np.float32)
    if args.enable_filtering:
        signal = filter_ecg(signal, args)
    signal = zscore(signal)
    segments = slide_and_cut(signal, args)
    if segments is None:
        return None, None
    return segments, np.tile(label, (segments.shape[0], 1))


def create_h5(path: str, window_size: int):
    h5f = h5py.File(path, "w")
    h5f.create_dataset("data", shape=(0, 12, window_size), maxshape=(None, 12, window_size),
                       chunks=(100, 12, window_size), compression="gzip", compression_opts=4, dtype="f4")
    h5f.create_dataset("label", shape=(0, NUM_CLASSES), maxshape=(None, NUM_CLASSES),
                       chunks=(100, NUM_CLASSES), compression="gzip", compression_opts=4, dtype="i1")
    return h5f


def append_h5(h5f: h5py.File, data: np.ndarray, labels: np.ndarray):
    data_ds, label_ds = h5f["data"], h5f["label"]
    old = data_ds.shape[0]
    new = old + data.shape[0]
    data_ds.resize((new, 12, data_ds.shape[2]))
    label_ds.resize((new, NUM_CLASSES))
    data_ds[old:new] = data
    label_ds[old:new] = labels


def main():
    ap = argparse.ArgumentParser(description="Preprocess Chapman-Shaoxing ECG into train/val/test HDF5 files.")
    ap.add_argument("--root_dir", type=str, default="./datasets/chapman-shaoxing")
    ap.add_argument("--out_dir", type=str, default="./datasets/ChapmanShaoxing")
    ap.add_argument("--missing_leads", type=str, default="pad", choices=["pad", "drop"])
    ap.add_argument("--window_size", type=int, default=WINDOW_SIZE)
    ap.add_argument("--step_size", type=int, default=STEP_SIZE)
    ap.add_argument("--fs", type=float, default=SAMPLING_RATE)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--enable_filtering", action="store_true")
    ap.add_argument("--band_low", type=float, default=0.67)
    ap.add_argument("--band_high", type=float, default=40.0)
    ap.add_argument("--band_order", type=int, default=4)
    ap.add_argument("--notch_freq", type=float, default=50.0)
    ap.add_argument("--notch_q", type=float, default=30.0)
    ap.add_argument("--baseline_kernel_sec", type=float, default=0.4)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    mats = glob.glob(os.path.join(args.root_dir, "**", "*.mat"), recursive=True)

    records = []
    for mat_path in tqdm(mats, desc="Scanning records", ncols=100):
        label = parse_multihot_label(mat_path.replace(".mat", ".hea"))
        if label.sum() > 0:
            records.append({
                "record_id": os.path.splitext(os.path.basename(mat_path))[0],
                "mat_path": mat_path,
                "label": label,
            })

    np.random.seed(args.seed)
    np.random.shuffle(records)
    total = len(records)
    train_end = int(TRAIN_RATIO * total)
    val_end = train_end + int(VAL_RATIO * total)
    splits = {
        "train": records[:train_end],
        "val": records[train_end:val_end],
        "test": records[val_end:],
    }

    with open(os.path.join(args.out_dir, "record_splits.json"), "w", encoding="utf-8") as f:
        json.dump({name: [r["record_id"] for r in items] for name, items in splits.items()}, f, indent=2)

    for split_name, split_records in splits.items():
        h5f = create_h5(os.path.join(args.out_dir, f"{split_name}.h5"), args.window_size)
        for record in tqdm(split_records, desc=f"Processing {split_name}", ncols=100):
            data, labels = process_record(record["mat_path"], record["label"], args)
            if data is not None:
                append_h5(h5f, data, labels)
        h5f.close()

    print(f"Saved Chapman-Shaoxing train/val/test HDF5 files under {args.out_dir}.")


if __name__ == "__main__":
    main()
