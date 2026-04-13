#!/usr/bin/env python3

import argparse
import json
import os
import re

import h5py
import numpy as np
import scipy.io
from scipy.signal import butter, filtfilt, iirnotch, medfilt
from sklearn.model_selection import train_test_split
from tqdm import tqdm


FS = 500
WINDOW_SIZE = 2048
STEP_SIZE = 1024
CHUNK_SIZE = 500
NUM_LEADS = 12
SEED = 42
CLASS_SPECS = [
    ("SNR", {"426783006"}),
    ("AF", {"164889003"}),
    ("IAVB", {"270492004"}),
    ("LBBB", {"733534002", "164909002"}),
    ("RBBB", {"713427006", "59118001"}),
    ("PAC", {"284470004"}),
    ("PVC", {"427172004", "164884008", "17338001"}),
    ("STD", {"429622005"}),
    ("STE", {"164931005"}),
]
LABELS_9 = [name for name, _ in CLASS_SPECS]
CODE2IDX = {code: idx for idx, (_, codes) in enumerate(CLASS_SPECS) for code in codes}


def read_ecg_mat(path: str) -> np.ndarray:
    mat = scipy.io.loadmat(path)
    for key in ("val", "data", "ecg"):
        if key in mat and isinstance(mat[key], np.ndarray):
            signal = np.asarray(mat[key], dtype=np.float32)
            break
    else:
        raise ValueError(f"No ECG array found in {path}")
    if signal.shape[0] != NUM_LEADS and signal.shape[1] == NUM_LEADS:
        signal = signal.T
    return np.nan_to_num(signal.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def pad_leads(signal: np.ndarray, num_leads: int = NUM_LEADS) -> np.ndarray:
    if signal.shape[0] >= num_leads:
        return signal[:num_leads]
    padded = np.zeros((num_leads, signal.shape[1]), dtype=signal.dtype)
    padded[:signal.shape[0]] = signal
    return padded


def handle_missing_leads(signal: np.ndarray, mode: str, num_leads: int = NUM_LEADS) -> np.ndarray | None:
    if signal.shape[0] >= num_leads:
        return signal[:num_leads]
    if mode == "drop":
        return None
    return pad_leads(signal, num_leads)


def parse_dx_codes(header_path: str) -> list[str]:
    for encoding in ("utf-8", "latin-1", "gbk", "cp1252"):
        try:
            with open(header_path, "r", encoding=encoding, errors="strict") as f:
                for line in f:
                    line = line.strip()
                    if not (line.startswith("# Dx:") or line.startswith("#Dx:") or line.startswith("# Dx") or line.startswith("#Dx")):
                        continue
                    dx_part = line.split(":", 1)[1] if ":" in line else line[4:]
                    return [tok for tok in re.split(r"[,;\s]+", dx_part.strip()) if tok.isdigit()]
        except Exception:
            continue
    return []


def codes_to_multihot(codes: list[str]) -> tuple[bool, np.ndarray]:
    y = np.zeros((len(LABELS_9),), dtype=np.float32)
    matched = False
    for code in codes:
        idx = CODE2IDX.get(code)
        if idx is not None:
            y[idx] = 1.0
            matched = True
    return matched, y


def filter_ecg(ecg: np.ndarray, args) -> np.ndarray:
    x = np.nan_to_num(ecg.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
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


def zscore_windows(windows: np.ndarray) -> np.ndarray:
    mean = windows.mean(axis=2, keepdims=True)
    std = windows.std(axis=2, keepdims=True) + 1e-8
    return ((windows - mean) / std).astype(np.float32)


def sliding_windows(ecg: np.ndarray, args) -> list[np.ndarray]:
    starts = list(range(0, max(ecg.shape[1] - args.window_size + 1, 0), args.step_size)) or [0]
    if starts[-1] + args.window_size < ecg.shape[1]:
        starts.append(starts[-1] + args.step_size)
    windows = []
    for start in starts:
        end = start + args.window_size
        if end <= ecg.shape[1]:
            seg = ecg[:, start:end]
        else:
            seg = np.zeros((ecg.shape[0], args.window_size), dtype=np.float32)
            remain = ecg.shape[1] - start
            if remain > 0:
                seg[:, :remain] = ecg[:, start:]
        if args.enable_filtering:
            seg = filter_ecg(seg, args)
        windows.append(seg.astype(np.float32))
    return windows


def create_h5(path: str, num_classes: int, window_size: int):
    h5f = h5py.File(path, "w")
    h5f.create_dataset("data", shape=(0, NUM_LEADS, window_size), maxshape=(None, NUM_LEADS, window_size),
                       chunks=(CHUNK_SIZE, NUM_LEADS, window_size), dtype=np.float32, compression="gzip")
    h5f.create_dataset("label", shape=(0, num_classes), maxshape=(None, num_classes),
                       chunks=(CHUNK_SIZE, num_classes), dtype=np.float32, compression="gzip")
    return h5f


def append_h5(h5f: h5py.File, x_batch: np.ndarray, y_batch: np.ndarray):
    data_ds, label_ds = h5f["data"], h5f["label"]
    old = data_ds.shape[0]
    new = old + x_batch.shape[0]
    data_ds.resize((new,) + data_ds.shape[1:])
    label_ds.resize((new, label_ds.shape[1]))
    data_ds[old:new] = x_batch
    label_ds[old:new] = y_batch


def collect_records(root_dirs: list[str]) -> list[dict]:
    items = []
    for root in root_dirs:
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            hea_files = [name for name in filenames if name.lower().endswith(".hea")]
            for hea_name in hea_files:
                rec_id = os.path.splitext(hea_name)[0]
                hea_path = os.path.join(dirpath, hea_name)
                mat_path = next(
                    (os.path.join(dirpath, rec_id + ext) for ext in (".mat", ".MAT")
                     if os.path.exists(os.path.join(dirpath, rec_id + ext))),
                    None,
                )
                if mat_path is None:
                    continue
                items.append({
                    "rec_id": rec_id,
                    "hea_path": hea_path,
                    "mat_path": mat_path,
                    "codes": parse_dx_codes(hea_path),
                })
    return items


def process_split(items: list[dict], split_name: str, args):
    out_path = os.path.join(args.out_dir, f"cpsc_9class_{split_name}.h5")
    h5f = create_h5(out_path, len(LABELS_9), args.window_size)
    buf_x, buf_y = [], []

    for item in tqdm(items, desc=f"{split_name} records", ncols=120):
        ecg = handle_missing_leads(read_ecg_mat(item["mat_path"]), args.missing_leads)
        if ecg is None:
            continue
        matched, label = codes_to_multihot(item["codes"])
        if not matched:
            continue
        windows = np.asarray(sliding_windows(ecg, args), dtype=np.float32)
        windows = zscore_windows(windows)
        labels = np.repeat(label[None, :], windows.shape[0], axis=0)
        buf_x.extend(windows)
        buf_y.extend(labels)
        while len(buf_x) >= CHUNK_SIZE:
            append_h5(h5f, np.stack(buf_x[:CHUNK_SIZE]), np.stack(buf_y[:CHUNK_SIZE]))
            buf_x = buf_x[CHUNK_SIZE:]
            buf_y = buf_y[CHUNK_SIZE:]

    if buf_x:
        append_h5(h5f, np.stack(buf_x), np.stack(buf_y))
    h5f.close()


def main():
    ap = argparse.ArgumentParser(description="Preprocess CPSC 2018 into 9-class multilabel HDF5 files.")
    ap.add_argument("--root_dirs", nargs="+", default=["datasets/cpsc-2018/cpsc_2018"])
    ap.add_argument("--out_dir", type=str, default="datasets/CPSC")
    ap.add_argument("--missing_leads", type=str, default="pad", choices=["pad", "drop"])
    ap.add_argument("--window_size", type=int, default=WINDOW_SIZE)
    ap.add_argument("--step_size", type=int, default=STEP_SIZE)
    ap.add_argument("--fs", type=float, default=FS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--enable_filtering", action="store_true")
    ap.add_argument("--band_low", type=float, default=0.67)
    ap.add_argument("--band_high", type=float, default=40.0)
    ap.add_argument("--band_order", type=int, default=4)
    ap.add_argument("--notch_freq", type=float, default=50.0)
    ap.add_argument("--notch_q", type=float, default=30.0)
    ap.add_argument("--baseline_kernel_sec", type=float, default=0.4)
    args = ap.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    items = collect_records(args.root_dirs)
    train_items, temp_items = train_test_split(items, test_size=0.3, random_state=args.seed, shuffle=True)
    val_items, test_items = train_test_split(temp_items, test_size=1 / 3, random_state=args.seed, shuffle=True)

    with open(os.path.join(args.out_dir, "record_splits.json"), "w", encoding="utf-8") as f:
        json.dump({
            "train": [item["rec_id"] for item in train_items],
            "val": [item["rec_id"] for item in val_items],
            "test": [item["rec_id"] for item in test_items],
        }, f, indent=2)

    for split_name, split_items in (("train", train_items), ("val", val_items), ("test", test_items)):
        process_split(split_items, split_name, args)

    with open(os.path.join(args.out_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump({name: idx for idx, name in enumerate(LABELS_9)}, f, indent=2)

    print(f"Saved CPSC train/val/test HDF5 files under {args.out_dir}.")


if __name__ == "__main__":
    main()
