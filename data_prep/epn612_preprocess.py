#!/usr/bin/env python3

import argparse
import glob
import json
import os

import h5py
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt
from tqdm.auto import tqdm


FS = 200.0
N_CH = 8
GESTURE_MAP = {
    "noGesture": 0,
    "waveIn": 1,
    "waveOut": 2,
    "pinch": 3,
    "open": 4,
    "fist": 5,
    "notProvided": 6,
}
INVALID_LABEL = GESTURE_MAP["notProvided"]
EPS = 1e-8
STD_FLOOR = 1e-3
CLIP_VALUE = 10.0


def normalize_maxabs(signal: np.ndarray) -> np.ndarray:
    scale = np.max(np.abs(signal), axis=1, keepdims=True)
    scale[scale == 0] = 1.0
    return (signal / scale).astype(np.float32)


def adjust_length(signal: np.ndarray, seq_len: int) -> np.ndarray:
    if signal.shape[1] >= seq_len:
        return signal[:, :seq_len]
    pad = np.zeros((signal.shape[0], seq_len - signal.shape[1]), dtype=signal.dtype)
    return np.concatenate((signal, pad), axis=1)


def make_windows(signal: np.ndarray, seq_len: int, step_size: int) -> list[np.ndarray]:
    signal = adjust_length(signal, seq_len)
    if signal.shape[1] == seq_len:
        return [signal]
    windows = []
    for start in range(0, signal.shape[1] - seq_len + 1, step_size):
        windows.append(signal[:, start:start + seq_len])
    if not windows or (signal.shape[1] - seq_len) % step_size != 0:
        windows.append(signal[:, -seq_len:])
    return windows


def bandpass_sos(fs: float, low: float, high: float, order: int):
    nyq = 0.5 * fs
    low = max(low, 0.01 * nyq)
    high = min(high, 0.95 * nyq)
    if high <= low:
        raise ValueError(f"Invalid bandpass settings: low={low}, high={high}, fs={fs}")
    return butter(order, [low, high], btype="bandpass", fs=fs, output="sos")


def filter_emg(signal: np.ndarray, fs: float, band_low: float, band_high: float, band_order: int,
               notch_freq: float, notch_q: float) -> np.ndarray:
    x = np.nan_to_num(signal.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    y = sosfiltfilt(bandpass_sos(fs, band_low, band_high, band_order), x, axis=1)
    if 0 < notch_freq < 0.5 * fs:
        b, a = iirnotch(notch_freq, notch_q, fs)
        y = filtfilt(b, a, y, axis=1)
    return np.nan_to_num(y.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def normalize_filtered(signal: np.ndarray) -> np.ndarray:
    mean = signal.mean(axis=1, keepdims=True)
    std = np.maximum(signal.std(axis=1, keepdims=True), STD_FLOOR)
    out = (signal - mean) / (std + EPS)
    return np.clip(out, -CLIP_VALUE, CLIP_VALUE).astype(np.float32)


def extract_emg_signal(sample: dict, seq_len: int, args) -> tuple[list[np.ndarray], int]:
    emg = np.stack([sample["emg"][key] for key in sorted(sample["emg"].keys())], dtype=np.float32) / 128.0
    windows = make_windows(emg, seq_len, args.step_size)
    processed = []
    for emg in windows:
        if args.enable_filtering:
            emg = filter_emg(
                emg,
                fs=args.fs,
                band_low=args.band_low,
                band_high=args.band_high,
                band_order=args.band_order,
                notch_freq=args.notch_freq,
                notch_q=args.notch_q,
            )
        if args.normalization == "zscore":
            processed.append(normalize_filtered(emg))
        else:
            processed.append(normalize_maxabs(emg))
    label = GESTURE_MAP.get(sample.get("gestureName", "notProvided"), INVALID_LABEL)
    return processed, label


def save_h5(path: str, data: list[np.ndarray], labels: list[int]):
    with h5py.File(path, "w") as h5f:
        h5f.create_dataset("data", data=np.asarray(data, dtype=np.float32))
        h5f.create_dataset("label", data=np.asarray(labels, dtype=np.int64))


def process_training_json(source_dir: str, seq_len: int, args, train_data, train_labels, val_data, val_labels):
    for file_path in tqdm(glob.glob(os.path.join(source_dir, "user*", "user*.json")),
                          desc="Training JSON", leave=False):
        with open(file_path, "r", encoding="utf-8") as f:
            user_data = json.load(f)

        for sample in user_data.get("trainingSamples", {}).values():
            emg_list, label = extract_emg_signal(sample, seq_len, args)
            if label != INVALID_LABEL:
                train_data.extend(emg_list)
                train_labels.extend([label] * len(emg_list))

        for sample in user_data.get("testingSamples", {}).values():
            emg_list, label = extract_emg_signal(sample, seq_len, args)
            if label != INVALID_LABEL:
                val_data.extend(emg_list)
                val_labels.extend([label] * len(emg_list))


def process_testing_json(source_dir: str, seq_len: int, args, train_data, train_labels, test_data, test_labels):
    for file_path in tqdm(glob.glob(os.path.join(source_dir, "user*", "user*.json")),
                          desc="Testing JSON", leave=False):
        with open(file_path, "r", encoding="utf-8") as f:
            user_data = json.load(f)

        grouped = {name: [] for name in GESTURE_MAP}
        for sample in user_data.get("trainingSamples", {}).values():
            grouped[sample.get("gestureName", "notProvided")].append(sample)

        for samples in grouped.values():
            for idx, sample in enumerate(samples):
                emg_list, label = extract_emg_signal(sample, seq_len, args)
                if label == INVALID_LABEL:
                    continue
                if idx < 10:
                    train_data.extend(emg_list)
                    train_labels.extend([label] * len(emg_list))
                else:
                    test_data.extend(emg_list)
                    test_labels.extend([label] * len(emg_list))


def main():
    ap = argparse.ArgumentParser(description="Preprocess EPN612 JSON dataset into train/val/test HDF5 files.")
    ap.add_argument("--source_training", type=str, default="./EMG-EPN612 Dataset/trainingJSON")
    ap.add_argument("--source_testing", type=str, default="./EMG-EPN612 Dataset/testingJSON")
    ap.add_argument("--out_dir", type=str, default="./EPN612_processed")
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--step_size", type=int, default=512)
    ap.add_argument("--normalization", type=str, default="maxabs", choices=["maxabs", "zscore"])
    ap.add_argument("--enable_filtering", action="store_true")
    ap.add_argument("--fs", type=float, default=FS)
    ap.add_argument("--band_low", type=float, default=20.0)
    ap.add_argument("--band_high", type=float, default=190.0)
    ap.add_argument("--band_order", type=int, default=4)
    ap.add_argument("--notch_freq", type=float, default=50.0)
    ap.add_argument("--notch_q", type=float, default=30.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train_data, train_labels = [], []
    val_data, val_labels = [], []
    test_data, test_labels = [], []

    process_training_json(args.source_training, args.seq_len, args, train_data, train_labels, val_data, val_labels)
    process_testing_json(args.source_testing, args.seq_len, args, train_data, train_labels, test_data, test_labels)

    save_h5(os.path.join(args.out_dir, "epn612_train_set.h5"), train_data, train_labels)
    save_h5(os.path.join(args.out_dir, "epn612_val_set.h5"), val_data, val_labels)
    save_h5(os.path.join(args.out_dir, "epn612_test_set.h5"), test_data, test_labels)

    mode = args.normalization
    print(f"Saved EPN612 splits to {args.out_dir} using {mode} preprocessing.")


if __name__ == "__main__":
    main()
