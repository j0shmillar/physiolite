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


# ------------------------------------------------------
# Global Parameters
# ------------------------------------------------------
FS = 500
WINDOW_SIZE = 2048
STEP_SIZE = 1024
CHUNK_SIZE = 500

SUPERCLASS_MAP = {
    "NORM": 0,
    "MI": 1,
    "STTC": 2,
    "CD": 3,
    "HYP": 4,
}
NUM_CLASSES = len(SUPERCLASS_MAP)


# ------------------------------------------------------
# 1. Build SCP code to diagnostic_class mapping
# ------------------------------------------------------
def load_diag_map(scp_csv_path):
    """Load SCP code to diagnostic class mapping."""
    df_scp = pd.read_csv(scp_csv_path, index_col=0)
    df_scp = df_scp[df_scp["diagnostic"] == 1]
    return {code: row["diagnostic_class"] for code, row in df_scp.iterrows()}


# ------------------------------------------------------
# 2. Parse scp_codes to single label
# ------------------------------------------------------
def parse_scp_codes_single_label(scp_str, diag_map, threshold=80.0):
    """
    Parse SCP codes string to single label with priority selection.

    - Deserialize scp_codes string -> dict{code:likelihood}
    - Find all classes with likelihood >= threshold that are in SUPERCLASS_MAP
    - Select single label by priority:
        NORM=0 < MI=1 < STTC=2 < CD=3 < HYP=4
    - Returns single class index, or -1 if none found
    """
    try:
        scp_dict = ast.literal_eval(scp_str)
    except Exception:
        return -1

    found_classes = []
    for code, lk in scp_dict.items():
        if lk < threshold:
            continue
        cls_name = diag_map.get(code)
        if cls_name in SUPERCLASS_MAP:
            found_classes.append(cls_name)

    if not found_classes:
        return -1

    selected_class = max(found_classes, key=lambda x: SUPERCLASS_MAP[x])
    return SUPERCLASS_MAP[selected_class]


# ------------------------------------------------------
# 3. Read ECG signal (.dat/.hea) - 500 Hz
# ------------------------------------------------------
def read_ecg_data(record_path):
    """Read 500 Hz ECG data."""
    sig, _ = wfdb.rdsamp(record_path)
    return sig.T  # preserve old working dtype/behavior exactly


# ------------------------------------------------------
# 4. MinMax normalization (exact old behavior)
# ------------------------------------------------------
def normalize_ecg_minmax(ecg):
    """
    Apply MinMax normalization to [-1, 1] range for each lead.

    Args:
        ecg: shape (12, n_samples) or generally (C, T)

    Returns:
        Normalized ECG signal with same dtype behavior as the old script.
    """
    normalized = np.zeros_like(ecg)
    for i in range(ecg.shape[0]):
        channel_data = ecg[i]
        min_val = channel_data.min()
        max_val = channel_data.max()

        if max_val - min_val > 1e-8:
            normalized[i] = 2 * (channel_data - min_val) / (max_val - min_val) - 1
        else:
            normalized[i] = 0.0

    return normalized


# ------------------------------------------------------
# 5. Optional filtering path
# ------------------------------------------------------
def filter_ecg(
    ecg,
    fs,
    band_low,
    band_high,
    band_order,
    notch_freq,
    notch_q,
    baseline_kernel_sec,
):
    """
    Apply optional filtering:
      - notch
      - bandpass
      - median baseline removal

    Input/output shape: (C, T)
    """
    x = np.nan_to_num(ecg.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    nyq = 0.5 * fs

    if 0 < notch_freq < nyq:
        b, a = iirnotch(notch_freq, notch_q, fs)
        for idx in range(x.shape[0]):
            x[idx] = filtfilt(b, a, x[idx])

    low = float(band_low)
    high = min(float(band_high), 0.95 * nyq)
    if high <= low:
        raise ValueError(
            f"Invalid band after clamp: low={low:.6f}, high={high:.6f}, fs={fs}, nyq={nyq}"
        )

    b, a = butter(band_order, [low, high], btype="bandpass", fs=fs)
    for idx in range(x.shape[0]):
        x[idx] = filtfilt(b, a, x[idx])

    kernel = int(baseline_kernel_sec * fs) + 1
    if kernel % 2 == 0:
        kernel += 1
    kernel = max(kernel, 3)

    baseline = np.zeros_like(x)
    for idx in range(x.shape[0]):
        baseline[idx] = medfilt(x[idx], kernel_size=kernel)

    y = x - baseline
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y


# ------------------------------------------------------
# 6. Lead handling
# ------------------------------------------------------
def pad_leads(ecg, num_leads=12):
    if ecg.shape[0] >= num_leads:
        return ecg[:num_leads]
    out = np.zeros((num_leads, ecg.shape[1]), dtype=ecg.dtype)
    out[:ecg.shape[0]] = ecg
    return out


def handle_missing_leads(ecg, mode, num_leads=12):
    """
    Old working preprocessing effectively assumed 12 leads, but keep the current
    optional handling for filtered/unfiltered compatibility.
    """
    if ecg.shape[0] >= num_leads:
        return ecg[:num_leads]
    if mode == "drop":
        return None
    return pad_leads(ecg, num_leads)


# ------------------------------------------------------
# 7. Sliding window segmentation
# ------------------------------------------------------
def sliding_window_ecg(ecg, window_size, step_size, args):
    """
    Sliding window segmentation.

    Exact old unfiltered behavior:
      - segment first
      - MinMax normalize each window independently
      - final tail window is zero-padded then normalized

    Filtered path:
      - segment first
      - optionally filter that window
      - then apply the same MinMax normalization
    """
    segs = []
    L = ecg.shape[1]

    for start in range(0, L - window_size + 1, step_size):
        seg = ecg[:, start:start + window_size]

        if args.enable_filtering:
            seg = filter_ecg(
                seg,
                args.fs,
                args.band_low,
                args.band_high,
                args.band_order,
                args.notch_freq,
                args.notch_q,
                args.baseline_kernel_sec,
            )

        seg = normalize_ecg_minmax(seg)
        segs.append(seg)

    if L % step_size != 0:
        pad = np.zeros((ecg.shape[0], window_size))
        rem = ecg[:, (L // step_size) * step_size:]
        pad[:, :rem.shape[1]] = rem

        if args.enable_filtering:
            pad = filter_ecg(
                pad,
                args.fs,
                args.band_low,
                args.band_high,
                args.band_order,
                args.notch_freq,
                args.notch_q,
                args.baseline_kernel_sec,
            )

        pad = normalize_ecg_minmax(pad)
        segs.append(pad)

    return segs


# ------------------------------------------------------
# 8. HDF5 creation/writing (exact old writer behavior)
# ------------------------------------------------------
def create_h5(path, data_shape, chunk_size=CHUNK_SIZE):
    """Create HDF5 file with compatible format."""
    f = h5py.File(path, "w")

    f.create_dataset(
        "data",
        shape=(0,) + data_shape,
        maxshape=(None,) + data_shape,
        chunks=(chunk_size,) + data_shape,
        dtype=np.float32,
    )

    f.create_dataset(
        "label",
        shape=(0,),
        maxshape=(None,),
        chunks=(chunk_size,),
        dtype=np.int64,
    )

    return f


def append_h5(f, data_batch, label_batch):
    """Append data to HDF5 file."""
    x_ds = f["data"]
    y_ds = f["label"]

    old_size = x_ds.shape[0]
    new_size = old_size + data_batch.shape[0]

    x_ds.resize((new_size,) + x_ds.shape[1:])
    y_ds.resize((new_size,))

    x_ds[old_size:new_size] = data_batch
    y_ds[old_size:new_size] = label_batch


# ------------------------------------------------------
# 9. Process one split
# ------------------------------------------------------
def process_split(df, root, out_file, diag_map, threshold, args):
    """Process a single split exactly like the old script, with optional filtering."""
    example_path = os.path.join(root, df.iloc[0]["filename_hr"])
    example = read_ecg_data(example_path)

    assert example.shape[0] >= 12, f"Expected at least 12 leads, got {example.shape[0]}"

    data_shape = (12, args.window_size)
    h5f = create_h5(out_file, data_shape)

    data_buffer = []
    label_buffer = []

    valid_samples = 0
    total_segments = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {os.path.basename(out_file)}"):
        single_label = parse_scp_codes_single_label(row["scp_codes"], diag_map, threshold)
        if single_label == -1:
            continue

        record_path = os.path.join(root, row["filename_hr"])
        try:
            sig = read_ecg_data(record_path)
        except Exception as e:
            print(f"Error reading {record_path}: {e}")
            continue

        sig = handle_missing_leads(sig, args.missing_leads, 12)
        if sig is None:
            continue

        if sig.shape[1] < args.window_size:
            continue

        sig = sig[:12]

        segments = sliding_window_ecg(sig, args.window_size, args.step_size, args)

        for seg in segments:
            data_buffer.append(seg.astype(np.float32))
            label_buffer.append(single_label)
            total_segments += 1

        valid_samples += 1

        if len(data_buffer) >= CHUNK_SIZE:
            data_batch = np.stack(data_buffer)
            label_batch = np.array(label_buffer, dtype=np.int64)
            append_h5(h5f, data_batch, label_batch)
            data_buffer.clear()
            label_buffer.clear()

    if data_buffer:
        data_batch = np.stack(data_buffer)
        label_batch = np.array(label_buffer, dtype=np.int64)
        append_h5(h5f, data_batch, label_batch)

    print(f"\n[Statistics] {out_file}")
    print(f"  Valid samples: {valid_samples}/{len(df)} ({valid_samples/len(df)*100:.1f}%)")
    print(f"  Total segments: {total_segments}")
    print(f"  HDF5 keys: {list(h5f.keys())}")
    print(f"  Data shape: {h5f['data'].shape}")
    print(f"  Label shape: {h5f['label'].shape}")

    labels = h5f["label"][:]
    print("  Label distribution:")
    for class_name, class_idx in SUPERCLASS_MAP.items():
        count = np.sum(labels == class_idx)
        pct = (count / len(labels) * 100.0) if len(labels) > 0 else 0.0
        print(f"    {class_name} ({class_idx}): {count} ({pct:.1f}%)")

    h5f.close()


# ------------------------------------------------------
# 10. Main
# ------------------------------------------------------
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

    root = args.root
    df = pd.read_csv(os.path.join(root, "ptbxl_database.csv"))
    diag_map = load_diag_map(os.path.join(root, "scp_statements.csv"))

    print("=== PTB-XL Single Label Preprocessing ===")
    print(f"Total records: {len(df)}")
    print(f"Superclasses: {list(SUPERCLASS_MAP.keys())}")
    print(f"Window size: {args.window_size}, Step size: {args.step_size}")
    print(f"Sampling rate: {args.fs} Hz")
    print("Normalization: MinMax [-1,1]")
    print(f"Filtering: {'Enabled' if args.enable_filtering else 'None (raw signals)'}")
    print("Data source: records500/ via filename_hr")

    splits = {
        "train": df[df["strat_fold"] <= 8].reset_index(drop=True),
        "val": df[df["strat_fold"] == 9].reset_index(drop=True),
        "test": df[df["strat_fold"] == 10].reset_index(drop=True),
    }

    print("\nData splits:")
    for name, subdf in splits.items():
        print(f"  {name}: {len(subdf)} records")

    for name, subdf in splits.items():
        out_file = os.path.join(root, f"{name}.h5")
        print(f"\n{'=' * 50}")
        print(f"Processing {name} split...")
        process_split(subdf, root, out_file, diag_map, threshold=args.threshold, args=args)
        print(f"Saved to: {out_file}")

    print(f"\n{'=' * 50}")
    print("Preprocessing completed!")
    print("\nGenerated files are compatible with the old loader:")
    print("- Keys: 'data' (N, C, T), 'label' (N,)")
    print("- Data type: float32, Label type: int64")
    print("- Single label classification with 5 classes")
    print(f"- Filtering mode: {'filtered' if args.enable_filtering else 'unfiltered'}")


if __name__ == "__main__":
    main()
