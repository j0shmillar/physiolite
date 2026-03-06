#!/usr/bin/env python3
"""Create tiny synthetic H5 datasets + a random teacher checkpoint for smoke testing."""

import argparse
import os

import h5py
import numpy as np
import torch

from model import BERTWaveletTransformer


def write_h5(path: str, x: np.ndarray, y: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=x, compression="gzip")
        f.create_dataset("label", data=y, compression="gzip")


def make_split(n: int, in_channels: int, seq_len: int, num_classes: int, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=(n, in_channels, seq_len)).astype(np.float32)
    y = rng.integers(0, num_classes, size=(n,), endpoint=False, dtype=np.int64)
    return x, y


def build_teacher_ckpt(path: str, in_channels: int, patch_size: int, num_classes: int, seed: int):
    torch.manual_seed(seed)
    model = BERTWaveletTransformer(
        in_channels=in_channels,
        max_level=3,
        wave_kernel_size=16,
        wavelet_names=["db6"],
        use_separate_channel=True,
        patch_size=(1, patch_size),
        embed_dim=128,
        depth=2,
        num_heads=4,
        mlp_ratio=4.0,
        dropout=0.1,
        use_pos_embed=True,
        pos_embed_type="2d",
        task_type="classification",
        num_classes=num_classes,
        head_config={"hidden_dims": [128], "dropout": 0.1, "pooling": "mean"},
        pooling="mean",
    )

    ckpt = {
        "model_state_dict": model.state_dict(),
        "args": {
            "in_channels": in_channels,
            "max_level": 3,
            "wave_kernel_size": 16,
            "wavelet_names": ["db6"],
            "use_separate_channel": True,
            "patch_size": patch_size,
            "embed_dim": 128,
            "depth": 2,
            "num_heads": 4,
            "mlp_ratio": 4.0,
            "dropout": 0.1,
            "use_pos_embed": True,
            "pos_embed_type": "2d",
            "pooling": "mean",
            "head_hidden_dim": 128,
            "head_dropout": 0.1,
            "hidden_factor": 1,
        },
    }
    torch.save(ckpt, path)


def main():
    p = argparse.ArgumentParser(description="Generate tiny smoke-test assets for run_kd.py")
    p.add_argument("--out_dir", type=str, default="smoke_assets")
    p.add_argument("--in_channels", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--patch_size", type=int, default=8)
    p.add_argument("--num_classes", type=int, default=4)
    p.add_argument("--n_train", type=int, default=64)
    p.add_argument("--n_val", type=int, default=16)
    p.add_argument("--n_test", type=int, default=16)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    x_train, y_train = make_split(args.n_train, args.in_channels, args.seq_len, args.num_classes, args.seed)
    x_val, y_val = make_split(args.n_val, args.in_channels, args.seq_len, args.num_classes, args.seed + 1)
    x_test, y_test = make_split(args.n_test, args.in_channels, args.seq_len, args.num_classes, args.seed + 2)

    write_h5(os.path.join(args.out_dir, "smoke_train.h5"), x_train, y_train)
    write_h5(os.path.join(args.out_dir, "smoke_val.h5"), x_val, y_val)
    write_h5(os.path.join(args.out_dir, "smoke_test.h5"), x_test, y_test)

    build_teacher_ckpt(
        os.path.join(args.out_dir, "smoke_teacher.pth"),
        in_channels=args.in_channels,
        patch_size=args.patch_size,
        num_classes=args.num_classes,
        seed=args.seed,
    )

    print(f"Wrote smoke assets to: {args.out_dir}")


if __name__ == "__main__":
    main()
