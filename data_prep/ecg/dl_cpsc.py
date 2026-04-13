#!/usr/bin/env python3

import argparse
from pathlib import Path

import kagglehub


def main():
    ap = argparse.ArgumentParser(description="Download the CPSC 2018 Kaggle mirror.")
    ap.add_argument("--out_dir", type=str, default="datasets/cpsc-2018")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / args.out_dir
    path = kagglehub.dataset_download("bobaaayoung/cpsc-2018", output_dir=str(out_dir))
    print(f"Downloaded CPSC dataset to: {path}")


if __name__ == "__main__":
    main()
