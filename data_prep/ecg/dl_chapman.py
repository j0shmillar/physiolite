#!/usr/bin/env python3

import argparse
from pathlib import Path

import kagglehub


def main():
    ap = argparse.ArgumentParser(description="Download the Chapman-Shaoxing Kaggle mirror.")
    ap.add_argument("--out_dir", type=str, default="datasets/ChapmanShaoxing")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / args.out_dir
    path = kagglehub.dataset_download(
        "erarayamorenzomuten/chapmanshaoxing-12lead-ecg-database",
        output_dir=str(out_dir),
    )
    print(f"Data available at: {path}")


if __name__ == "__main__":
    main()
