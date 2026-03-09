from pathlib import Path
import kagglehub

# data_prep/dl_cpsc.py
repo_root = Path(__file__).resolve().parent.parent
outdir = repo_root / "datasets" / "cpsc-2018"

path = kagglehub.dataset_download(
    "bobaaayoung/cpsc-2018",
    output_dir=str(outdir),
)

print("Path to dataset files:", path)
