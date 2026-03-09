from pathlib import Path
import kagglehub

# data_prep/dl_cpsc.py
repo_root = Path(__file__).resolve().parent.parent
outdir = repo_root / "datasets" / "chapman-shaoxing"

path = kagglehub.dataset_download(
    "erarayamorenzomuten/chapmanshaoxing-12lead-ecg-database",
    output_dir=str(outdir),
)

print("Path to dataset files:", path)
