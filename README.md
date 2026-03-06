# PhysioLite

Unified training repo for physiological-signal knowledge distillation experiments (ECG + EMG).

## Single Entrypoint

All KD runs now use one script:

```bash
python -m torch.distributed.run --nproc_per_node=1 --master_port=<PORT> run_kd.py <ARGS>
```

You can also pass a config file:

```bash
python -m torch.distributed.run --nproc_per_node=1 --master_port=<PORT> run_kd.py --config configs/kd/<experiment>.json
```

This single entrypoint supports:
- single-label and multi-label tasks (`--task_type classification|multilabel`)
- all student backbones (`--student_arch ...`)
- configurable teacher backbone for KD (`--teacher_model ...`)
- ablation flags and preprocessing controls (`--student_kernel_set`, `--pp_mode`, `--teacher_logits_h5`, etc.)
- dataset-specific PhysioWaveNPU profiles (`--student_dataset_profile`)
- resolved student config debug print (`--print_student_config`)

Legacy KD scripts were removed.

## Repo Structure

- `run_kd.py`: canonical KD entrypoint
- `kd/runner.py`: main training/eval pipeline
- `kd/common.py`: shared runtime helpers
- `kd/data.py`: dataset wrappers + preprocessing wrappers + teacher-logits wrapper
- `kd/losses.py`: KD losses + CE+SoftF1 criterion
- `kd/schedulers.py`: LR schedulers
- `kd/teacher.py`: teacher model loading/building helpers
- `dataset_multilabel.py`: core HDF5 dataset classes/collate
- `models/physiowave.py`: PhysioWave (BERTWaveletTransformer) model definition used by teacher loader
- `configs/kd/`: versioned experiment configs for reproducible runs
- `data_prep/`: all dataset preprocessing scripts (EMG + ECG)
- `data_prep/README.md`: preprocessing quick-reference
- `scripts/make_smoke_assets.py`: tiny synthetic dataset + teacher checkpoint generator for smoke tests

## Environment

```bash
conda create -n physiolite python=3.11
conda activate physiolite
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Data Format

Expected HDF5 layout for training:
- `data`: `(N, C, T)` float32
- `label`: `(N,)` for classification or `(N, K)` for multilabel

## Data Generation

### EMG datasets

#### 1) UCI EMG for Gestures (single merged script)

```bash
python data_prep/uci_emg_preprocess.py \
  --root_dir ../datasets/UCI_EMG_for_Gestures/EMG_data_for_gestures-master \
  --out_dir ./UCI_EMG_processed_v2 \
  --seq_len 600 \
  --min_len 300 \
  --edge_trim 120 \
  --stride 0 \
  --zscore_per_subject \
  --val_subjects "33-34" \
  --test_subjects "35-36" \
  --save_subject_ids
```

Outputs:
- `UCI_EMG_processed_v2/uci_emg_train.h5`
- `UCI_EMG_processed_v2/uci_emg_val.h5`
- `UCI_EMG_processed_v2/uci_emg_test.h5`

Filtered mode (optional):

```bash
python data_prep/uci_emg_preprocess.py \
  --root_dir ../datasets/UCI_EMG_for_Gestures/EMG_data_for_gestures-master \
  --out_dir ./UCI_EMG_processed_v2_filtered \
  --enable_filtering \
  --fs 1000 \
  --seq_len 600 \
  --val_subjects "33-34" \
  --test_subjects "35-36"
```

#### 2) EPN612

Run:

```bash
python data_prep/epn612_preprocess.py \
  --source_training "./EMG-EPN612 Dataset/trainingJSON" \
  --source_testing "./EMG-EPN612 Dataset/testingJSON" \
  --out_dir "./EPN612_processed" \
  --seq_len 1024
```

Outputs:
- `EPN612_processed/epn612_train_set.h5`
- `EPN612_processed/epn612_val_set.h5`
- `EPN612_processed/epn612_test_set.h5`

### ECG datasets

#### 1) PTB-XL

Run:

```bash
python data_prep/ptbxl_preprocess.py \
  --root "physionet.org/files/ptb-xl/1.0.3/" \
  --threshold 80
```

Outputs (under the same `root` path):
- `train.h5`
- `val.h5`
- `test.h5`

#### 2) NinaPro DB5 (merged filtered/unfiltered script)

```bash
python data_prep/db5_preprocess.py \
  --input_data ../ninapro_db5/raw/ \
  --output_h5 test1_db5 \
  --window_size 512 \
  --stride 64
```

Filtered mode (optional):

```bash
python data_prep/db5_preprocess.py \
  --input_data ../ninapro_db5/raw/ \
  --output_h5 test1_db5_filtered \
  --window_size 512 \
  --stride 64 \
  --enable_filtering
```

#### 3) CPSC / Chapman-Shaoxing

These experiments use pre-existing split H5 files (no converter script included here):
- `CPSC/cpsc_9class_{train,val,test}.h5`
- `data/ChapmanShaoxing/{train,val,test}.h5`

## Experiment Commands (Converted to Single Entrypoint)

These original runs are captured as config files in `configs/kd/`:
- `uci_emg_replicated.json`
- `epn612_replicated.json`
- `ptbxl_replicated.json`
- `cpsc_replicated.json`
- `chapman_replicated.json`
- `chapman_ablation_kset7.json`
- `db5_replicated.json`

Run any config:

```bash
python -m torch.distributed.run --nproc_per_node=1 --master_port=29791 run_kd.py --config configs/kd/uci_emg_replicated.json
```

Teacher backbone selection:

```bash
python -m torch.distributed.run --nproc_per_node=1 --master_port=29791 run_kd.py --config configs/kd/uci_emg_replicated.json --teacher_model physiowave
```

You can also use other integrated backbones as teacher (for example `ecgfm`, `ecgfounder`, `clef`, `hubertecg`, `waveformer`, `otis`, `tinymyo`, `physiowavenpu`) by setting `--teacher_model` and passing a compatible `--teacher_checkpoint` for that architecture.

PhysioWaveNPU dataset profiles (applies `patch_t`, `front_pool_k`, `post_patch_pool_t`, and sets `student_pos_freqs=8`):
- `uci`
- `db5`
- `epn612`
- `ptb`
- `cpsc`
- `chapman`

Default is `--student_dataset_profile auto` (inferred from `--train_file` path).

Example with explicit profile:

```bash
python -m torch.distributed.run --nproc_per_node=1 --master_port=29791 run_kd.py --config configs/kd/uci_emg_replicated.json --student_dataset_profile uci
```

Print resolved student config before model build:

```bash
python -m torch.distributed.run --nproc_per_node=1 --master_port=29791 run_kd.py --config configs/kd/uci_emg_replicated.json --print_student_config
```

Override any config value from CLI (CLI wins):

```bash
python -m torch.distributed.run --nproc_per_node=1 --master_port=29791 run_kd.py --config configs/kd/uci_emg_replicated.json --epochs 5 --output_dir quick_test
```

### UCI EMG

```bash
python -m torch.distributed.run --nproc_per_node=1 --master_port=29791 run_kd.py --train_file UCI_EMG_processed_v2/uci_emg_train.h5 --val_file UCI_EMG_processed_v2/uci_emg_val.h5 --test_file UCI_EMG_processed_v2/uci_emg_test.h5 --teacher_checkpoint multilabel_uci/best_model.pth --in_channels 8 --epochs 50 --lr 2e-4 --weight_decay 1e-3 --threshold 0.3 --use_amp --output_dir multilabel_uci_student_replicated --batch_size 32 --accum_steps 1 --task_type classification --max_length 600 --patch_size 4 --alpha_kd 0.2 --student_arch physiowavenpu --student_dataset_profile uci
```

### EPN612

```bash
python -m torch.distributed.run --nproc_per_node=1 --master_port=29810 run_kd.py --train_file ./EPN612_processed/epn612_train_set.h5 --val_file ./EPN612_processed/epn612_val_set.h5 --test_file ./EPN612_processed/epn612_test_set.h5 --teacher_checkpoint multilabel_epn612/best_model.pth --in_channels 8 --epochs 50 --lr 1e-3 --weight_decay 1e-3 --threshold 0.3 --use_amp --max_length 1024 --patch_size 8 --output_dir multilabel_epn612_student_replicated --batch_size 32 --accum_steps 1 --task_type classification --alpha_kd 0.2 --num_workers 1 --student_arch physiowavenpu --student_dataset_profile epn612
```

### PTB-XL

```bash
python -m torch.distributed.run --nproc_per_node=1 --master_port=29661 run_kd.py --train_file physionet.org/files/ptb-xl/1.0.3/train.h5 --val_file physionet.org/files/ptb-xl/1.0.3/val.h5 --test_file physionet.org/files/ptb-xl/1.0.3/test.h5 --teacher_checkpoint multilabel_ptb/best_model.pth --in_channels 12 --epochs 50 --lr 1e-3 --weight_decay 1e-3 --threshold 0.3 --use_amp --max_length 2048 --patch_size 16 --output_dir multilabel_ptb_student_replicated --batch_size 32 --accum_steps 1 --task_type classification --alpha_kd 0.2 --student_arch physiowavenpu --student_dataset_profile ptb
```

### CPSC

```bash
python -m torch.distributed.run --nproc_per_node=1 --master_port=29593 run_kd.py --train_file CPSC/cpsc_9class_train.h5 --val_file CPSC/cpsc_9class_val.h5 --test_file CPSC/cpsc_9class_test.h5 --teacher_checkpoint multilabel_cpsc/best_model.pth --in_channels 12 --epochs 50 --lr 1e-3 --weight_decay 1e-3 --threshold 0.3 --use_amp --max_length 2048 --output_dir multilabel_cpsc_student_replicated --batch_size 32 --accum_steps 1 --patch_size 16 --alpha_kd 0.20 --student_arch physiowavenpu --student_dataset_profile cpsc
```

### Chapman-Shaoxing

```bash
python -m torch.distributed.run --nproc_per_node=1 --master_port=29511 run_kd.py --train_file ./data/ChapmanShaoxing/train.h5 --val_file ./data/ChapmanShaoxing/val.h5 --test_file ./data/ChapmanShaoxing/test.h5 --teacher_checkpoint multilabel_chapman/best_model.pth --in_channels 12 --epochs 50 --lr 1e-3 --weight_decay 1e-3 --threshold 0.3 --use_amp --max_length 2048 --patch_size 16 --output_dir multilabel_chapman_student_replicated --batch_size 32 --accum_steps 1 --alpha_kd 0.20 --num_workers 1 --student_arch physiowavenpu --student_dataset_profile chapman
```

### Chapman-Shaoxing Ablation Style Run

```bash
python -m torch.distributed.run --nproc_per_node=1 --master_port=29511 run_kd.py --train_file ./data/ChapmanShaoxing/train.h5 --val_file ./data/ChapmanShaoxing/val.h5 --test_file ./data/ChapmanShaoxing/test.h5 --teacher_checkpoint multilabel_chapman/best_model.pth --in_channels 12 --epochs 50 --lr 1e-3 --weight_decay 1e-3 --threshold 0.3 --max_length 2048 --patch_size 16 --output_dir multilabel_chapman_student_replicated_kset7 --batch_size 32 --accum_steps 1 --alpha_kd 0.20 --num_workers 1 --student_kernel_set 7 --student_arch physiowavenpu --student_dataset_profile chapman --pp_mode none --pp_apply_to none --teacher_logits_h5 chapman_logits.h5 --teacher_model physiowave --task_type multilabel
```

### DB5 KD Run

```bash
python -m torch.distributed.run --nproc_per_node=1 --master_port=28881 run_kd.py --train_file test1_db5/db5_train_set.h5 --val_file test1_db5/db5_val_set.h5 --test_file test1_db5/db5_test_set.h5 --teacher_checkpoint multilabel_db5_student_physiowave_new_tiny_1/best_student.pth --task_type classification --student_arch physiowavenpu --alpha_kd 0.2 --in_channels 8 --max_length 512 --patch_size 8 --batch_size 32 --accum_steps 1 --epochs 50 --lr 1e-3 --weight_decay 1e-3 --threshold 0.3 --num_workers 1 --output_dir multilabel_db5_student_kd --scheduler cosine --teacher_logits_h5 test1_db5/db5_train_teacher_logits.h5 --sanity_teacher_test --teacher_model physiowave --student_dataset_profile db5
```

## Smoke Test

Use this to verify end-to-end training/eval wiring with tiny synthetic data.

Generate tiny smoke assets:

```bash
python scripts/make_smoke_assets.py --out_dir smoke_assets
```

Run one-epoch smoke KD:

```bash
python -m torch.distributed.run --nproc_per_node=1 --master_port=29999 run_kd.py --config configs/kd/smoke_classification.json
```

Expected output:
- `smoke_run_output/best_student.pth`
- `smoke_run_output/test_results_kd.json`

## Outputs

Each run writes to `--output_dir`:
- `best_student.pth`
- `test_results_kd.json`

## Useful Help

```bash
python run_kd.py --help
python data_prep/uci_emg_preprocess.py --help
python data_prep/epn612_preprocess.py --help
python data_prep/ptbxl_preprocess.py --help
python data_prep/db5_preprocess.py --help
```
