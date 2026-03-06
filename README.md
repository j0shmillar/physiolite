# PhysioLite

Unified training repo for physiological-signal knowledge distillation experiments (ECG + EMG).

## Single Entrypoint

All KD runs now use one script:

```bash
python -m torch.distributed.run --nproc_per_node=1 --master_port=<PORT> run_kd.py <ARGS>
```

This single entrypoint supports:
- single-label and multi-label tasks (`--task_type classification|multilabel`)
- all student backbones (`--student_arch ...`)
- ablation flags and preprocessing controls (`--student_kernel_set`, `--pp_mode`, `--teacher_logits_h5`, etc.)

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
- `data_prep/`: all dataset preprocessing scripts (EMG + ECG)

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

### UCI EMG

```bash
python -m torch.distributed.run --nproc_per_node=1 --master_port=29791 run_kd.py --train_file UCI_EMG_processed_v2/uci_emg_train.h5 --val_file UCI_EMG_processed_v2/uci_emg_val.h5 --test_file UCI_EMG_processed_v2/uci_emg_test.h5 --teacher_checkpoint multilabel_uci/best_model.pth --in_channels 8 --epochs 50 --lr 2e-4 --weight_decay 1e-3 --threshold 0.3 --use_amp --output_dir multilabel_uci_student_replicated --batch_size 32 --accum_steps 1 --task_type classification --max_length 600 --patch_size 4 --alpha_kd 0.2 --student_arch physiowavenpu
```

### EPN612

```bash
python -m torch.distributed.run --nproc_per_node=1 --master_port=29810 run_kd.py --train_file ./EPN612_processed/epn612_train_set.h5 --val_file ./EPN612_processed/epn612_val_set.h5 --test_file ./EPN612_processed/epn612_test_set.h5 --teacher_checkpoint multilabel_epn612/best_model.pth --in_channels 8 --epochs 50 --lr 1e-3 --weight_decay 1e-3 --threshold 0.3 --use_amp --max_length 1024 --patch_size 8 --output_dir multilabel_epn612_student_replicated --batch_size 32 --accum_steps 1 --task_type classification --alpha_kd 0.2 --num_workers 1 --student_arch physiowavenpu
```

### PTB-XL

```bash
python -m torch.distributed.run --nproc_per_node=1 --master_port=29661 run_kd.py --train_file physionet.org/files/ptb-xl/1.0.3/train.h5 --val_file physionet.org/files/ptb-xl/1.0.3/val.h5 --test_file physionet.org/files/ptb-xl/1.0.3/test.h5 --teacher_checkpoint multilabel_ptb/best_model.pth --in_channels 12 --epochs 50 --lr 1e-3 --weight_decay 1e-3 --threshold 0.3 --use_amp --max_length 2048 --patch_size 16 --output_dir multilabel_ptb_student_replicated --batch_size 32 --accum_steps 1 --task_type classification --alpha_kd 0.2 --student_arch physiowavenpu
```

### CPSC

```bash
python -m torch.distributed.run --nproc_per_node=1 --master_port=29593 run_kd.py --train_file CPSC/cpsc_9class_train.h5 --val_file CPSC/cpsc_9class_val.h5 --test_file CPSC/cpsc_9class_test.h5 --teacher_checkpoint multilabel_cpsc/best_model.pth --in_channels 12 --epochs 50 --lr 1e-3 --weight_decay 1e-3 --threshold 0.3 --use_amp --max_length 2048 --output_dir multilabel_cpsc_student_replicated --batch_size 32 --accum_steps 1 --patch_size 16 --alpha_kd 0.20 --student_arch physiowavenpu
```

### Chapman-Shaoxing

```bash
python -m torch.distributed.run --nproc_per_node=1 --master_port=29511 run_kd.py --train_file ./data/ChapmanShaoxing/train.h5 --val_file ./data/ChapmanShaoxing/val.h5 --test_file ./data/ChapmanShaoxing/test.h5 --teacher_checkpoint multilabel_chapman/best_model.pth --in_channels 12 --epochs 50 --lr 1e-3 --weight_decay 1e-3 --threshold 0.3 --use_amp --max_length 2048 --patch_size 16 --output_dir multilabel_chapman_student_replicated --batch_size 32 --accum_steps 1 --alpha_kd 0.20 --num_workers 1 --student_arch physiowavenpu
```

### Chapman-Shaoxing Ablation Style Run

```bash
python -m torch.distributed.run --nproc_per_node=1 --master_port=29511 run_kd.py --train_file ./data/ChapmanShaoxing/train.h5 --val_file ./data/ChapmanShaoxing/val.h5 --test_file ./data/ChapmanShaoxing/test.h5 --teacher_checkpoint multilabel_chapman/best_model.pth --in_channels 12 --epochs 50 --lr 1e-3 --weight_decay 1e-3 --threshold 0.3 --max_length 2048 --patch_size 16 --output_dir multilabel_chapman_student_replicated_kset7 --batch_size 32 --accum_steps 1 --alpha_kd 0.20 --num_workers 1 --student_kernel_set 7 --student_arch physiowavenpu --pp_mode none --pp_apply_to none --teacher_logits_h5 chapman_logits.h5 --teacher_arch physiowave --task_type multilabel
```

### DB5 KD Run

```bash
python -m torch.distributed.run --nproc_per_node=1 --master_port=28881 run_kd.py --train_file test1_db5/db5_train_set.h5 --val_file test1_db5/db5_val_set.h5 --test_file test1_db5/db5_test_set.h5 --teacher_checkpoint multilabel_db5_student_physiowave_new_tiny_1/best_student.pth --task_type classification --student_arch physiowavenpu --alpha_kd 0.2 --in_channels 8 --max_length 512 --patch_size 8 --batch_size 32 --accum_steps 1 --epochs 50 --lr 1e-3 --weight_decay 1e-3 --threshold 0.3 --num_workers 1 --output_dir multilabel_db5_student_kd --scheduler cosine --teacher_logits_h5 test1_db5/db5_train_teacher_logits.h5 --sanity_teacher_test --teacher_arch physiowave
```

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
