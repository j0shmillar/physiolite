# PhysioLite

Unified KD training for physiological signals.

## Environment

```bash
conda create -n physiolite python=3.11
conda activate physiolite
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## KD Entrypoint

```bash
python -m torch.distributed.run --nproc_per_node=1 --master_port=<PORT> run_kd.py <ARGS>
```

Useful runtime options:
- `--teacher_model physiowave|physiowavenpu`
- `--teacher_logits_h5 <path>`
- `--hard_loss ce|ce_softf1|bce`
- `--save_criterion loss|f1_macro|f1_weighted|accuracy`
- `--deterministic`

## Dataset Setup

Canonical preprocessors live under `data_prep/emg` and `data_prep/ecg`.
Legacy `*_preprocess.py` paths still forward to the same logic.

### DB5

Download:

```bash
python data_prep/emg/dl_db5.py
```

Preprocess:

```bash
python data_prep/emg/db5.py --input_data datasets/ninapro_db5/ --output_h5 datasets/db5 --window_size 512 --stride 64
```

Preprocess with filtering:

```bash
python data_prep/emg/db5.py --input_data datasets/ninapro_db5/ --output_h5 datasets/db5_filtered --window_size 512 --stride 64 --enable_filtering
```

### UCI EMG

Download:

```bash
cd datasets
wget https://archive.ics.uci.edu/static/public/481/emg+data+for+gestures.zip
unzip emg+data+for+gestures.zip
```

Preprocess:

```bash
python data_prep/uci_emg_preprocess.py   --root_dir datasets/EMG_data_for_gestures-master   --out_dir datasets/UCI   --seq_len 1024   --min_len 1024   --edge_trim 120   --stride 512   --zscore_per_subject   --val_subjects "33-34"   --test_subjects "35-36"   --save_subject_ids
```

Preprocess with filtering:

```bash
python data_prep/uci_emg_preprocess.py   --root_dir datasets/EMG_data_for_gestures-master   --out_dir datasets/UCI_filtered   --seq_len 1024   --min_len 1024   --edge_trim 120   --stride 512   --zscore_per_subject   --val_subjects "33-34"   --test_subjects "35-36"   --save_subject_ids --enable_filtering
```

### EPN-612

Download:

```bash
cd datasets
wget https://zenodo.org/records/4421500/files/EMG-EPN612%20Dataset.zip?download=1
mv 'EMG-EPN612 Dataset.zip?download=1' EMG-EPN612.zip
unzip EMG-EPN612.zip
```

Preprocess:

```bash
python data_prep/emg/epn612.py --source_training "datasets/EMG-EPN612 Dataset/trainingJSON" --source_testing "datasets/EMG-EPN612 Dataset/testingJSON" --out_dir "datasets/EPN612" --seq_len 1024
```

Preprocess with filtering:

```bash
python data_prep/emg/epn612.py --source_training "datasets/EMG-EPN612 Dataset/trainingJSON" --source_testing "datasets/EMG-EPN612 Dataset/testingJSON" --out_dir "datasets/EPN612_filtered" --seq_len 1024 --enable_filtering
```

### PTB-XL

Download:

```bash
cd datasets/
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/
```

Preprocess:

```bash
python data_prep/ptbxl_preprocess.py --root datasets/physionet.org/files/ptb-xl/1.0.3/ --threshold 80.0 --window_size 2048 --step_size 1024
```

Preprocess with filtering:

```bash
python data_prep/ptbxl_preprocess.py \
  --root datasets/physionet.org/files/ptb-xl/1.0.3/ \
  --threshold 80.0 \
  --window_size 2048 \
  --step_size 1024 \
  --enable_filtering \
  --fs 500 \
  --band_low 0.67 \
  --band_high 40.0 \
  --band_order 4 \
  --notch_freq 50.0 \
  --notch_q 30.0 \
  --baseline_kernel_sec 0.4
```

### CPSC

Download:

```bash
python data_prep/ecg/dl_cpsc.py
```

Preprocess:

```bash
python data_prep/ecg/cpsc.py
```

Preprocess with filtering:

```bash
python data_prep/ecg/cpsc.py --enable_filtering
```

### Chapman-Shaoxing

Download:

```bash
python data_prep/ecg/dl_chapman.py
```

Preprocess:

```bash
python data_prep/ecg/chapman.py --root_dir datasets/chapman-shaoxing --out_dir datasets/ChapmanShaoxing
```

Preprocess with filtering:

```bash
python data_prep/ecg/chapman.py --root_dir datasets/chapman-shaoxing --out_dir datasets/ChapmanShaoxing_filtered --enable_filtering
```

## Benchmark

Below, we give some example commands to evaluate PhysioLite on the range of datasets used. 

### DB5

```bash
python -m torch.distributed.run --nproc_per_node=1 --master_port=12345 run_kd.py --train_file datasets/db5/db5_train_set.h5 --val_file datasets/db5/db5_val_set.h5 --test_file datasets/db5/db5_test_set.h5 --teacher_checkpoin pwave/db5.pth  --task_type classification --student_arch physiowavenpu --teacher_model physiowave --alpha_kd 0.5  --in_channels 8 --max_length 512 --patch_size 8 --batch_size 32 --accum_steps 1 --epochs 150 --lr 1e-3 --weight_decay 1e-3 --threshold 0.3 --num_workers 1 --output_dir out --scheduler cosine --teacher_logits_h5 pwave/db5_logits.h5 --student_dataset_profile db5 --hard_loss ce_softf1 --save_criterion f1_macro
```

### UCI

```bash
python -m torch.distributed.run --nproc_per_node=1 --master_port=12345 run_kd.py --train_file datasets/UCI/uci_emg_train.h5 --val_file datasets/UCI/uci_emg_val.h5 --test_file datasets/UCI/uci_emg_test.h5 --teacher_checkpoint pwave/uci.pth --teacher_model physiowave --task_type classification --in_channels 8 --epochs 50 --lr 2e-4 --weight_decay 1e-3 --threshold 0.3 --use_amp --output_dir out --batch_size 64 --accum_steps 1 --task_type classification --max_length 1024 --patch_size 4 --alpha_kd 0.5 --student_arch physiowavenpu --student_dataset_profile uci --print_student_config
```

### EPN-612

```bash
python -m torch.distributed.run --nproc_per_node=1 --master_port=12345 run_kd.py --train_file ./datasets/EPN612/epn612_train_set.h5 --val_file ./datasets/EPN612/epn612_val_set.h5 --test_file ./datasets/EPN612/epn612_test_set.h5 --teacher_checkpoint pwave/epn612.pth --in_channels 8 --epochs 50 --lr 1e-3 --weight_decay 1e-3 --threshold 0.3 --use_amp --max_length 1024 --patch_size 8 --output_dir out --batch_size 32 --accum_steps 1 --task_type classification --alpha_kd 0.5 --num_workers 1 --student_arch physiowavenpu --student_dataset_profile none --teacher_logits_h5 pwave/epn612_logits.h5 --student_post_patch_pool_t 4 --student_front_pool_k 4
```

### PTB

```bash
python -m torch.distributed.run --nproc_per_node=1 --master_port=12345 run_kd.py --train_file datasets/physionet.org/files/ptb-xl/1.0.3/train.h5 --val_file datasets/physionet.org/files/ptb-xl/1.0.3/val.h5 --test_file datasets/physionet.org/files/ptb-xl/1.0.3/test.h5 --teacher_checkpoint pwave/ptb.pth   --in_channels 12   --epochs 50   --lr 1e-3   --weight_decay 1e-3   --threshold 0.3   --use_amp   --max_length 2048   --patch_size 16    --output_dir out --batch_size 1 --accum_steps 1  --task_type classification  --alpha_kd 0.5 
```

### CPSC

```bash
python -m torch.distributed.run --nproc_per_node=1 --master_port=12345 run_kd.py --train_file datasets/CPSC/cpsc_9class_train.h5 --val_file datasets/CPSC/cpsc_9class_val.h5 --test_file datasets/CPSC/cpsc_9class_test.h5 --teacher_checkpoint pwave/cpsc.pth --in_channels 12 --epochs 50 --lr 1e-3 --weight_decay 1e-3 --threshold 0.3 --use_amp --max_length 2048 --output_dir out --batch_size 32 --accum_steps 1 --patch_size 16 --alpha_kd 0.5 --student_arch physiowavenpu --student_dataset_profile cpsc --teacher_logits_h5 pwave/cpsc_logits.h5 --task_type multilabel --hard_loss bce
```

### CHAPMAN

```bash
python -m torch.distributed.run --nproc_per_node=1 --master_port=12345 run_kd.py --train_file datasets/ChapmanShaoxing/train.h5 --val_file datasets/ChapmanShaoxing/val.h5 --test_file datasets/ChapmanShaoxing/test.h5 --teacher_checkpoint pwave/chapman.pth --in_channels 12 --epochs 50 --lr 1e-3 --weight_decay 1e-3 --threshold 0.3 --use_amp --max_length 2048 --patch_size 16 --output_dir out --batch_size 32 --accum_steps 1 --alpha_kd 0.5 --num_workers 1 --student_arch physiowavenpu --student_dataset_profile chapman --teacher_logits_h5 pwave/chapman_logits.h5 --task_type multilabel  --hard_loss bce
```

