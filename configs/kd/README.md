# KD Experiment Configs

These JSON files encode the original paper experiment runs for `run_kd.py`.

Available configs:
- `uci_emg_replicated.json`
- `epn612_replicated.json`
- `ptbxl_replicated.json`
- `cpsc_replicated.json`
- `chapman_replicated.json`
- `chapman_ablation_kset7.json`
- `db5_replicated.json`
- `smoke_classification.json`

Run:

```bash
python -m torch.distributed.run --nproc_per_node=1 --master_port=29791 run_kd.py --config configs/kd/uci_emg_replicated.json
```

Override values from CLI:

```bash
python -m torch.distributed.run --nproc_per_node=1 --master_port=29791 run_kd.py --config configs/kd/uci_emg_replicated.json --epochs 5 --output_dir quick_test
```
