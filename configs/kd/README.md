# KD Experiment Configs

These JSON files encode reproducible `run_kd.py` runs.
For PhysioLite runs, each config sets `student_dataset_profile` to apply the expected
`patch_t/front_pool_k/post_patch_pool_t/student_pos_freqs` combination per dataset.
Teacher selection is controlled by `teacher_model` and `teacher_checkpoint`.
CLI flags can also override `hard_loss`, `save_criterion`, and `deterministic`.

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
