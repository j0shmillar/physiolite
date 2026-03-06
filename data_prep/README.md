# Data Preprocessing

Canonical preprocessing scripts live in `data_prep/`.

Use that directory for all dataset generation workflows:
- `data_prep/uci_emg_preprocess.py`
- `data_prep/db5_preprocess.py`
- `data_prep/epn612_preprocess.py`
- `data_prep/ptbxl_preprocess.py`

For full commands and EMG/ECG split instructions, see:
- `data_prep/README.md`

Quick help:

```bash
python data_prep/uci_emg_preprocess.py --help
python data_prep/db5_preprocess.py --help
python data_prep/epn612_preprocess.py --help
python data_prep/ptbxl_preprocess.py --help
```
