# dump_teacher_logits.py
import argparse
import os
import torch
import h5py
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset_multilabel import (
    SingleLabelTimeSeriesDataset,
    MultiLabelTimeSeriesDataset,
    parse_file_paths,
    collate_singlelabel_fn,
    collate_multilabel_fn,
)

# Reuse teacher loading utilities from unified KD package.
from kd.teacher import build_teacher_for_kd, load_physiowave, load_teacher_model_for_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_file", required=True, help="Your train .h5 path (or your file list handled by parse_file_paths)")
    ap.add_argument("--teacher_checkpoint", required=True, help="Path to teacher checkpoint (same as KD script)")
    ap.add_argument("--out_h5", required=True, help="Where to write logits (HDF5)")
    ap.add_argument("--task_type", default="classification", choices=["classification", "multilabel"])
    ap.add_argument("--data_key", default="data")
    ap.add_argument("--label_key", default="label")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--amp_dtype", default="bf16", choices=["bf16", "fp16", "none"])
    args = ap.parse_args()

    device = torch.device(args.device)

    # speed knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    files = parse_file_paths(args.data_file)

    if args.task_type == "multilabel":
        ds = MultiLabelTimeSeriesDataset(files, max_length=args.max_length, data_key=args.data_key, label_key=args.label_key)
        collate_fn = collate_multilabel_fn
        num_out = ds.num_classes
    else:
        ds = SingleLabelTimeSeriesDataset(files, max_length=args.max_length, data_key=args.data_key, label_key=args.label_key)
        collate_fn = collate_singlelabel_fn
        num_out = ds.num_classes

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        collate_fn=collate_fn,
    )

    # Build teacher

    ckpt = torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=False)
    #teacher, _ = load_physiowave()  # builds the exact PhysioWave architecture you used
    sd = ckpt.get("student_state_dict") or ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    #teacher.load_state_dict(sd, strict=True)
    #teacher, _ = build_teacher_for_kd(args.teacher_checkpoint, "multilabel", num_out, rank=0)
    teacher, _kind = load_teacher_model_for_eval(
        args=argparse.Namespace(
            teacher_model="physiowave",
            teacher_checkpoint=args.teacher_checkpoint,
            task_type="classification",           # "multilabel" or "classification"
            waveformer_patch_width=64,          # unused here; keep something
            amp_dtype=args.amp_dtype,
            threshold=0.5,
            sanity_teacher_test=False,
        ),
        device=device,
        num_outputs=num_out,
        rank=0,
    )
    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # AMP setup
    use_amp = (device.type == "cuda") and (args.amp_dtype != "none")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    os.makedirs(os.path.dirname(os.path.abspath(args.out_h5)), exist_ok=True)

    with h5py.File(args.out_h5, "w") as f:
        logits_ds = f.create_dataset(
            "teacher_logits",
            shape=(len(ds), num_out),
            dtype="float16",
            chunks=True,
        )

        idx = 0
        pbar = tqdm(loader, desc="Dump teacher logits", ncols=120)
        with torch.inference_mode():
            for x, y in pbar:
                x = x.to(device, non_blocking=True)

                if use_amp:
                    with torch.amp.autocast("cuda", dtype=amp_dtype):
                        t = teacher(x, task="downstream", task_name="classification", return_logits=True)
                else:
                    t = teacher(x, task="downstream", task_name=args.task_type, return_logits=True)

                t = t.detach().to("cpu", dtype=torch.float16).numpy()
                b = t.shape[0]
                logits_ds[idx:idx + b] = t
                idx += b

    print(f"\nWrote: {args.out_h5}")
    print(f"Dataset: /teacher_logits  shape=({len(ds)},{num_out}) dtype=float16")


if __name__ == "__main__":
    main()
