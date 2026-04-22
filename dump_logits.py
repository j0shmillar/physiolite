import argparse
import os
import torch
import h5py
from tqdm import tqdm
from torch.utils.data import DataLoader

from timeseries_ds import (
    SingleLabelTimeSeriesDataset,
    MultiLabelTimeSeriesDataset,
    parse_file_paths,
    collate_singlelabel_fn,
    collate_multilabel_fn,
)

from kd.runner import build_teacher_model, forward_model_logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_file", required=True, help="Your train .h5 path (or your file list handled by parse_file_paths)")
    ap.add_argument("--teacher_checkpoint", required=True, help="Path to teacher checkpoint (same as KD script)")
    ap.add_argument("--out_h5", required=True, help="Where to write logits (HDF5)")
    ap.add_argument("--task_type", default="classification", choices=["classification", "multilabel"])
    ap.add_argument("--teacher_model", default="physiowave", choices=["physiowave", "physiolite"])
    ap.add_argument("--teacher_dataset_profile", default="auto",
                    choices=["none", "auto", "uci", "db5", "epn612", "ptb", "cpsc", "chapman"])
    ap.add_argument("--data_key", default="data")
    ap.add_argument("--label_key", default="label")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--in_channels", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--amp_dtype", default="bf16", choices=["bf16", "fp16", "none"])
    args = ap.parse_args()

    device = torch.device(args.device)

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

    build_args = argparse.Namespace(
        teacher_model=args.teacher_model,
        teacher_checkpoint=args.teacher_checkpoint,
        teacher_dataset_profile=args.teacher_dataset_profile,
        student_dataset_profile=args.teacher_dataset_profile,
        train_file=args.data_file,
        task_type=args.task_type,
        in_channels=args.in_channels,
        max_length=args.max_length,
        patch_size=8,
        student_depth=3,
        student_embed_dim=256,
        student_bank_ch=16,
        student_reduce=4,
        student_pos_freqs=8,
        student_front_pool_k=4,
        student_post_patch_pool_t=5,
        student_kernel_set="3,5,7",
        print_student_config=False,
        teacher_pe_scale=0.1,
        student_pe_scale=0.1,
    )
    teacher, teacher_pe_cache = build_teacher_model(build_args, ds, device, rank=0, kd_outputs=num_out)
    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

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
                        t = forward_model_logits(
                            teacher,
                            x,
                            arch=args.teacher_model,
                            args=build_args,
                            pe_cache=teacher_pe_cache,
                            pe_scale=build_args.teacher_pe_scale,
                        )
                else:
                    t = forward_model_logits(
                        teacher,
                        x,
                        arch=args.teacher_model,
                        args=build_args,
                        pe_cache=teacher_pe_cache,
                        pe_scale=build_args.teacher_pe_scale,
                    )

                t = t.detach().to("cpu", dtype=torch.float16).numpy()
                b = t.shape[0]
                logits_ds[idx:idx + b] = t
                idx += b

    print(f"\nWrote: {args.out_h5}")
    print(f"Dataset: /teacher_logits  shape=({len(ds)},{num_out}) dtype=float16")


if __name__ == "__main__":
    main()
