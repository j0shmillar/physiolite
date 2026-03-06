# cache_teacher_logits.py
import os, math, argparse, torch
from torch.utils.data import DataLoader, SequentialSampler
from contextlib import nullcontext

# --- import your project code ---
from kd.teacher import build_teacher_for_kd
from dataset_multilabel import (
    MultiLabelTimeSeriesDataset,
    SingleLabelTimeSeriesDataset,
    collate_multilabel_fn,
    collate_singlelabel_fn,
)
# --------------------------------

def _load_ds(split_path, task_type, max_length, data_key, label_key):
    if not split_path:
        return None, None
    files = [split_path]  # your parse_file_paths() supports comma lists; one is fine here
    if task_type == 'multilabel':
        ds = MultiLabelTimeSeriesDataset(files, max_length=max_length, data_key=data_key, label_key=label_key)
        collate = collate_multilabel_fn
    else:
        ds = SingleLabelTimeSeriesDataset(files, max_length=max_length, data_key=data_key, label_key=label_key)
        collate = collate_singlelabel_fn
    return ds, collate

@torch.no_grad()
def dump_split(split_name, ds, collate, teacher, device, batch_size, use_amp, out_dir):
    if ds is None: 
        return None
    os.makedirs(out_dir, exist_ok=True)
    sampler = SequentialSampler(ds)  # fixed order that we’ll reuse later
    loader = DataLoader(ds, batch_size=batch_size, sampler=sampler, collate_fn=collate,
                        num_workers=2, pin_memory=True)

    logits_chunks = []
    amp_ctx = torch.cuda.amp.autocast if use_amp else nullcontext

    teacher.eval().half()  # reduce VRAM and I/O
    for x, _y in loader:
        x = x.to(device, non_blocking=True).half()
        with torch.inference_mode(), amp_ctx():
            # IMPORTANT: feed *raw* x to the teacher (no student PE concat)
            t_logits = teacher(x, task='downstream', task_name=args.task_type, return_logits=True)
        logits_chunks.append(t_logits.detach().cpu().to(torch.float16))

    logits = torch.cat(logits_chunks, dim=0)  # [N, C]
    out_path = os.path.join(out_dir, f"{split_name}_teacher_logits_fp16.pt")
    meta = dict(
        split=split_name, dtype="float16", num_samples=len(ds), num_labels=ds.num_classes
    )
    torch.save({"logits": logits, "meta": meta}, out_path)
    print(f"[{split_name}] saved {tuple(logits.shape)} -> {out_path}")
    return out_path

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # data
    p.add_argument('--train_file', type=str, required=True)
    p.add_argument('--val_file',   type=str, required=True)
    p.add_argument('--test_file',  type=str, default="")
    p.add_argument('--data_key',   type=str, default='data')
    p.add_argument('--label_key',  type=str, default='label')
    p.add_argument('--max_length', type=int, default=1024)
    p.add_argument('--task_type',  type=str, default='multilabel', choices=['multilabel','classification'])
    # teacher
    p.add_argument('--teacher_checkpoint', type=str, required=True)
    # loader/compute
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--use_amp', action='store_true')
    p.add_argument('--out_dir', type=str, default='cached_logits')
    args = p.parse_args()

    device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')

    # Build datasets (to know label dim and input length)
    train_ds, train_collate = _load_ds(args.train_file, args.task_type, args.max_length, args.data_key, args.label_key)
    val_ds,   val_collate   = _load_ds(args.val_file,   args.task_type, args.max_length, args.data_key, args.label_key)
    test_ds,  test_collate  = _load_ds(args.test_file,  args.task_type, args.max_length, args.data_key, args.label_key)

    kd_outputs = train_ds.num_classes

    # Build teacher for the KD task size
    teacher, _ = build_teacher_for_kd(
        args.teacher_checkpoint,
        kd_task_type=args.task_type,
        num_outputs_for_kd=kd_outputs,
        rank=0
    )
    teacher = teacher.to(device).eval()

    # Dump all splits
    dump_split("train", train_ds, train_collate, teacher, device, args.batch_size, args.use_amp, args.out_dir)
    dump_split("val",   val_ds,   val_collate,   teacher, device, args.batch_size, args.use_amp, args.out_dir)
    if test_ds is not None:
        dump_split("test",  test_ds,  test_collate,  teacher, device, args.batch_size, args.use_amp, args.out_dir)

    print("Done.")
