from __future__ import annotations

import os

import torch

from model import BERTWaveletTransformer
from .common import patch_wavelet_modules_io


@torch.no_grad()
def build_teacher_for_kd(ckpt_path, kd_task_type, num_outputs_for_kd, rank=0):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    targs = ckpt.get("args", {}) or {}

    in_channels = int(targs.get("in_channels", 12))
    max_level = int(targs.get("max_level", 3))
    wave_kernel_size = int(targs.get("wave_kernel_size", 16))
    wavelet_names = targs.get("wavelet_names", ["db6"])
    use_sep_ch = bool(targs.get("use_separate_channel", True))
    patch_size_t = int(targs.get("patch_size", 40))
    embed_dim = int(targs.get("embed_dim", 256))
    depth = int(targs.get("depth", 6))
    num_heads = int(targs.get("num_heads", 8))
    mlp_ratio = float(targs.get("mlp_ratio", 4.0))
    dropout = float(targs.get("dropout", 0.1))
    use_pos_embed = bool(targs.get("use_pos_embed", True))
    pos_embed_type = targs.get("pos_embed_type", "2d")
    pooling = targs.get("pooling", "mean")

    head_hidden_dim = targs.get("head_hidden_dim", None)
    head_dropout = float(targs.get("head_dropout", 0.1))
    hidden_factor = int(targs.get("hidden_factor", 2))

    if head_hidden_dim is None:
        k = "task_heads.classification.head.1.weight"
        if k in sd and sd[k].ndim == 2:
            head_hidden_dim = int(sd[k].shape[0])

    head_config = {
        "hidden_dims": [int(head_hidden_dim)] if head_hidden_dim else None,
        "dropout": head_dropout,
        "pooling": pooling,
        "hidden_factor": 1 if head_hidden_dim else hidden_factor,
    }

    if kd_task_type == "multilabel":
        teacher = BERTWaveletTransformer(
            in_channels=in_channels,
            max_level=max_level,
            wave_kernel_size=wave_kernel_size,
            wavelet_names=wavelet_names,
            use_separate_channel=use_sep_ch,
            patch_size=(1, patch_size_t),
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            use_pos_embed=use_pos_embed,
            pos_embed_type=pos_embed_type,
            task_type="multilabel",
            num_labels=num_outputs_for_kd,
            head_config=head_config,
            pooling=pooling,
        )
        want_prefix = "task_heads.multilabel."
        alt_prefix = "task_heads.classification."
    else:
        teacher = BERTWaveletTransformer(
            in_channels=in_channels,
            max_level=max_level,
            wave_kernel_size=wave_kernel_size,
            wavelet_names=wavelet_names,
            use_separate_channel=use_sep_ch,
            patch_size=(1, patch_size_t),
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            use_pos_embed=use_pos_embed,
            pos_embed_type=pos_embed_type,
            task_type="classification",
            num_classes=num_outputs_for_kd,
            head_config=head_config,
            pooling=pooling,
        )
        want_prefix = "task_heads.classification."
        alt_prefix = "task_heads.multilabel."

    missing, unexpected = teacher.load_state_dict(sd, strict=False)
    has_wanted_head_sd = any(k.startswith(want_prefix) for k in sd)

    if not has_wanted_head_sd:
        if rank == 0:
            print(f">> Teacher checkpoint lacks '{want_prefix[:-1]}' head. Loading backbone only.")
        backbone_sd = {k: v for k, v in sd.items() if not k.startswith(want_prefix) and not k.startswith(alt_prefix)}
        missing, unexpected = teacher.load_state_dict(backbone_sd, strict=False)
        if rank == 0:
            print(f"   Loaded backbone. Missing={len(missing)}, Unexpected={len(unexpected)}")
        return teacher, False

    head_loaded = not any(k.startswith(want_prefix) for k in missing)
    if rank == 0:
        print(
            f"   Loaded teacher SD. Missing={len(missing)}, Unexpected={len(unexpected)} "
            f"(matching head loaded: {head_loaded})"
        )
    return teacher, head_loaded


def load_physiowave():
    rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{rank}")
    head_hidden_dim = 512
    head_dropout = 0.1
    pooling = "mean"
    in_channels = 8
    max_level = 3
    wave_kernel_size = 16
    wavelet_names = ["sym4", "sym5", "db6", "coif3", "bior4.4"]
    use_separate_channel = True
    patch_size = 64
    embed_dim = 256
    depth = 4
    num_heads = 8
    mlp_ratio = 4.0
    dropout = 0.1
    use_pos_embed = True
    pos_embed_type = "2d"
    num_classes = 53
    freeze_encoder = False

    head_config = {
        "hidden_dims": [head_hidden_dim],
        "dropout": head_dropout,
        "pooling": pooling,
    }

    model = BERTWaveletTransformer(
        in_channels=in_channels,
        max_level=max_level,
        wave_kernel_size=wave_kernel_size,
        wavelet_names=wavelet_names,
        use_separate_channel=use_separate_channel,
        patch_size=(1, patch_size),
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        use_pos_embed=use_pos_embed,
        pos_embed_type=pos_embed_type,
        task_type="classification",
        num_classes=num_classes,
        head_config=head_config,
        pooling=pooling,
    ).to(device)

    if hasattr(model, "initialize_weights"):
        model.initialize_weights()
        print("Initialized model weights")

    if freeze_encoder:
        for name, param in model.named_parameters():
            if "task_heads" not in name:
                param.requires_grad = False
        print("Frozen encoder parameters (excluding task heads)")

    return model, None


@torch.no_grad()
def load_teacher_model_for_eval(args, device: torch.device, num_outputs: int, rank: int):
    teacher, _ = build_teacher_for_kd(args.teacher_checkpoint, args.task_type, num_outputs, rank=rank)
    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    patch_wavelet_modules_io(teacher, rank=rank)
    return teacher, "bertwavelet"
