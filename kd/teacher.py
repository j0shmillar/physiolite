from __future__ import annotations

import os

import torch

from models.physiowave import BERTWaveletTransformer


def _extract_teacher_checkpoint_state_dict(ckpt_obj):
    return ckpt_obj.get("model_state_dict") or ckpt_obj.get("state_dict") or ckpt_obj.get("student_state_dict") or ckpt_obj


@torch.no_grad()
def build_teacher_for_kd(ckpt_path, kd_task_type, num_outputs_for_kd, rank=0):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("student_state_dict") or ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt

    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
        sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}

    cls_head_keys = [k for k in sd.keys() if k.startswith("task_heads.classification.")]
    ml_head_keys = [k for k in sd.keys() if k.startswith("task_heads.multilabel.")]

    if rank == 0:
        print(f">> Found classification head keys: {len(cls_head_keys)}")
        print(f">> Found multilabel head keys: {len(ml_head_keys)}")

    targs = ckpt.get("args", {}) or {}

    # Infer backbone shape from checkpoint tensors first.
    in_channels = int(targs.get("in_channels", 8))

    patch_size_t = 64
    if "patch_embed.proj.weight" in sd:
        patch_size_t = int(sd["patch_embed.proj.weight"].shape[-1])

    embed_dim = 256
    if "patch_embed.proj.weight" in sd:
        embed_dim = int(sd["patch_embed.proj.weight"].shape[0])

    num_wavelets = 5
    if "wavelet_decomp.selector.selector.4.weight" in sd:
        num_wavelets = int(sd["wavelet_decomp.selector.selector.4.weight"].shape[0])

    all_wavelets = ["sym4", "sym5", "db6", "coif3", "bior4.4"]
    wavelet_names = all_wavelets[:num_wavelets]

    # Keep conservative defaults for parts not easy to infer.
    max_level = int(targs.get("max_level", 3))
    wave_kernel_size = int(targs.get("wave_kernel_size", 16))
    use_separate_channel = bool(targs.get("use_separate_channel", True))
    depth = int(targs.get("depth", 4))
    num_heads = int(targs.get("num_heads", 8))
    mlp_ratio = float(targs.get("mlp_ratio", 4.0))
    dropout = float(targs.get("dropout", 0.1))
    use_pos_embed = bool(targs.get("use_pos_embed", True))
    pos_embed_type = targs.get("pos_embed_type", "2d")
    pooling = str(targs.get("pooling", "mean"))

    # Infer head hidden dim from the checkpoint task head.
    head_hidden_dim = None
    if kd_task_type == "classification":
        if "task_heads.classification.head.1.weight" in sd:
            head_hidden_dim = int(sd["task_heads.classification.head.1.weight"].shape[0])
    else:
        if "task_heads.multilabel.head.1.weight" in sd:
            head_hidden_dim = int(sd["task_heads.multilabel.head.1.weight"].shape[0])

    # Fallbacks
    if head_hidden_dim is None:
        head_hidden_dim = int(targs.get("head_hidden_dim", 512))

    head_config = {
        "hidden_dims": [head_hidden_dim] if head_hidden_dim else None,
        "dropout": float(targs.get("head_dropout", 0.1)),
        "pooling": pooling,
        "hidden_factor": 1 if head_hidden_dim else int(targs.get("hidden_factor", 2)),
    }

    if kd_task_type == "multilabel":
        teacher = BERTWaveletTransformer(
            in_channels=in_channels,
            max_level=max_level,
            wave_kernel_size=wave_kernel_size,
            wavelet_names=wavelet_names,
            use_separate_channel=use_separate_channel,
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
    else:
        teacher = BERTWaveletTransformer(
            in_channels=in_channels,
            max_level=max_level,
            wave_kernel_size=wave_kernel_size,
            wavelet_names=wavelet_names,
            use_separate_channel=use_separate_channel,
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

    model_sd = teacher.state_dict()
    filtered_sd = {}
    shape_mismatch = []

    for k, v in sd.items():
        if k not in model_sd:
            continue
        if tuple(v.shape) != tuple(model_sd[k].shape):
            shape_mismatch.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
            continue
        filtered_sd[k] = v

    missing, unexpected = teacher.load_state_dict(filtered_sd, strict=False)

    if rank == 0:
        print(
            f">> Loaded teacher SD. matched={len(filtered_sd)} "
            f"missing={len(missing)} unexpected={len(unexpected)} "
            f"shape_mismatch={len(shape_mismatch)}"
        )
        if shape_mismatch:
            for k, a, b in shape_mismatch[:10]:
                print(f"   shape mismatch: {k}: ckpt={a}, model={b}")

    head_loaded = any(k.startswith(want_prefix) for k in filtered_sd.keys())
    if rank == 0:
        print(f">> Matching teacher head loaded: {head_loaded}")

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
