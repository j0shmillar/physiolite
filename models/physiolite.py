import math
import torch
import torch.nn as nn


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


class Conv1dRF7(nn.Module):
    """Approximate RF=7 via three 3x1 Conv1d layers."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        mid = out_ch
        if in_ch > 64:
            raise AssertionError(
                f"Conv1dRF7: in_ch={in_ch} > 64 would violate ai8x padding rule (padding>0 not allowed)."
            )
        self.c1 = nn.Conv1d(in_ch, mid, kernel_size=3, padding=1, bias=False)
        self.a1 = nn.ReLU(inplace=True)
        self.c2 = nn.Conv1d(mid, mid, kernel_size=3, padding=1, bias=False)
        self.a2 = nn.ReLU(inplace=True)
        self.c3 = nn.Conv1d(mid, out_ch, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = self.a1(self.c1(x))
        x = self.a2(self.c2(x))
        return self.c3(x)


class FrontDownsample1D(nn.Module):
    """AvgPool1d(k, k) front downsample."""

    def __init__(self, k: int):
        super().__init__()
        assert 1 <= k <= 16, f"front pool k={k} must be in [1..16] on MAX78000"
        self.pool = nn.AvgPool1d(kernel_size=k, stride=k)

    def forward(self, x):  # x: [B, C, T]
        return self.pool(x)


class Bank(nn.Module):
    def __init__(self, in_ch: int, bank_ch: int = 12, kernel_set=(3, 5, 7), front_pool_k=3):
        super().__init__()
        if front_pool_k is not None:
            assert 1 <= front_pool_k <= 16, f"front pool k={front_pool_k} must be in [1..16]"
            self.front_pool = FrontDownsample1D(front_pool_k)
        else:
            self.front_pool = nn.Identity()

        self.stem = nn.Conv1d(in_ch, in_ch, kernel_size=1, bias=False)

        branches = []
        for k in kernel_set:
            if k == 3:
                if in_ch > 64:
                    raise AssertionError("Bank: padding>0 with in_ch>64 is not allowed on ai8x")
                branches.append(nn.Conv1d(in_ch, bank_ch, kernel_size=3, padding=1, bias=False))
            elif k == 5:
                if in_ch > 64:
                    raise AssertionError("Bank: padding>0 with in_ch>64 is not allowed on ai8x")
                branches.append(nn.Conv1d(in_ch, bank_ch, kernel_size=5, padding=2, bias=False))
            elif k == 7:
                branches.append(Conv1dRF7(in_ch, bank_ch))
            else:
                raise ValueError("kernel_set must be subset of {3,5,7}")

        self.branches = nn.ModuleList(branches)
        self.act = nn.ReLU(inplace=True)
        self.F = bank_ch * len(self.branches)
        assert self.F % 4 == 0, "Concatenated bank channels (F) must be a multiple of 4"

    def forward(self, x):  # [B, C, T]
        x = self.front_pool(x)
        x = self.stem(x)
        outs = [self.act(b(x)) for b in self.branches]
        return torch.cat(outs, dim=1)  # [B, F, T']


class PatchEmbed1D_FusedPool(nn.Module):
    def __init__(self, in_ch: int, embed_dim: int, patch_t: int, conv_k: int = 3):
        super().__init__()
        if not (1 <= patch_t <= 16):
            raise AssertionError(
                f"patch_t={patch_t} invalid for MAX78000; must be in [1..16]. Use an additional downsample stage to reach your target."
            )
        if conv_k not in (1, 3, 5, 7, 9):
            raise ValueError("conv_k must be in {1,3,5,7,9}")

        self.pool = nn.AvgPool1d(kernel_size=patch_t, stride=patch_t)
        if conv_k == 7:
            self.proj = Conv1dRF7(in_ch, embed_dim)
        elif conv_k == 9:
            self.proj = nn.Sequential(
                nn.Conv1d(in_ch, embed_dim, kernel_size=5, padding=2, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2, bias=False),
            )
        else:
            padding = 0 if conv_k == 1 else (1 if conv_k == 3 else 2)
            if in_ch > 64 and padding > 0:
                raise AssertionError(
                    f"PatchEmbed1D: in_ch={in_ch}>64 with padding={padding} is not supported on ai8x"
                )
            self.proj = nn.Conv1d(in_ch, embed_dim, kernel_size=conv_k, padding=padding, bias=False)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):  # [B, F, T]
        x = self.pool(x)
        x = self.act(self.proj(x))
        return x  # [B, D, T']


class BottleneckReduce1D(nn.Module):
    def __init__(self, dim: int, reduce: int = 4, rescon=True):
        super().__init__()
        mid = max(8, dim // reduce)
        self.conv1 = nn.Conv1d(dim, mid, kernel_size=1, bias=False)
        self.act1 = nn.ReLU(inplace=True)
        if mid > 64:
            raise AssertionError(
                f"BottleneckReduce1D: mid={mid} > 64 would violate ai8x padding rule with 3x1 conv"
            )
        self.conv2 = nn.Conv1d(mid, mid, kernel_size=3, padding=1, bias=False)
        self.act2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv1d(mid, dim, kernel_size=1, bias=False)
        self.rescon = rescon

    def forward(self, x):
        if self.rescon:
            idn = x
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.conv3(x)
        if self.rescon:
            x = torch.add(x, idn)
        return x


class PoolThenPointwise1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int):
        super().__init__()
        assert 1 <= k <= 16, f"pool kernel k={k} must be in [1..16] on MAX78000"
        self.pool = nn.AvgPool1d(kernel_size=k, stride=k)
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=(1, 1), bias=False)

    def forward(self, x):  # x: [B, C, T]
        x = self.pool(x)
        x = x.unsqueeze(2)
        x = self.proj(x)
        return x.squeeze(2)


class Barrier2D(nn.Module):
    def __init__(self, channels: int, use_relu: bool = True):
        super().__init__()
        self.conv2d = nn.Conv2d(channels, channels, kernel_size=(1, 1), bias=False)
        self.act = nn.ReLU(inplace=True) if use_relu else nn.Identity()

    def forward(self, x):  # x: [B, C, T]
        x = x.unsqueeze(2)
        return self.act(self.conv2d(x))


class PoolThenPointwise2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int):
        super().__init__()
        assert 1 <= k <= 16, f"pool kernel k={k} must be in [1..16] on MAX78000"
        self.pool = nn.AvgPool2d(kernel_size=(1, k), stride=(1, k))
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=(1, 1), bias=False)

    def forward(self, x):  # x: [B, C, 1, T]
        x = self.pool(x)
        return self.proj(x)


class ConvHead2D(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, t_final: int):
        super().__init__()
        assert 1 <= t_final <= 16, "t_final must be in [1..16] on MAX78000"
        self.gap = nn.AvgPool2d(kernel_size=(1, t_final), stride=(1, 1))
        self.classifier = nn.Conv2d(in_ch, num_classes, kernel_size=(1, 1), bias=False)

    def forward(self, x):  # x: [B, C, 1, T]
        x = self.gap(x)
        x = self.classifier(x)
        return x.squeeze(-1).squeeze(-1)


class PhysioLite(nn.Module):
    """
    Input: [B, C_raw + 2*pos_freqs, T] (PosEnc concatenated outside model).

    Path:
      bank(1D) -> patch(1D) -> depth*(1x1->3x1->1x1+add)
      -> barrier(2D) -> optional post-patch pool(2D)
      -> GAP head(2D)
    """

    def __init__(
        self,
        input_length: int = 600,
        in_channels: int = 24,
        num_classes: int = 6,
        bank_ch: int = 12,
        kernel_set=(3, 5, 7),
        patch_t: int = 40,
        embed_dim: int = 224,
        depth: int = 3,
        reduce: int = 4,
        conv1d_k: int = 3,
        head_residual: bool = False,
        front_pool_k: int = 4,
        post_patch_pool_t=None,
        legacy_bank_front_pool_bug: bool = False,
    ):
        super().__init__()
        del head_residual  # retained for API compatibility

        self.input_length = int(input_length)
        patch_t = int(patch_t)
        front_pool_k = int(front_pool_k)

        assert self.input_length > 0, "input_length must be positive"
        assert self.input_length % front_pool_k == 0, (
            f"input_length={self.input_length} must be divisible by front_pool_k={front_pool_k}"
        )
        L1 = self.input_length // front_pool_k
        assert L1 % patch_t == 0, f"L1={L1} must be divisible by patch_t={patch_t}"

        if patch_t > 16 and post_patch_pool_t is None:
            raise AssertionError(
                "patch_t > 16 is not supported. Choose patch_t in [1..16] and, if needed, set post_patch_pool_t to further downsample."
            )
        
        if legacy_bank_front_pool_bug:
            self.bank = Bank(
                in_channels,
                bank_ch=bank_ch,
                kernel_set=kernel_set,
            )
        else:
            self.bank = Bank(
                in_channels,
                bank_ch=bank_ch,
                kernel_set=kernel_set,
                front_pool_k=front_pool_k,
            )
        self.patch = PatchEmbed1D_FusedPool(
            in_ch=self.bank.F,
            embed_dim=embed_dim,
            patch_t=patch_t,
            conv_k=conv1d_k,
        )
        self.blocks = nn.Sequential(*[BottleneckReduce1D(embed_dim, reduce=reduce) for _ in range(depth)])
        self.fusion_barrier = Barrier2D(embed_dim, use_relu=True)

        t1 = L1 // patch_t
        t2 = t1
        if post_patch_pool_t is not None:
            assert t1 % post_patch_pool_t == 0
            t2 = t1 // post_patch_pool_t
            self.head_proj = PoolThenPointwise2D(embed_dim, embed_dim, k=post_patch_pool_t)
        else:
            self.head_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=(1, 1), bias=False)

        assert 1 <= t2 <= 16, "Final temporal length must be in [1..16]"
        self.head = ConvHead2D(embed_dim, num_classes, t_final=t2)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):  # x: [B, C_total, T]
        assert x.size(-1) == self.input_length, f"Expected T={self.input_length}"
        x = self.bank(x)
        x = self.patch(x)
        x = self.blocks(x)
        x = self.fusion_barrier(x)
        x = self.head_proj(x)
        return self.head(x)


# -------- factory --------
def ai85_physiolite(**kw):
    c_raw = kw.get("in_channels", 8)
    pos_freqs = kw.get("pos_freqs", 8)
    return PhysioLite(
        input_length=kw.get("input_length", 600),
        in_channels=c_raw + 2 * pos_freqs,
        num_classes=kw.get("num_classes", 6),
        bank_ch=kw.get("bank_ch", 12),
        kernel_set=kw.get("kernel_set", (3, 5, 7)),
        patch_t=kw.get("patch_t", 4),
        embed_dim=kw.get("embed_dim", 256),
        depth=kw.get("depth", 3),
        reduce=kw.get("reduce", 4),
        conv1d_k=kw.get("conv1d_k", 3),
        head_residual=kw.get("head_residual", False),
        front_pool_k=kw.get("front_pool_k", 4),
        post_patch_pool_t=kw.get("post_patch_pool_t", 5),
        legacy_bank_front_pool_bug=kw.get("legacy_bank_front_pool_bug", False),
    )


models = [{"name": "ai85_physiolite", "min_input": 1, "dim": 1}]
