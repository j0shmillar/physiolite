import torch
import torch.nn as nn
import torch.nn.functional as F


def distillation_kl_loss(student_logits, teacher_logits, T=1.0):
    s = F.log_softmax(student_logits / T, dim=1)
    t = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(s, t, reduction='batchmean') * (T * T)


def distillation_bce_loss(student_logits, teacher_logits, T=1.0):
    if T <= 0:
        T = 1.0
    with torch.no_grad():
        soft_targets = torch.sigmoid(teacher_logits / T)
    return F.binary_cross_entropy_with_logits(student_logits / T, soft_targets, reduction='mean') * (T * T)


class CrossEntropyWithSoftMacroF1Loss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        lam: float = 0.25,
        weight: torch.Tensor | None = None,
        ignore_index: int = -100,
        eps: float = 1e-8,
        ce_reduction: str = "mean",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lam = lam
        self.ignore_index = ignore_index
        self.eps = eps
        self.ce = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_index,
            reduction=ce_reduction,
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, target)

        if self.ignore_index is not None and self.ignore_index >= 0:
            valid = target != self.ignore_index
            logits = logits[valid]
            target = target[valid]
            if target.numel() == 0:
                return ce_loss

        probs = F.softmax(logits, dim=1)
        y = F.one_hot(target, num_classes=self.num_classes).float()

        tp = (probs * y).sum(dim=0)
        fp = (probs * (1.0 - y)).sum(dim=0)
        fn = ((1.0 - probs) * y).sum(dim=0)

        f1_per_class = (2.0 * tp) / (2.0 * tp + fp + fn + self.eps)
        soft_f1_macro = f1_per_class.mean()
        return ce_loss + (self.lam * (1.0 - soft_f1_macro))
