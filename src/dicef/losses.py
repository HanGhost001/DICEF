"""Private/shared supervision with a fixed cosine-ramped teacher penalty."""
import math
import torch
import torch.nn.functional as F

MODALITIES = ("t1", "pet", "dti")


def ramp_weight(epoch, maximum=2.2, start=1, end=30):
    if maximum < 0 or not math.isfinite(maximum) or end <= start:
        raise ValueError("Invalid Ramp UMT parameters")
    if epoch <= start:
        return 0.0
    if epoch >= end:
        return float(maximum)
    progress = (epoch-start) / float(end-start)
    fraction = 0.5 * (1.0-math.cos(math.pi*progress))
    return maximum * fraction


def stage1_loss(outputs, teacher_features, labels, class_weights, umt_weight):
    private = torch.stack([F.cross_entropy(outputs[f"{m}_logits"],labels,weight=class_weights)
                           for m in MODALITIES]).mean()
    shared = torch.stack([F.cross_entropy(outputs[f"shared_{m}_logits"],labels,weight=class_weights)
                          for m in MODALITIES]).mean()
    umt = torch.stack([F.mse_loss(outputs[f"{m}_feature"].float(),teacher_features[m].detach().float())
                       for m in MODALITIES]).mean()
    total = private + shared + umt_weight * umt
    return total, {"private_ce":private.detach(), "shared_ce":shared.detach(), "umt_mse":umt.detach()}
