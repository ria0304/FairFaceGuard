"""Week 3 -- Baseline detector.

This is the "detector as currently built" -- a standard backbone + binary
real/fake head, with NO disentanglement machinery. It is trained once here
and then used two ways downstream:
    1. As the subject of the Week 3 subgroup accuracy/EER gap tables.
    2. As the FROZEN model probed and counterfactually tested in Week 4 --
       the causal claims in the paper are about *this* model, not a
       retrained adversarial variant.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_backbone(name: str = "efficientnet_b4", pretrained: bool = True) -> tuple[nn.Module, int]:
    """Returns (feature_extractor, embedding_dim). Swap architectures here only."""
    import torchvision.models as tvm

    if name == "efficientnet_b4":
        weights = tvm.EfficientNet_B4_Weights.DEFAULT if pretrained else None
        net = tvm.efficientnet_b4(weights=weights)
        embed_dim = net.classifier[1].in_features
        net.classifier = nn.Identity()
        return net, embed_dim

    if name == "xception":
        import timm  # pip install timm --break-system-packages

        net = timm.create_model("xception", pretrained=pretrained, num_classes=0)
        return net, net.num_features

    if name.startswith("vit") or name.startswith("clip"):
        import timm

        net = timm.create_model(name, pretrained=pretrained, num_classes=0)
        return net, net.num_features

    raise ValueError(f"Unknown backbone: {name}")


class BaselineDeepfakeDetector(nn.Module):
    """Standard detector: f(x) -> z -> g(z) -> real/fake logits.
    `encode()` is exposed separately so Week 4's probing/counterfactual code
    can reuse this exact frozen model."""

    def __init__(self, backbone_name: str = "efficientnet_b4", pretrained: bool = True, head_hidden: int = 256):
        super().__init__()
        self.backbone, self.embed_dim = build_backbone(backbone_name, pretrained)
        self.head = nn.Sequential(
            nn.Linear(self.embed_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(head_hidden, 2),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        if z.dim() > 2:
            z = F.adaptive_avg_pool2d(z, 1).flatten(1)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)[:, 1]  # P(fake)
