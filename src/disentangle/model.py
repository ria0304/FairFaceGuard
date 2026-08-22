"""Week 4 -- Disentangled deepfake detector.

    shared backbone f(x) -> z
    real/fake head        g(z)             (the detector itself)
    skin-tone probe       h1(z), behind GRL(lambda_skin)
    illumination probe    h2(z), behind GRL(lambda_illum)

Train with lambda_skin, lambda_illum swept INDEPENDENTLY (including 0, the
undisentangled baseline) to trace two separate accuracy-vs-invariance
frontiers. If forcing invariance to skin tone costs real/fake accuracy while
invariance to illumination is nearly free, that's evidence the detector was
relying on skin-tone-correlated cues (H_skin); the reverse pattern supports
H_illum. See src/disentangle/train.py for the sweep driver.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.disentangle.grl import GradientReversalLayer


def build_backbone(name: str = "efficientnet_b4", pretrained: bool = True) -> tuple[nn.Module, int]:
    import torchvision.models as tvm

    if name == "efficientnet_b4":
        weights = tvm.EfficientNet_B4_Weights.DEFAULT if pretrained else None
        net = tvm.efficientnet_b4(weights=weights)
        embed_dim = net.classifier[1].in_features
        net.classifier = nn.Identity()
        return net, embed_dim

    if name == "xception":
        import timm

        net = timm.create_model("xception", pretrained=pretrained, num_classes=0)
        return net, net.num_features

    if name.startswith("vit") or name.startswith("clip"):
        import timm

        net = timm.create_model(name, pretrained=pretrained, num_classes=0)
        return net, net.num_features

    raise ValueError(f"Unknown backbone: {name}")


def _mlp_head(in_dim: int, hidden: int, out_dim: int, dropout: float = 0.2) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden, out_dim),
    )


class DisentangledDeepfakeDetector(nn.Module):
    def __init__(
        self,
        backbone_name: str = "efficientnet_b4",
        pretrained: bool = True,
        n_skin_bins: int = 6,     # Fitzpatrick I-VI
        n_illum_bins: int = 5,
        head_hidden: int = 256,
        lambda_skin: float = 1.0,
        lambda_illum: float = 1.0,
    ):
        super().__init__()
        self.backbone, self.embed_dim = build_backbone(backbone_name, pretrained)

        self.fake_head = _mlp_head(self.embed_dim, head_hidden, 2)

        self.grl_skin = GradientReversalLayer(lambda_skin)
        self.grl_illum = GradientReversalLayer(lambda_illum)
        self.skin_probe = _mlp_head(self.embed_dim, head_hidden, n_skin_bins)
        self.illum_probe = _mlp_head(self.embed_dim, head_hidden, n_illum_bins)

    def set_lambdas(self, lambda_skin: float, lambda_illum: float) -> None:
        self.grl_skin.set_lambda(lambda_skin)
        self.grl_illum.set_lambda(lambda_illum)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        if z.dim() > 2:
            z = F.adaptive_avg_pool2d(z, 1).flatten(1)
        return z

    def forward(self, x: torch.Tensor) -> dict:
        z = self.encode(x)
        return {
            "z": z,
            "fake_logits": self.fake_head(z),
            "skin_logits": self.skin_probe(self.grl_skin(z)),
            "illum_logits": self.illum_probe(self.grl_illum(z)),
        }

    @torch.no_grad()
    def load_backbone_from_baseline(self, baseline_state_dict: dict) -> None:
        """Warm-start from the Week 3 baseline's backbone weights so the
        disentanglement run starts from the same representation being studied."""
        own_state = self.backbone.state_dict()
        matched = {k: v for k, v in baseline_state_dict.items() if k in own_state and v.shape == own_state[k].shape}
        own_state.update(matched)
        self.backbone.load_state_dict(own_state)
