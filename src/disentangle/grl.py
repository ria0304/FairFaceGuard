"""Gradient Reversal Layer (Ganin & Lempitsky, 2015).

Identity on the forward pass; scales-and-flips the gradient on backward.
This is what lets a single backward pass simultaneously (a) train the
adversarial probe to predict skin-tone/illumination from z, and (b) push
the backbone to make z LESS predictive of them -- without a separate
min-max training loop.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class _GradReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambd: float = 1.0):
        super().__init__()
        self.lambd = lambd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradReverseFn.apply(x, self.lambd)

    def set_lambda(self, lambd: float) -> None:
        self.lambd = lambd


def dann_lambda_schedule(progress: float, gamma: float = 10.0) -> float:
    """Standard DANN warm-up: 0 -> 1 as training progresses (progress in [0,1])."""
    progress = min(max(progress, 0.0), 1.0)
    return 2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0
