"""Various custom normalisation layers."""

import math

import torch as T
from torch import nn


def unit_norm(x, dim: int | tuple = -1, eps: float = 1e-4) -> T.Tensor:
    """Normalise the vector to a unit length."""
    n = T.linalg.vector_norm(x.float(), dim=dim, keepdim=True, dtype=T.float32)
    n = T.add(eps, n)
    return x / n.to(x.dtype)


def rms_norm(x, dim: int | tuple = -1, eps: float = 1e-4) -> T.Tensor:
    """Normalise the vector to have unit variance."""
    n = T.linalg.vector_norm(x.float(), dim=dim, keepdim=True, dtype=T.float32)
    n = T.add(eps, n, alpha=math.sqrt(n.numel() / x.numel()))
    return x / n.to(x.dtype)


class RMSNorm(nn.Module):
    """Root Mean Square Normalisation layer."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: T.Tensor) -> T.Tensor:
        return rms_norm(x, dim=-1)


class TokenNorm(nn.Module):
    """Subtract the mean of the tokens before normalising."""

    def __init__(self, dim: int, alpha: float = 1e-4) -> None:
        super().__init__()
        self.dim = dim
        self.alpha = alpha
        self.register_buffer("mean", T.zeros((1, 1, self.dim), dtype=T.float32))

    def forward(self, x: T.Tensor) -> T.Tensor:
        if self.training:
            mean = x.detach().float().mean(dim=(0, 1), keepdim=True)
            self.mean.lerp_(mean, self.alpha)
        x = x - self.mean.to(x.dtype)
        return rms_norm(x, dim=-1)


class Identity(nn.Module):
    """Identity layer to prevent if statements in the model."""

    def __init__(self, dim: int) -> None:
        super().__init__()

    def forward(self, x: T.Tensor, **kwargs) -> T.Tensor:
        return x

    def __repr__(self) -> str:
        return "Id()"


def get_norm(name: str, dim: int, **kwargs) -> nn.Module:
    if name == "none":
        return Identity(dim)
    if name == "rms":
        return RMSNorm(dim)
    if name == "token":
        return TokenNorm(dim, **kwargs)
    if name == "layer":
        return nn.LayerNorm(dim, elementwise_affine=False)
    raise ValueError(f"Unknown normalisation layer: {name}")
