"""Various custom normalisation layers."""

import torch as T
from torch import nn


@T.compile
def quick_norm(x, dim: int | tuple = -1, eps: float = 1e-8):
    norm = T.linalg.vector_norm(x.float(), dim=dim, keepdim=True) + eps
    return (x / norm).to(x.dtype)


class RMSNorm(nn.Module):
    """Root Mean Square Normalisation layer."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.const = dim**0.5

    def forward(self, x: T.Tensor) -> T.Tensor:
        return quick_norm(x, dim=-1) * self.const


class TokenNorm(nn.Module):
    """Subtract the mean of the tokens before normalising."""

    def __init__(self, dim: int, gamma: float = 0.999) -> None:
        super().__init__()
        self.dim = dim
        self.const = dim**0.5
        self.gamma = gamma
        self.register_buffer("mean", T.zeros((1, 1, self.dim), dtype=T.float32))

    def forward(self, x: T.Tensor) -> T.Tensor:
        if self.training:
            mean = x.float().mean(dim=(0, 1), keepdim=True)
            self.mean = self.mean * self.gamma + mean.detach() * (1 - self.gamma)
        else:
            mean = self.mean
        x = x - mean.to(x.dtype)
        return quick_norm(x, dim=-1) * self.const


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
