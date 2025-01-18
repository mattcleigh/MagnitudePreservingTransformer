import math

import torch as T
from torch import nn
from torch.nn import functional as F

from src.layers.normalisation import rms_norm, unit_norm
from src.layers.transformer import apply_rope


def mp_sum(a: T.Tensor, b: T.Tensor, t: float = 0.5) -> T.Tensor:
    """Magnitude preserving weighted addition."""
    if isinstance(t, T.Tensor):
        denom = T.sqrt((1 - t.detach()) ** 2 + t.detach() ** 2)
        return (a + t * (b - a)) / denom
    return a.lerp(b, t) / math.sqrt((1 - t) ** 2 + t**2)


class MPModule(nn.Module):
    """Base class for magnitude preserving modules, mainly to allow searching."""

    @T.no_grad
    def force_norm(self) -> None:
        """Force normalisation of the weights."""
        self.weight.data.copy_(rms_norm(self.weight.data))


class MPParameter(MPModule):
    """Fully learnable parameter with consistant normalisation."""

    def __init__(self, weight: T.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(rms_norm(weight))

    def forward(self) -> T.Tensor:
        return rms_norm(self.weight)


class MPLinear(MPModule):
    """Magnitude Preserving Linear layer.

    Normalisation is done twice in the forward pass due to Adam optimiser.
    Forced weight normalisation uses rms_norm.
    Standard weight normalisation uses unit_norm.
    This is due to Adams' normalisation of the gradients.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(rms_norm(T.randn(out_features, in_features)))

    def forward(self, x: T.Tensor) -> T.Tensor:
        return F.linear(x, unit_norm(self.weight))


class MPEmbedding(MPModule):
    """Embedding layer to project the input to a sphere."""

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(rms_norm(T.randn(num_embeddings, embedding_dim)))

    def forward(self, x: T.Tensor) -> T.Tensor:
        return F.embedding(x, rms_norm(self.weight))


class MPSwiGLUNet(nn.Module):
    """Magnitude preserving Swish-Gated Linear Unit layer."""

    def __init__(self, dim: int, mult: int = 2) -> None:
        super().__init__()
        self.lin1 = MPLinear(dim, 2 * mult * dim)
        self.lin2 = MPLinear(mult * dim, dim)

    def forward(self, x: T.Tensor) -> T.Tensor:
        x1, x2 = self.lin1(x).chunk(2, dim=-1)
        return self.lin2(F.silu(x1) * x2 / 0.596)


class MPMLP(nn.Module):
    """Magnitude preserving MLP layer."""

    def __init__(self, dim: int, mult: int = 2) -> None:
        super().__init__()
        self.lin1 = MPLinear(dim, mult * dim)
        self.lin2 = MPLinear(mult * dim, dim)

    def forward(self, x: T.Tensor) -> T.Tensor:
        return self.lin2(F.silu(self.lin1(x)) / 0.596)


class MPSelfAttention(nn.Module):
    """Magnitude preserving self-attention layer."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.attn_dim = dim // num_heads
        self.in_proj = MPLinear(dim, dim * 3)
        self.out_proj = MPLinear(dim, dim)

    def forward(self, x: T.Tensor, rp_freqs: T.Tensor | None = None) -> T.Tensor:
        B, S, D = x.shape
        shape = (B, S, 3, self.num_heads, self.attn_dim)
        q, k, v = self.in_proj(x).reshape(shape).permute(2, 0, 3, 1, 4).unbind(0)
        if rp_freqs is not None:
            q = apply_rope(q, rp_freqs)
            k = apply_rope(k, rp_freqs)
        q = rms_norm(q)
        k = rms_norm(k)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).contiguous().view(B, S, D)
        a = rms_norm(a)
        return self.out_proj(a)


class MPEncoderBlock(nn.Module):
    """Encoder block with Riemannian optimization updates on a sphere."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        ff_mult: int = 2,
    ) -> None:
        super().__init__()
        self.attn = MPSelfAttention(dim, num_heads)
        self.ff = MPSwiGLUNet(dim, ff_mult)
        self.ls_1 = nn.Parameter(T.ones(dim) / 10)
        self.ls_2 = nn.Parameter(T.ones(dim) / 10)

    def forward(self, x: T.Tensor, rp: T.Tensor | None = None) -> T.Tensor:
        x = mp_sum(x, self.attn(rms_norm(x), rp), self.ls_1.abs().clamp(0, 1))
        return mp_sum(x, self.ff(rms_norm(x)), self.ls_2.abs().clamp(0, 1))
