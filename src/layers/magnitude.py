import math

import torch as T
from torch import nn
from torch.nn import functional as F

from src.torch_utils import apply_rope, mp_sum, rms_norm, unit_norm


class SParameter(nn.Module):
    """Scaled parameter for equivalent learning rates."""

    def __init__(
        self,
        values: T.Tensor,
        init: float = 1.0,
        scale: float | None = None,
    ) -> None:
        super().__init__()
        if scale is None:
            scale = 1 / math.sqrt(values.shape[-1])
        self.value = nn.Parameter(values * scale)
        self.scale = scale
        self.init = init

    def forward(self) -> T.Tensor:
        return self.value * self.init / self.scale

    def eff_clamp_(self, min_: float, max_: float) -> None:
        """Clamp the effective output of the parameter."""
        min_ = min_ * self.scale / self.init
        max_ = max_ * self.scale / self.init
        self.value.data.clamp_(min_, max_)


class MPModule(nn.Module):
    """Base class for magnitude preserving modules, mainly to allow searching."""

    @T.no_grad
    def force_norm(self) -> None:
        """Force normalisation of the weights."""
        self.weight.data.copy_(unit_norm(self.weight.data))


class MPParameter(MPModule):
    """Fully learnable parameter with consistant normalisation."""

    def __init__(self, weight: T.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(rms_norm(weight))

    def force_norm(self) -> None:
        """Direct output so must be RMS normalised."""
        self.weight.data.copy_(rms_norm(self.weight.data))

    def forward(self) -> T.Tensor:
        return rms_norm(self.weight)


class MPLinear(MPModule):
    """Magnitude Preserving Linear layer.

    Normalisation is done twice in the forward pass to ensure gradients are tangental
    to the sphere.
    EDM2 paper suggests using rms_norm in the forced update but I have seen that make
    things much worse.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(unit_norm(T.randn(out_features, in_features)))

    def forward(self, x: T.Tensor) -> T.Tensor:
        return F.linear(x, unit_norm(self.weight))


class MPEmbedding(MPModule):
    """Embedding layer to project the input to a sphere."""

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(unit_norm(T.randn(num_embeddings, embedding_dim)))

    def force_norm(self) -> None:
        """Direct output so must be RMS normalised."""
        self.weight.data.copy_(unit_norm(self.weight.data))

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
    """Magnitude Preserving Self-Attention layer."""

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
        self.qk_gain = SParameter(T.ones(dim))

    def forward(self, x: T.Tensor, rp_freqs: T.Tensor | None = None) -> T.Tensor:
        B, S, D = x.shape
        shape = (B, S, 3, self.num_heads, self.attn_dim)
        q, k, v = self.in_proj(x).reshape(shape).permute(2, 0, 3, 1, 4).unbind(0)
        if rp_freqs is not None:
            q = apply_rope(q, rp_freqs)
            k = apply_rope(k, rp_freqs)
        qk_gain = self.qk_gain().view(1, self.num_heads, 1, -1)
        q = rms_norm(q) * qk_gain
        k = rms_norm(k) * qk_gain
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).contiguous().view(B, S, D)
        a = rms_norm(a)
        return self.out_proj(a)


class MPEncoderBlock(nn.Module):
    """Magnitude Preserving Encoder Block."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        ff_mult: int = 2,
        res_type: str = "ngpt",
    ) -> None:
        super().__init__()
        assert res_type in {"ngpt", "mp-pre", "mp-post"}
        self.attn = MPSelfAttention(dim, num_heads)
        self.ff = MPSwiGLUNet(dim, ff_mult)
        self.ls_1 = SParameter(T.ones(dim), 0.1)
        self.ls_2 = SParameter(T.ones(dim), 0.1)
        self.res_type = res_type

    def forward(self, x: T.Tensor, rp: T.Tensor | None = None) -> T.Tensor:
        if self.res_type == "ngpt":
            x = rms_norm(x + self.ls_1() * (rms_norm(self.attn(x, rp) - x)))
            x = rms_norm(x + self.ls_2() * (rms_norm(self.ff(x) - x)))
        elif self.res_type == "mp-pre":
            x = mp_sum(x, self.attn(rms_norm(x), rp), self.ls_1())
            x = mp_sum(x, self.ff(rms_norm(x)), self.ls_2())
        elif self.res_type == "mp-post":
            x = mp_sum(x, rms_norm(self.attn(x, rp)), self.ls_1())
            x = mp_sum(x, rms_norm(self.ff(x)), self.ls_2())
        return x
