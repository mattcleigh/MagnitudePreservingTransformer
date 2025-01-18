import torch as T
from torch import nn
from torch.nn import functional as F

from src.layers.normalisation import get_norm
from src.torch_utils import apply_rope


class SwiGLUNet(nn.Module):
    """Simple gated bilinear feedfoward network with the Swish activation."""

    def __init__(self, dim: int, mult: int = 2, drop: float = 0.0) -> None:
        super().__init__()
        self.lin1 = nn.Linear(dim, 2 * mult * dim)
        self.lin2 = nn.Linear(mult * dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: T.Tensor) -> T.Tensor:
        x1, x2 = self.lin1(x).chunk(2, dim=-1)
        return self.lin2(self.drop(F.silu(x1) * x2))


class SelfAttention(nn.Module):
    """Multi-head self-attention layer."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        drop: float = 0,
        qk_norm: str = "none",
        out_norm: str = "none",
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.attn_dim = dim // num_heads
        self.drop = drop

        self.in_proj = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(drop)

        self.qk_norm = get_norm(qk_norm, self.attn_dim)
        self.out_norm = get_norm(out_norm, self.dim)

    def forward(self, x: T.Tensor, rp_freqs: T.Tensor | None = None) -> T.Tensor:
        B, S, D = x.shape
        shape = (B, S, 3, self.num_heads, self.attn_dim)
        q, k, v = self.in_proj(x).reshape(shape).permute(2, 0, 3, 1, 4).unbind(0)
        if rp_freqs is not None:
            q = apply_rope(q, rp_freqs)
            k = apply_rope(k, rp_freqs)
        q = self.qk_norm(q)
        k = self.qk_norm(k)
        a = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.drop * self.training,
            is_causal=True,
        )
        a = a.transpose(1, 2).contiguous().view(B, S, D)
        a = self.out_norm(a)
        a = self.out_proj(a)
        return self.out_drop(a)


class EncoderBlock(nn.Module):
    """Basic pre-norm transformer encoder block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ff_mult: int = 2,
        drop: float = 0.0,
        pre_norm: str = "layer",
        qk_norm: str = "none",
        out_norm: str = "none",
    ) -> None:
        super().__init__()
        self.attn = SelfAttention(dim, num_heads, drop, qk_norm, out_norm)
        self.ff = SwiGLUNet(dim, ff_mult, drop)
        self.norm1 = get_norm(pre_norm, dim)
        self.norm2 = get_norm(pre_norm, dim)
        self.ls_1 = nn.Parameter(T.ones(dim) / 10)
        self.ls_2 = nn.Parameter(T.ones(dim) / 10)

    def forward(self, x: T.Tensor, rp: T.Tensor | None = None) -> T.Tensor:
        x = x + self.ls_1 * self.attn(self.norm1(x), rp)
        return x + self.ls_2 * self.ff(self.norm2(x))
