import torch as T
from torch import nn
from torch.nn import functional as F

from src.layers.normalisation import get_norm


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
        num_heads: int = 1,
        drop: float = 0,
        qk_norm: str = "none",
        out_norm: str = "none",
        causal: bool = True,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.attn_dim = dim // num_heads
        self.drop = drop
        self.causal = causal

        self.in_proj = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(drop)

        self.qk_norm = get_norm(qk_norm, dim)
        self.out_norm = get_norm(out_norm, dim)

    def forward(self, x: T.Tensor) -> T.Tensor:
        B, S, D = x.shape
        shape = (B, S, 3, self.num_heads, D // self.num_heads)

        q, k, v = self.in_proj(x).reshape(shape).permute(2, 0, 3, 1, 4).unbind(0)
        q = self.qk_norm(q)
        k = self.qk_norm(k)

        attn_drop = self.drop * self.training
        a = F.scaled_dot_product_attention(q, k, v, attn_drop, is_causal=self.causal)
        a = a.transpose(1, 2).contiguous().view(B, S, D)

        a = self.out_norm(a)
        return self.out_proj(a)


class EncoderBlock(nn.Module):
    """Basic pre-norm transformer encoder block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ff_mult: int = 4,
        drop: float = 0.0,
        pre_norm: str = "layer",
        qk_norm: str = "none",
        out_norm: str = "none",
        causal: bool = True,
    ) -> None:
        super().__init__()
        self.attn = SelfAttention(dim, num_heads, drop, qk_norm, out_norm, causal)
        self.ff = SwiGLUNet(dim, ff_mult, drop)
        self.norm1 = get_norm(pre_norm, dim)
        self.norm2 = get_norm(pre_norm, dim)

    def forward(self, x: T.Tensor) -> T.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.ff(self.norm2(x))


class Transformer(nn.Module):
    """Transformer encoder stack with embedding layer."""

    def __init__(
        self,
        *,
        dim: int = 128,
        inpt_dim: int = 0,
        outp_dim: int = 0,
        num_layers: int = 6,
        max_seq_len: int = 0,
        do_pos_enc: bool = False,
        final_norm: str = "layer",
        layer_config: dict | None = None,
    ) -> None:
        super().__init__()
        assert not (
            do_pos_enc and not max_seq_len
        ), "Define max_seq_len for positional encoding"
        layer_config = layer_config or {}

        self.dim = dim
        self.num_layers = num_layers
        self.inpt_dim = inpt_dim or dim
        self.outp_dim = outp_dim or dim

        # Layer structure
        self.in_layer = nn.Linear(inpt_dim, dim)
        self.layers = nn.ModuleList([
            EncoderBlock(dim, **layer_config) for _ in range(num_layers)
        ])
        self.final_norm = get_norm(final_norm, dim)
        self.out_layer = nn.Linear(dim, outp_dim)

        # Optional positional encoding
        if do_pos_enc:
            self.abs_enc = nn.Parameter(T.randn(1, max_seq_len, dim) / 1000)

    def forward(self, x: T.Tensor) -> T.Tensor:
        x = self.in_layer(x)
        if hasattr(self, "abs_enc"):
            x = x + self.abs_enc
        for layer in self.layers:
            x = layer(x)
        return self.out_layer(self.final_norm(x))
