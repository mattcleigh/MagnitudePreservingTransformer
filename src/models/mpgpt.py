import math
from functools import partial

import torch as T
from lightning import LightningModule
from torch import nn
from torch.nn import functional as F
import wandb.sync

from src.layers.normalisation import rms_norm, unit_norm
from src.torch_utils import get_activations, remove_hooks
import wandb


def mp_silu(x: T.Tensor) -> T.Tensor:
    """Magnitude preserving Silu activation."""
    return F.silu(x) / 0.596


def mp_add(a: T.Tensor, b: T.Tensor, t: float = 0.5) -> T.Tensor:
    """Magnitude preserving weighted addition."""
    return a.lerp(b, t) / math.sqrt((1 - t) ** 2 + t**2)


class MPParameter(nn.Module):
    """Fully learnable parameter with consistant normalisation."""

    def __init__(self, *shape: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(T.randn(*shape))

    def forward(self) -> T.Tensor:
        if self.training:
            with T.no_grad():
                self.weight.data.copy_(rms_norm(self.weight.data))
        return rms_norm(self.weight)


class MPLinear(nn.Module):
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
        self.weight = nn.Parameter(T.randn(out_features, in_features))

    def forward(self, x: T.Tensor) -> T.Tensor:
        if self.training:
            with T.no_grad():
                self.weight.data.copy_(rms_norm(self.weight.data))
        return F.linear(x, unit_norm(self.weight))


class MPEmbedding(nn.Module):
    """Embedding layer to project the input to a sphere."""

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(T.randn(num_embeddings, embedding_dim))

    def forward(self, x: T.Tensor) -> T.Tensor:
        if self.training:
            with T.no_grad():
                self.weight.data.copy_(rms_norm(self.weight.data))
        return F.embedding(x, rms_norm(self.weight))


class MPSwiGLUNet(nn.Module):
    """Magnitude preserving Swish-Gated Linear Unit layer."""

    def __init__(self, dim: int, mult: int = 2) -> None:
        super().__init__()
        self.lin1 = MPLinear(dim, 2 * mult * dim)
        self.lin2 = MPLinear(mult * dim, dim)

    def forward(self, x: T.Tensor) -> T.Tensor:
        x1, x2 = self.lin1(x).chunk(2, dim=-1)
        return self.lin2(mp_silu(x1) * x2)


class MPSelfAttention(nn.Module):
    """Magnitude preserving self-attention layer."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        causal: bool = True,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.attn_dim = dim // num_heads
        self.causal = causal
        self.scale = self.attn_dim**0.5
        self.in_proj = MPLinear(dim, dim * 3)
        self.out_proj = MPLinear(dim, dim)

    def forward(self, x: T.Tensor) -> T.Tensor:
        B, S, D = x.shape
        shape = (B, S, 3, self.num_heads, self.attn_dim)
        q, k, v = self.in_proj(x).reshape(shape).permute(2, 0, 3, 1, 4).unbind(0)
        q = rms_norm(q)
        k = rms_norm(k)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        a = rms_norm(a)
        a = a.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(a)


class MPEncoderBlock(nn.Module):
    """Encoder block with Riemannian optimization updates on a sphere."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        ff_mult: int = 4,
        causal: bool = True,
    ) -> None:
        super().__init__()
        self.attn = MPSelfAttention(dim, num_heads, causal)
        self.ff = MPSwiGLUNet(dim, ff_mult)

        self.alpha_v = 0.01
        self.alpha_scale = dim ** (-0.5)
        self.alpha_attn = nn.Parameter(self.alpha_scale * T.ones(dim))
        self.alpha_ff = nn.Parameter(self.alpha_scale * T.ones(dim))

    def forward(self, x: T.Tensor) -> T.Tensor:
        alpha = self.alpha_attn * self.alpha_v / self.alpha_scale
        x = x + alpha.abs() * (rms_norm(self.attn(x)) - x)
        x = rms_norm(x)

        alpha = self.alpha_ff * self.alpha_v / self.alpha_scale
        x = x + alpha.abs() * (rms_norm(self.ff(x)) - x)
        return rms_norm(x)


class MPGPT(LightningModule):
    """Magnitude Preserving GPT model.

    nGPT paper = https://arxiv.org/pdf/2410.01131v1.
    MP layers = https://arxiv.org/pdf/2312.02696.

    """

    def __init__(
        self,
        *,
        vocab_size: int,
        optimizer: partial,
        scheduler: partial,
        dim: int = 128,
        max_seq_len: int = 1024,
        num_layers: int = 6,
        layer_config: dict | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        layer_config = layer_config or {}
        self.dim = dim
        self.num_layers = num_layers

        self.embed = MPEmbedding(vocab_size, dim)
        self.abs_enc = MPParameter(1, max_seq_len, dim)
        self.layers = nn.ModuleList([
            MPEncoderBlock(dim, **layer_config) for _ in range(num_layers)
        ])
        self.out_layer = MPLinear(dim, vocab_size)

        self.alpha_scale = dim ** (-0.5)
        self.alpha_out = nn.Parameter(self.alpha_scale * T.ones(vocab_size))

    def forward(self, x: T.LongTensor, y: T.LongTensor | None = None) -> T.Tensor:
        _B, S = x.shape
        x = mp_add(self.embed(x), self.abs_enc()[:, :S], 0.1)
        for layer in self.layers:
            x = layer(x)
        alpha = self.alpha_out / self.alpha_scale
        if y is not None:
            output = self.out_layer(x) * alpha
            loss = F.cross_entropy(output.view(-1, output.size(-1)), y.view(-1))
        else:
            output = self.out_layer(x[:, [-1]]) * alpha
            loss = None
        return output, loss

    def training_step(self, data: dict, batch_idx: int) -> T.Tensor:
        if batch_idx % 100 == 0:
            act_dict = {}
            hooks = get_activations(
                self, act_dict, types=[MPSelfAttention, MPSwiGLUNet]
            )

        _, loss = self.forward(*data)
        self.log("train/total_loss", loss)

        if batch_idx % 100 == 0:
            for key, val in act_dict.items():
                self.log(f"act/{key}", val)
            remove_hooks(hooks)

        return loss

    def validation_step(self, data: dict, batch_idx: int) -> T.Tensor:
        _, loss = self.forward(*data)
        self.log("val/total_loss", loss)
        return loss

    def configure_optimizers(self) -> dict:
        opt = self.hparams.optimizer(self.parameters())
        scheduler = {
            "scheduler": self.hparams.scheduler(optimizer=opt),
            "interval": "step",
        }
        return [opt], [scheduler]
