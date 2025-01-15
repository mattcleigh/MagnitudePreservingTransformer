from functools import partial

import torch as T
from lightning import LightningModule
from torch import nn
from torch.nn import functional as F

from src.layers.normalisation import quick_norm


def norm_weights(module, dim):
    """Normalise weights of a module."""
    module.weight.data.copy_(quick_norm(module.weight.data, dim))


class SphereSwiGLUNet(nn.Module):
    """Simple gated bilinear feedfoward network with the Swish activation."""

    def __init__(self, dim: int, mult: int = 2) -> None:
        super().__init__()
        self.lin1 = nn.Linear(dim, 2 * mult * dim, bias=False)
        self.lin2 = nn.Linear(mult * dim, dim, bias=False)
        self.suv = nn.Parameter(T.ones(2 * mult * dim))

    def forward(self, x: T.Tensor) -> T.Tensor:
        x1, x2 = (self.lin1(x) * self.suv).chunk(2, dim=-1)
        return self.lin2(F.silu(x1) * x2)


class SphereSelfAttention(nn.Module):
    """Multi-head self-attention layer."""

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

        self.wk = nn.Linear(dim, dim, bias=False)
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.sqk = nn.Parameter(T.ones(1, self.num_heads, 1, self.attn_dim))
        self.sqk_scale = dim**-0.5

    def forward(self, x: T.Tensor) -> T.Tensor:
        B, S, D = x.shape
        shape = (B, S, self.num_heads, self.attn_dim)
        q = self.wq(x).reshape(shape).permute(0, 2, 1, 3)
        k = self.wk(x).reshape(shape).permute(0, 2, 1, 3)
        v = self.wv(x).reshape(shape).permute(0, 2, 1, 3)

        q = quick_norm(q) * self.sqk / self.sqk_scale
        k = quick_norm(k) * self.sqk / self.sqk_scale

        a = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.drop * self.training,
            is_causal=self.causal,
            scale=self.attn_dim**0.5,
        )
        a = a.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(a)


class SphereEncoderBlock(nn.Module):
    """Basic pre-norm transformer encoder block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ff_mult: int = 4,
        causal: bool = True,
    ) -> None:
        super().__init__()
        self.attn = SphereSelfAttention(dim, num_heads, causal)
        self.ff = SphereSwiGLUNet(dim, ff_mult)
        self.aa = nn.Parameter(T.randn(1, 1, dim) / 100)
        self.am = nn.Parameter(T.randn(1, 1, dim) / 100)

        self.a_init = 0.1
        self.a_scale = dim ** (-0.5)
        self.aa = nn.Parameter(self.a_init * T.ones(dim))
        self.am = nn.Parameter(self.a_init * T.ones(dim))

    def forward(self, x: T.Tensor) -> T.Tensor:
        attn_out = quick_norm(self.attn(x))
        x = x + (attn_out - x) * (self.aa * self.a_init / self.a_scale).abs()
        x = quick_norm(x)

        ff_out = quick_norm(self.ff(x))
        x = x + (ff_out - x) * (self.am * self.a_init / self.a_scale).abs()

        return quick_norm(x)


class NGPT(LightningModule):
    """Normalised GPT model. https://arxiv.org/pdf/2410.01131v1."""

    def __init__(
        self,
        *,
        vocab_size: int,
        optimizer: partial,
        scheduler: partial,
        dim: int = 128,
        num_layers: int = 6,
        layer_config: dict | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        layer_config = layer_config or {}
        self.dim = dim
        self.num_layers = num_layers

        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            SphereEncoderBlock(dim, **layer_config) for _ in range(num_layers)
        ])
        self.out_layer = nn.Linear(dim, vocab_size)

        self.sz_scale = dim**-0.5
        self.sz = nn.Parameter(T.ones(vocab_size))

    def forward(self, x: T.Tensor, y: T.Tensor | None = None) -> T.Tensor:
        x = self.embed(x.long())
        for layer in self.layers:
            x = layer(x)
        if y is not None:
            output = self.out_layer(x)
            output = output * self.sz / self.sz_scale
            loss = F.cross_entropy(output.view(-1, output.size(-1)), y.long().view(-1))
        else:
            output = self.out_layer(x[:, [-1]])
            output = output * self.sz / self.sz_scale
            loss = None
        return output, loss

    def training_step(self, data: dict, batch_idx: int) -> T.Tensor:
        _, loss = self.forward(*data)
        self.log("train/total_loss", loss)
        return loss

    @T.no_grad()
    def on_train_batch_end(self, *args, **kwargs) -> None:
        """After each training step normalise all weight martices in the model."""
        norm_weights(self.embed, 1)  # V, n_embd
        for layer in self.layers:
            for block in layer.blocks:
                norm_weights(block.attn.wq, 1)
                norm_weights(block.attn.wk, 1)
                norm_weights(block.attn.wv, 1)
                norm_weights(block.attn.out_proj, 0)
                norm_weights(block.ff.lin1, 1)
                norm_weights(block.ff.lin2, 0)
        norm_weights(self.out_layer, 1)

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
