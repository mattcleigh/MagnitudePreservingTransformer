import math
from functools import partial

import torch as T
from lightning import LightningModule
from torch import nn
from torch.nn import functional as F

from src.layers.normalisation import unit_norm
from src.layers.transformer import apply_rope, calc_rope_freqs
from src.torch_utils import get_activations, remove_hooks


def copy_norm_weights(model: nn.Module, dim: int) -> None:
    model.weight.data.copy_(unit_norm(model.weight.data, dim))


class NSwiGLUNet(nn.Module):
    """Magnitude preserving Swish-Gated Linear Unit layer."""

    def __init__(self, dim: int, mult: int = 2) -> None:
        super().__init__()
        self.lin1 = nn.Linear(dim, 2 * mult * dim, bias=False)
        self.lin2 = nn.Linear(mult * dim, dim, bias=False)
        self.gain = nn.Parameter(T.ones(2 * mult * dim))
        self.scale = math.sqrt(dim)

    def forward(self, x: T.Tensor) -> T.Tensor:
        x = self.lin1(x) * self.gain * self.scale
        x1, x2 = x.chunk(2, dim=-1)
        return self.lin2(F.silu(x1) * x2)


class NSelfAttention(nn.Module):
    """Normalised Self-Attention layer."""

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
        self.in_proj = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.qk_scale = 1 / math.sqrt(dim)
        self.qk_gain = nn.Parameter(T.ones(dim) * self.qk_scale)
        self.scale = math.sqrt(self.attn_dim)

    def forward(self, x: T.Tensor, rp_freqs: T.Tensor | None = None) -> T.Tensor:
        B, S, D = x.shape
        shape = (B, S, 3, self.num_heads, self.attn_dim)
        q, k, v = self.in_proj(x).reshape(shape).permute(2, 0, 3, 1, 4).unbind(0)
        if rp_freqs is not None:
            q = apply_rope(q, rp_freqs)
            k = apply_rope(k, rp_freqs)
        qk_gain = self.qk_gain.view(1, self.num_heads, 1, -1) / self.qk_scale
        q = unit_norm(q) * qk_gain
        k = unit_norm(k) * qk_gain
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale)
        a = a.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(a)


class NEncoderBlock(nn.Module):
    """Normalised Encoder Block."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        ff_mult: int = 2,
    ) -> None:
        super().__init__()
        self.attn = NSelfAttention(dim, num_heads)
        self.ff = NSwiGLUNet(dim, ff_mult)

        self.ls_scale = 1 / math.sqrt(dim)
        self.ls_1 = nn.Parameter(T.ones(dim) * self.ls_scale)
        self.ls_2 = nn.Parameter(T.ones(dim) * self.ls_scale)

    def forward(self, x: T.Tensor, rp: T.Tensor | None = None) -> T.Tensor:
        ls_1 = self.ls_1.abs() * 0.1 / self.ls_scale
        x = unit_norm(x + ls_1 * (unit_norm(self.attn(x, rp) - x)))
        ls_2 = self.ls_2.abs() * 0.1 / self.ls_scale
        return unit_norm(x + ls_2 * (unit_norm(self.ff(x) - x)))


class NGPT(LightningModule):
    """Normalised GPT model.
    nGPT paper = https://arxiv.org/pdf/2410.01131v1.

    """

    def __init__(
        self,
        *,
        vocab_size: int,
        optimizer: partial,
        scheduler: partial,
        dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 4,
        ff_mult: int = 2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.dim = dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            NEncoderBlock(dim, num_heads, ff_mult) for _ in range(num_layers)
        ])
        self.out_layer = nn.Linear(dim, vocab_size, bias=False)

        self.out_scale = 1 / math.sqrt(dim)
        self.out_gain = nn.Parameter(T.ones(vocab_size) * self.out_scale)

    def forward(self, x: T.LongTensor, y: T.LongTensor | None = None) -> T.Tensor:
        x = self.embed(x)
        rp = calc_rope_freqs(x, self.num_heads)
        for layer in self.layers:
            x = layer(x, rp)
        if y is not None:
            output = self.out_layer(x) * self.out_gain / self.out_scale
            loss = F.cross_entropy(output.view(-1, output.size(-1)), y.view(-1))
        else:
            x = x[:, [-1]]
            output = self.out_layer(x) * self.out_gain / self.out_scale
            loss = None
        return output, loss

    def training_step(self, data: dict, batch_idx: int) -> T.Tensor:
        if batch_idx % 100 == 0:
            act_dict = {}
            hooks = get_activations(
                self, act_dict, types=[NSelfAttention, NSwiGLUNet, NEncoderBlock]
            )
            param_dict = {}
            for n, param in self.named_parameters():
                if "ls_" in n:
                    param_dict[n] = param.detach().abs().mean() * 0.1 / self.out_scale

        _, loss = self.forward(*data)
        self.log("train/total_loss", loss)

        if batch_idx % 100 == 0:
            for key, val in act_dict.items():
                self.log(f"act/{key}", val)
            for key, val in param_dict.items():
                self.log(f"param/{key}_mean", val)
            remove_hooks(hooks)

        return loss

    def validation_step(self, data: dict, batch_idx: int) -> T.Tensor:
        _, loss = self.forward(*data)
        self.log("val/total_loss", loss)
        return loss

    def normalise_weights(self) -> None:
        self.embed.weight.data.copy_(unit_norm(self.embed.weight.data, 1))
        self.out_layer.weight.data.copy_(unit_norm(self.out_layer.weight.data, 1))
        for layer in self.layers:
            copy_norm_weights(layer.attn.in_proj, 1)
            copy_norm_weights(layer.attn.out_proj, 0)
            copy_norm_weights(layer.ff.lin1, 1)
            copy_norm_weights(layer.ff.lin2, 0)

    def on_fit_start(self):
        self.normalise_weights()

    def optimizer_step(self, epoch, batch_idx, *args, **kwargs) -> None:
        """Ensures that all weights are properly normalised after one step."""
        super().optimizer_step(epoch, batch_idx, *args, **kwargs)
        self.normalise_weights()

    def configure_optimizers(self) -> dict:
        opt = self.hparams.optimizer(self.parameters())
        scheduler = {
            "scheduler": self.hparams.scheduler(optimizer=opt),
            "interval": "step",
        }
        return [opt], [scheduler]
