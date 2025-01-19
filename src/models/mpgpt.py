from functools import partial

import torch as T
from lightning import LightningModule
from torch import nn
from torch.nn import functional as F

from src.layers.magnitude import (
    MPEmbedding,
    MPEncoderBlock,
    MPLinear,
    MPModule,
    MPSelfAttention,
    MPSwiGLUNet,
    SParameter,
)
from src.torch_utils import calc_rope_freqs, get_activations, remove_hooks


class MPGPT(LightningModule):
    """Magnitude Preserving GPT model.
    MP layers = https://arxiv.org/pdf/2312.02696.

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
        res_type: str = "ngpt",
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.dim = dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.embed = MPEmbedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            MPEncoderBlock(dim, num_heads, ff_mult, res_type) for _ in range(num_layers)
        ])
        self.out_layer = MPLinear(dim, vocab_size)
        self.out_gain = SParameter(T.ones(vocab_size))

    def forward(self, x: T.LongTensor, y: T.LongTensor | None = None) -> T.Tensor:
        x = self.embed(x)
        rp = calc_rope_freqs(x, self.num_heads)
        for layer in self.layers:
            x = layer(x, rp)
        if y is not None:
            output = self.out_layer(x) * self.out_gain()
            loss = F.cross_entropy(output.view(-1, output.size(-1)), y.view(-1))
        else:
            x = x[:, [-1]]
            output = self.out_layer(x) * self.out_gain()
            loss = None
        return output, loss

    def training_step(self, data: dict, batch_idx: int) -> T.Tensor:
        if batch_idx % 100 == 0:
            act_dict = {}
            hooks = get_activations(
                self, act_dict, types=[MPSelfAttention, MPSwiGLUNet, MPEncoderBlock]
            )
            param_dict = {}
            for n, m in self.named_modules():  # LS is now a module
                if "ls_" in n:
                    param_dict[n] = m().detach().abs().mean()

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
        self.log("val/total_loss", loss, sync_dist=True)
        return loss

    def normalise_weights(self) -> None:
        for n, m in self.named_modules():
            if isinstance(m, MPModule):
                m.force_norm()
            if "ls_" in n:
                m.eff_clamp_(0.02, 0.98)

    def optimizer_step(self, *args, **kwargs) -> None:
        """Ensures that all weights are properly normalised after one step."""
        super().optimizer_step(*args, **kwargs)
        self.normalise_weights()

    def on_fit_start(self) -> None:
        self.normalise_weights()

    def configure_optimizers(self) -> dict:
        opt = self.hparams.optimizer(self.parameters())
        scheduler = {
            "scheduler": self.hparams.scheduler(optimizer=opt),
            "interval": "step",
        }
        return [opt], [scheduler]
