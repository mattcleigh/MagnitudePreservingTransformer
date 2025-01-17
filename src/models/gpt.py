from functools import partial

import torch as T
from lightning import LightningModule
from torch import nn
from torch.nn import functional as F

from src.layers.normalisation import get_norm
from src.layers.transformer import (
    EncoderBlock,
    SelfAttention,
    SwiGLUNet,
    calc_rope_freqs,
)
from src.torch_utils import get_activations, remove_hooks


class GPT(LightningModule):
    """Generative Pre-trained Transformer model."""

    def __init__(
        self,
        *,
        vocab_size: int,
        optimizer: partial,
        scheduler: partial,
        dim: int = 128,
        num_layers: int = 6,
        final_norm: str = "layer",
        layer_config: dict | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        layer_config = layer_config or {}
        self.dim = dim
        self.num_layers = num_layers
        self.num_heads = layer_config.get("num_heads", 4)

        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            EncoderBlock(dim, **layer_config) for _ in range(num_layers)
        ])
        self.final_norm = get_norm(final_norm, dim)
        self.out_layer = nn.Linear(dim, vocab_size)

    def forward(self, x: T.LongTensor, y: T.LongTensor | None = None) -> T.Tensor:
        x = self.embed(x)
        rp = calc_rope_freqs(x, self.num_heads)
        for layer in self.layers:
            x = layer(x, rp)
        if y is not None:
            output = self.out_layer(self.final_norm(x))
            loss = F.cross_entropy(output.view(-1, output.size(-1)), y.long().view(-1))
        else:
            output = self.out_layer(self.final_norm(x[:, [-1]]))
            loss = None
        return output, loss

    def training_step(self, data: dict, batch_idx: int) -> T.Tensor:
        if batch_idx % 100 == 0:
            act_dict = {}
            hooks = get_activations(
                self, act_dict, types=[SelfAttention, SwiGLUNet, EncoderBlock]
            )
            param_dict = {}
            for n, param in self.named_parameters():
                if "ls_" in n:
                    param_dict[n] = param.detach().abs().mean()

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

    def configure_optimizers(self) -> dict:
        opt = self.hparams.optimizer(self.parameters())
        scheduler = {
            "scheduler": self.hparams.scheduler(optimizer=opt),
            "interval": "step",
        }
        return [opt], [scheduler]
