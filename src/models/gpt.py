from functools import partial

import torch as T
from lightning import LightningModule
from torch import nn
from torch.nn import functional as F

from src.layers.normalisation import get_norm
from src.layers.transformer import EncoderBlock


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
        max_seq_len: int = 0,
        final_norm: str = "layer",
        layer_config: dict | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        layer_config = layer_config or {}
        self.dim = dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        self.embed = nn.Embedding(vocab_size, dim)
        self.abs_enc = nn.Parameter(T.randn(1, max_seq_len, dim) / 1000)
        self.layers = nn.ModuleList([
            EncoderBlock(dim, **layer_config) for _ in range(num_layers)
        ])
        self.final_norm = get_norm(final_norm, dim)
        self.out_layer = nn.Linear(dim, vocab_size)

    def forward(self, x: T.Tensor, y: T.Tensor | None = None) -> T.Tensor:
        x = self.embed(x.long()) + self.abs_enc[:, : x.size(1)]
        for layer in self.layers:
            x = layer(x)
        if y is not None:
            output = self.out_layer(self.final_norm(x))
            loss = F.cross_entropy(output.view(-1, output.size(-1)), y.long().view(-1))
        else:
            output = self.out_layer(self.final_norm(x[:, [-1]]))
            loss = None
        return output, loss

    def training_step(self, data: dict, batch_idx: int) -> T.Tensor:
        _, loss = self.forward(*data)
        self.log("train/total_loss", loss)
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
