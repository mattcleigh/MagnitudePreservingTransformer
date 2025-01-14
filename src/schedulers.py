"""Collection of learning rate schedulers."""

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR


def linear_warmup(
    optimizer: Optimizer,
    warmup_steps: int = 1000,
    init_factor: float = 1e-2,
) -> LambdaLR:
    """Return a scheduler with a linear warmup."""

    def fn(x: int) -> float:
        return min(1, init_factor + x * (1 - init_factor) / max(1, warmup_steps))

    return LambdaLR(optimizer, fn)


def linear_warmup_exp_decay(
    optimizer: Optimizer,
    warmup_steps: int = 1000,
    half_life: int = 1000,
    final_factor: float = 1e-3,
    init_factor: float = 1e-1,
) -> LambdaLR:
    """Return a scheduler with a linear warmup and a sqrt decay."""

    def fn(x: int) -> float:
        if x < warmup_steps:
            return init_factor + x * (1 - init_factor) / max(1, warmup_steps)
        decay = -math.log(2) / half_life
        return max(math.exp(decay * (x - warmup_steps)), final_factor)

    return LambdaLR(optimizer, fn)


def one_cycle(
    optimizer: Optimizer,
    total_steps: int = 1000,
    **kwargs,
) -> OneCycleLR:
    """Get the learning rate scheduler."""
    return OneCycleLR(
        optimizer,
        **kwargs,
        total_steps=total_steps,
        max_lr=optimizer.param_groups[0]["lr"],
    )
