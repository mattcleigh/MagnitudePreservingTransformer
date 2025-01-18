import logging
import math
import re
from collections.abc import Iterable

import torch as T
from torch import nn

log = logging.getLogger(__name__)


def mp_sum(a: T.Tensor, b: T.Tensor, t: float = 0.5) -> T.Tensor:
    """Magnitude preserving weighted addition."""
    if isinstance(t, T.Tensor):
        return (a + t * (b - a)) / T.sqrt((1 - t) ** 2 + t**2)
    return a.lerp(b, t) / math.sqrt((1 - t) ** 2 + t**2)


def calc_rope_freqs(x: T.Tensor, num_heads: int, theta: float = 10000.0):
    """Precompute the frequencies for the rotary positional encoding."""
    _B, S, D = x.shape
    HD = D // num_heads
    freqs = 1.0 / (theta ** (T.arange(0, HD, 2, device=x.device).float() / HD))
    t = T.arange(S, device=x.device, dtype=T.float32)
    freqs = T.outer(t, freqs)
    return T.polar(T.ones_like(freqs), freqs)


def apply_rope(x: T.Tensor, freqs_cis: T.Tensor) -> T.Tensor:
    """Rotate the input tensor using the precomputed frequencies."""
    B, NH, S, HD = x.shape
    out = T.view_as_complex(x.float().reshape(B, NH, S, HD // 2, 2))
    out = T.view_as_real(out * freqs_cis)
    return out.view_as(x).type_as(x)


def unit_norm(x, dim: int | tuple = -1, eps: float = 1e-4) -> T.Tensor:
    """Normalise the vector to a unit length."""
    n = T.linalg.vector_norm(x.float(), dim=dim, keepdim=True, dtype=T.float32)
    n = T.add(eps, n)
    return x / n.to(x.dtype)


def rms_norm(x, dim: int | tuple = -1, eps: float = 1e-4) -> T.Tensor:
    """Normalise the vector to have unit variance."""
    n = T.linalg.vector_norm(x.float(), dim=dim, keepdim=True, dtype=T.float32)
    n = T.add(eps, n, alpha=math.sqrt(n.numel() / x.numel()))
    return x / n.to(x.dtype)


def get_activations(
    model: nn.Module,
    activation_dict: dict,
    regex: list | None = None,
    types: list | None = None,
) -> list:
    """Create hooks for storing the output activations of layers in a model."""
    hooks = []

    def hook(name) -> callable:
        def forward_hook(_module: nn.Module, _input: T.Tensor, output: T.Tensor):
            activation_dict[name] = output.detach().std().cpu().item()

        return forward_hook

    for n, m in model.named_modules():
        passed = False
        if regex is not None and any(re.match(r, n) for r in regex):
            passed = True
        if types is not None and any(isinstance(m, t) for t in types):
            passed = True
        if passed:
            h = m.register_forward_hook(hook(n))
            hooks.append(h)

    return hooks


def remove_hooks(hooks: list) -> None:
    """Remove a list of hooks."""
    for hook in hooks:
        hook.remove()


@T.no_grad
def unit_norm_weights(model: nn.Module, dim: int) -> None:
    """Make the weights of a model unit norm."""
    model.weight.data.copy_(unit_norm(model.weight.data, dim))


class AdamWS(T.optim.AdamW):
    """AdamW optimizer where weight decay is only applied to matrices."""

    def __init__(self, params: Iterable | dict, weight_decay: float = 1e-2, **kwargs):
        params = list(params)
        if isinstance(params[0], tuple):
            params = [x for _, x in params]
        decay_params = [p for p in params if p.dim() >= 2]
        nodecay_params = [p for p in params if p.dim() < 2]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        log.info(
            f"AdamWS: Applying weight decay {weight_decay} to {num_decay_params} "
            f"parameters, and 0.0 to {num_nodecay_params} parameters."
        )
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        super().__init__(optim_groups, **kwargs)
