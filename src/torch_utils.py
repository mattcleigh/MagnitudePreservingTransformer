import re
from collections.abc import Iterable

import torch as T
from torch import nn


def get_activations(
    model: nn.Module,
    activation_dict: dict,
    regex: None | list = None,
    types: None | list = None,
) -> list:
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


def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()


class AdamWS(T.optim.AdamW):
    """AdamW optimizer where weight decay is only applied to matrices."""

    def __init__(self, params: Iterable | dict, weight_decay: float = 1e-2, **kwargs):
        params = list(params)
        if isinstance(params[0], tuple):
            params = [x for _, x in params]
        decay_params = [p for p in params if p.dim() >= 2]
        nodecay_params = [p for p in params if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        super().__init__(optim_groups, **kwargs)
