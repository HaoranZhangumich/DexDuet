"""Reproducible initialization shared by the v2 ablation controls."""

from __future__ import annotations

import hashlib
import random

import numpy as np
import torch
from torch import nn


def seed_all(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def named_seed(seed, name):
    payload = f"coordinate_ablation_v2:{int(seed)}:{name}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def initialize_named_layers(network, seed, initializer, mu_initializer, sigma_initializer):
    """Give common layers equal weights despite different value-head sizes.

    Called on the CPU model before it moves to the training device. Each layer
    has a separate initialization stream, and each value output its own stream.
    This preserves the configured initializer and zero-bias convention.
    """
    if any(parameter.device.type != "cpu" for parameter in network.parameters()):
        raise ValueError("Named initialization must run before moving the model to CUDA")
    with torch.random.fork_rng(devices=[]):
        for name, module in network.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if name == "value":
                for row in range(module.out_features):
                    torch.random.default_generator.manual_seed(named_seed(seed, f"value.row.{row}"))
                    temporary = nn.Linear(module.in_features, 1)
                    initializer(temporary.weight)
                    with torch.no_grad():
                        module.weight[row].copy_(temporary.weight[0])
                        if module.bias is not None:
                            module.bias[row].zero_()
                continue
            torch.random.default_generator.manual_seed(named_seed(seed, name))
            module.reset_parameters()
            initializer(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            if name == "action_head":
                mu_initializer(module.weight)
        torch.random.default_generator.manual_seed(named_seed(seed, "sigma"))
        sigma_initializer(network.sigma)


def parameter_hash(network, component="actor"):
    digest = hashlib.sha256()
    for name, tensor in sorted(network.state_dict().items()):
        is_critic = name.startswith("critic.")
        is_value = name.startswith("value.")
        include = (
            component == "all"
            or (component == "actor" and not is_critic and not is_value)
            or (component == "critic_trunk" and is_critic)
            or (component == "value" and is_value)
        )
        if include:
            array = tensor.detach().cpu().contiguous().numpy()
            digest.update(name.encode())
            digest.update(str(array.shape).encode())
            digest.update(str(array.dtype).encode())
            digest.update(array.tobytes())
    return digest.hexdigest()
