"""Per-hand Gaussian likelihood, entropy and KL: each hand's action block is scored separately, as the per-hand PPO
learner in ablation_ppo.py needs.
"""

import math


NUM_HANDS = 2


def per_hand_neglogp(actions, mu, sigma, num_hands=NUM_HANDS):
    """Gaussian negative log likelihood summed only within each hand."""
    if actions.shape != mu.shape or sigma.shape != mu.shape:
        raise ValueError("Actions, means and standard deviations must match")
    if actions.ndim != 2 or actions.shape[-1] % num_hands:
        raise ValueError("Action dimensions must partition into equal hand blocks")
    per_dimension = (
        0.5 * ((actions - mu) / sigma).square()
        + sigma.log()
        + 0.5 * math.log(2.0 * math.pi)
    )
    return per_dimension.reshape(actions.shape[0], num_hands, -1).sum(-1)


def per_hand_entropy(sigma, num_hands=NUM_HANDS):
    return (sigma.log() + 0.5 * math.log(2.0 * math.pi * math.e)).reshape(
        sigma.shape[0], num_hands, -1
    ).sum(-1)


def per_hand_kl(old_mu, old_sigma, mu, sigma, num_hands=NUM_HANDS):
    """Exact KL(old || new), separately for each action distribution."""
    per_dimension = (
        (sigma / old_sigma).log()
        + (old_sigma.square() + (old_mu - mu).square()) / (2 * sigma.square())
        - 0.5
    )
    return per_dimension.reshape(mu.shape[0], num_hands, -1).sum(-1)
