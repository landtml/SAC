"""Plasticity tools for SimbaNet: shrink-and-perturb and dormant-unit ReDo resets.

``shrink_perturb`` pulls every parameter toward a freshly initialized copy (Ash and Adams, 2020).
``dormant_fraction`` measures the share of expansion units (``fc1`` after ReLU) that are dormant on
a probe batch, over all SimBa blocks and ensemble members (Sokar et al., 2023). ``redo_reset``
re-initializes the dormant units: their incoming weights and bias are redrawn, and their outgoing
rows in ``fc2`` are zeroed. The output on the probe batch is unchanged exactly when every dormant
unit emits zero there. A dormant unit with a small nonzero activation shifts the output by its
outgoing contribution.
"""

from __future__ import annotations

import torch
from torch import nn

from softedge.agent.networks import SimbaNet


@torch.no_grad()
def shrink_perturb(model: nn.Module, fresh: nn.Module, alpha: float = 0.8) -> None:
    """Set each parameter of ``model`` to ``alpha * p + (1 - alpha) * p_fresh``, in place.

    ``fresh`` must have the same parameter names and shapes. Buffers are not touched. All checks
    run before the first write, so a rejected call leaves ``model`` unchanged.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must lie in [0, 1], got {alpha}")
    own = dict(model.named_parameters())
    other = dict(fresh.named_parameters())
    if own.keys() != other.keys():
        raise ValueError("fresh module has different parameter names")
    for name, p in own.items():
        if p.shape != other[name].shape:
            raise ValueError(
                f"shape mismatch for {name}: {tuple(p.shape)} vs {tuple(other[name].shape)}")
    for name, p in own.items():
        p.mul_(alpha).add_(other[name], alpha=1.0 - alpha)


def _record_post_relu(store: list[torch.Tensor]):
    def hook(_module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
        # SimbaBlock applies ReLU outside fc1, so the hook applies it to read the activation.
        store.append(torch.relu(output))

    return hook


@torch.no_grad()
def _dormant_masks(model: SimbaNet, x: torch.Tensor, tau: float) -> list[torch.Tensor]:
    """Per block, a bool mask of shape (members, units) that is True for dormant units."""
    acts: list[torch.Tensor] = []
    handles = [blk.fc1.register_forward_hook(_record_post_relu(acts)) for blk in model.blocks]
    try:
        model(x)
    finally:
        for h in handles:
            h.remove()
    masks = []
    for a in acts:
        # Post-ReLU values are non-negative, so the batch mean is also the mean absolute value.
        score = a.mean(dim=1)  # (members, units)
        layer_mean = score.mean(dim=1, keepdim=True)  # (members, 1)
        # Multiplying instead of dividing makes an all-zero layer count as dormant, not 0/0.
        masks.append(score <= tau * layer_mean)
    return masks


@torch.no_grad()
def dormant_fraction(model: SimbaNet, x: torch.Tensor, tau: float = 0.025) -> float:
    """Share of dormant fc1 units over all blocks and members, measured on ``x``.

    ``x`` has shape (B, d_in) or (E, B, d_in). Gradients are not built and the train/eval mode
    is not changed.
    """
    masks = _dormant_masks(model, x, tau)
    total = sum(m.numel() for m in masks)
    if total == 0:
        return 0.0
    return sum(int(m.sum()) for m in masks) / total


@torch.no_grad()
def redo_reset(model: SimbaNet, x: torch.Tensor, tau: float = 0.025,
               generator: torch.Generator | None = None) -> int:
    """Re-initialize the dormant fc1 units of every block and member. Return how many were reset.

    Incoming weights get a random direction scaled to the layer's mean column norm. The bias
    becomes 0 and the outgoing fc2 rows become 0. ``generator`` must live on the model's device.
    """
    masks = _dormant_masks(model, x, tau)
    count = 0
    for blk, mask in zip(model.blocks, masks, strict=True):
        count += int(mask.sum())
        if not mask.any():
            continue
        fc1, fc2 = blk.fc1, blk.fc2
        # Taken before any write, so the scale describes the layer the units were in.
        mean_norm = fc1.weight.norm(dim=1).mean(dim=1)  # (members,)
        direction = torch.randn(fc1.weight.shape, generator=generator,
                                device=fc1.weight.device, dtype=fc1.weight.dtype)
        direction = direction / direction.norm(dim=1, keepdim=True) * mean_norm[:, None, None]
        col = mask[:, None, :]  # (members, 1, units)
        fc1.weight.copy_(torch.where(col, direction, fc1.weight))
        fc1.bias.masked_fill_(col, 0.0)
        # Row j of fc2 reads unit j, so zeroing it removes the unit's contribution.
        fc2.weight.masked_fill_(mask[:, :, None], 0.0)
    return count
