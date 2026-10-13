"""Hyperspherical ensemble MLP (SimbaV2-style) with the SimbaNet call signature.

Hidden states and hidden weight rows lie on the unit sphere, following SimbaV2 (arXiv 2502.15280).
Input (E, B, d_in) or (B, d_in) gives output (E, B, d_out). Members share no parameters and no
computation.

Embedding and block scales start at 1.0, so the first forward pass is a plain normalized MLP.
The block residual weight alpha starts at 1 / (blocks + 1). The output layer keeps its bias and
is not normalized. Call ``project_()`` after every optimizer step. The constructor calls it once
so the initial weights already satisfy the constraint.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from softedge.agent.networks import EnsembleLinear

EPS = 1e-8
EXPANSION = 4


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1, eps=EPS)


def _without_bias(layer: EnsembleLinear) -> EnsembleLinear:
    # A buffer keeps the zero bias out of the optimizer, so it is never trained.
    members, _, d_out = layer.bias.shape
    del layer.bias
    layer.register_buffer("bias", torch.zeros(members, 1, d_out))
    return layer


class HyperBlock(nn.Module):
    def __init__(self, members: int, hidden: int, blocks: int, dropout: float):
        super().__init__()
        self.fc1 = _without_bias(EnsembleLinear(members, hidden, hidden * EXPANSION))
        self.fc2 = _without_bias(EnsembleLinear(members, hidden * EXPANSION, hidden))
        self.scale = nn.Parameter(torch.ones(members, 1, hidden * EXPANSION))
        self.alpha = nn.Parameter(torch.full((members, 1, hidden), 1.0 / (blocks + 1)))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        u = _l2norm(self.fc2(self.drop(torch.relu(self.scale * self.fc1(h)))))
        # LERP residual: alpha sets the weight of the block output, then h returns to the sphere.
        return _l2norm(h + self.alpha * (u - h))


class HyperNet(nn.Module):
    """E independent hyperspherical MLPs evaluated together. Drop-in for SimbaNet."""

    def __init__(self, members: int, d_in: int, d_out: int, hidden: int = 256, blocks: int = 2,
                 out_gain: float = 1e-2, dropout: float = 0.0, shift: float = 3.0):
        super().__init__()
        self.members = members
        self.shift = shift
        self.inp = _without_bias(EnsembleLinear(members, d_in + 1, hidden))
        self.scale = nn.Parameter(torch.ones(members, 1, hidden))
        self.blocks = nn.ModuleList([HyperBlock(members, hidden, blocks, dropout)
                                     for _ in range(blocks)])
        self.out = EnsembleLinear(members, hidden, d_out, gain=out_gain)
        # Without this, the first forward pass would run on unnormalized weight rows.
        self.project_()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Last hidden state with shape (E, B, hidden); every row has unit norm."""
        if x.dim() == 2:
            x = x.unsqueeze(0).expand(self.members, -1, -1)
        # The constant channel lets the first layer express a bias after normalization.
        x = _l2norm(torch.cat([x, torch.full_like(x[..., :1], self.shift)], dim=-1))
        h = _l2norm(self.scale * self.inp(x))
        for blk in self.blocks:
            h = blk(h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.encode(x))

    @torch.no_grad()
    def project_(self) -> None:
        """Renormalize each output unit's weight vector over d_in, in the embedding and blocks."""
        hidden_layers = [self.inp]
        for blk in self.blocks:
            hidden_layers += [blk.fc1, blk.fc2]
        for layer in hidden_layers:
            layer.weight.copy_(F.normalize(layer.weight, dim=1, eps=EPS))
