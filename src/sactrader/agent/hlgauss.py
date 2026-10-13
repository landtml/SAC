"""HL-Gauss categorical value head and loss, and an ensemble of categorical critics.

A scalar return is modelled as a distribution over n_bins equal-width bins on [v_min, v_max].
Each target is smeared over the bins with a Gaussian of width sigma_ratio * bin_width (HL-Gauss,
Imani and White 2018), and the critic minimizes cross-entropy against that histogram. The value
is the softmax-weighted mean of the bin centers (Farebrother et al. 2024, arXiv:2403.03950).
Targets outside [v_min, v_max] clip to the edge bins.
"""

from __future__ import annotations

import math

import torch

from sactrader.agent.networks import SimbaNet


class HLGauss:
    def __init__(self, n_bins: int = 101, v_min: float = -10.0, v_max: float = 10.0,
                 sigma_ratio: float = 0.75):
        if n_bins < 1 or v_max <= v_min:
            raise ValueError("need n_bins >= 1 and v_max > v_min")
        self.n_bins = n_bins
        self.v_min = v_min
        self.v_max = v_max
        self.edges = torch.linspace(v_min, v_max, n_bins + 1)
        self.width = (v_max - v_min) / n_bins
        self.sigma = sigma_ratio * self.width
        self.centers = (self.edges[:-1] + self.edges[1:]) / 2

    def project(self, targets: torch.Tensor) -> torch.Tensor:
        """Map scalar targets (...,) to bin probabilities (..., n_bins) with no gradient."""
        y = targets.detach().clamp(self.v_min, self.v_max).unsqueeze(-1)
        edges = self.edges.to(y)
        # Differences of erf at the edges are proportional to differences of the Gaussian CDF.
        # The common factor 1/2 cancels in the normalization below, so it is left out.
        cdf = torch.special.erf((edges - y) / (math.sqrt(2) * self.sigma))
        mass = cdf[..., 1:] - cdf[..., :-1]
        # Gaussian mass beyond the outer edges is dropped; renormalizing makes each row a
        # distribution again. This shifts projections only within a few sigma of an edge.
        return mass / mass.sum(-1, keepdim=True)

    def value(self, logits: torch.Tensor) -> torch.Tensor:
        """Expected value of the categorical distribution, (..., n_bins) -> (...,)."""
        return torch.softmax(logits, dim=-1) @ self.centers.to(logits)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Cross-entropy from logits (E, B, n_bins) to projected targets (B,), averaged over the
        batch and summed over the E members."""
        target_probs = self.project(targets)
        log_probs = torch.log_softmax(logits, dim=-1)
        ce = -(target_probs * log_probs).sum(-1)  # (E, B)
        return ce.mean(-1).sum()


class CategoricalCritic(torch.nn.Module):
    """Ensemble of E critics, each a categorical distribution over the same bins."""

    def __init__(self, obs_dim: int, act_dim: int, members: int = 2, n_bins: int = 101,
                 v_min: float = -10.0, v_max: float = 10.0, hidden: int = 256, blocks: int = 2):
        super().__init__()
        self.members = members
        self.hl = HLGauss(n_bins, v_min, v_max)
        # Output gain 1.0 keeps initial logits at unit scale; the default 1e-2 suits policy heads.
        self.net = SimbaNet(members, obs_dim + act_dim, n_bins, hidden, blocks, out_gain=1.0)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1))  # (E, B, n_bins)

    def q(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.hl.value(self.forward(obs, action))  # (E, B)
