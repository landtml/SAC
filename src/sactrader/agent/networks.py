"""Network blocks: running observation normalization, SimBa residual MLPs, member-wise ensembles.

SimBa (Lee et al., 2025) adds input normalization, pre-LayerNorm residual feed-forward blocks and a
final LayerNorm to a plain MLP; that bias toward simple functions keeps larger networks stable in
off-policy RL. The critic ensemble runs all members in one batched matmul with shape (E, B, ...).
"""

from __future__ import annotations

import math

import torch
from torch import nn


class RunningNorm(nn.Module):
    """Per-feature running mean and variance (parallel Welford), applied with clipping."""

    def __init__(self, dim: int, clip: float = 10.0, eps: float = 1e-6):
        super().__init__()
        self.clip = clip
        self.eps = eps
        self.register_buffer("count", torch.zeros((), dtype=torch.float64))
        self.register_buffer("mean", torch.zeros(dim, dtype=torch.float64))
        self.register_buffer("m2", torch.zeros(dim, dtype=torch.float64))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1]).to(torch.float64)
        n = x.shape[0]
        if n == 0:
            return
        b_mean = x.mean(0)
        b_m2 = ((x - b_mean) ** 2).sum(0)
        tot = self.count + n
        delta = b_mean - self.mean
        self.mean += delta * n / tot
        self.m2 += b_m2 + delta**2 * self.count * n / tot
        self.count.copy_(tot)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.count < 2:
            return x
        var = (self.m2 / (self.count - 1)).clamp_min(self.eps)
        y = (x - self.mean.to(x.dtype)) / var.sqrt().to(x.dtype)
        return y.clamp(-self.clip, self.clip)


class EnsembleLinear(nn.Module):
    def __init__(self, members: int, d_in: int, d_out: int, gain: float = 1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(members, d_in, d_out))
        self.bias = nn.Parameter(torch.zeros(members, 1, d_out))
        for e in range(members):
            w = torch.empty(d_out, d_in)
            nn.init.orthogonal_(w, gain=gain)
            self.weight.data[e] = w.T

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (E, B, d_in) -> (E, B, d_out)
        return torch.baddbmm(self.bias, x, self.weight)


class EnsembleLayerNorm(nn.Module):
    def __init__(self, members: int, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(members, 1, dim))
        self.bias = nn.Parameter(torch.zeros(members, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.layer_norm(x, x.shape[-1:], eps=self.eps)
        return x * self.weight + self.bias


class SimbaBlock(nn.Module):
    def __init__(self, members: int, hidden: int, expansion: int = 4):
        super().__init__()
        self.norm = EnsembleLayerNorm(members, hidden)
        self.fc1 = EnsembleLinear(members, hidden, hidden * expansion, gain=math.sqrt(2))
        self.fc2 = EnsembleLinear(members, hidden * expansion, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(torch.relu(self.fc1(self.norm(x))))


class SimbaNet(nn.Module):
    """E independent SimBa MLPs evaluated together. Input (E, B, d_in) or (B, d_in)."""

    def __init__(self, members: int, d_in: int, d_out: int, hidden: int = 256, blocks: int = 2,
                 out_gain: float = 1e-2):
        super().__init__()
        self.members = members
        self.inp = EnsembleLinear(members, d_in, hidden)
        self.blocks = nn.ModuleList([SimbaBlock(members, hidden) for _ in range(blocks)])
        self.norm = EnsembleLayerNorm(members, hidden)
        self.out = EnsembleLinear(members, hidden, d_out, gain=out_gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0).expand(self.members, -1, -1)
        h = self.inp(x)
        for blk in self.blocks:
            h = blk(h)
        return self.out(self.norm(h))


class TanhGaussianActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256, blocks: int = 1,
                 log_std_min: float = -5.0, log_std_max: float = 2.0):
        super().__init__()
        self.net = SimbaNet(1, obs_dim, 2 * act_dim, hidden, blocks)
        self.act_dim = act_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, obs: torch.Tensor, deterministic: bool = False, with_logp: bool = True):
        out = self.net(obs)[0]
        mean, log_std = out.split(self.act_dim, dim=-1)
        # Soft clamp keeps gradients alive near the bounds.
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (torch.tanh(log_std) + 1)
        if deterministic:
            return torch.tanh(mean), None
        std = log_std.exp()
        z = mean + std * torch.randn_like(mean)
        action = torch.tanh(z)
        if not with_logp:
            return action, None
        # log N(z) - log|d tanh / dz|, using the stable identity log(1 - tanh^2 z) = 2 (log 2 - z - softplus(-2z)).
        logp = (-0.5 * ((z - mean) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
        logp = logp - (2 * (math.log(2) - z - nn.functional.softplus(-2 * z))).sum(-1)
        return action, logp


class QuantileCritic(nn.Module):
    """Ensemble of E critics, each predicting M quantiles of the soft return."""

    def __init__(self, obs_dim: int, act_dim: int, members: int = 2, quantiles: int = 25,
                 hidden: int = 256, blocks: int = 2):
        super().__init__()
        self.members = members
        self.quantiles = quantiles
        self.net = SimbaNet(members, obs_dim + act_dim, quantiles, hidden, blocks, out_gain=1.0)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1))  # (E, B, M)
