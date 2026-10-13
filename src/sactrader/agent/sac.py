"""Soft Actor-Critic with truncated quantile critics and a risk-sensitive actor.

Critic: an ensemble of E quantile networks (TQC, Kuznetsov et al., 2020). The target pools the
E * M target quantiles at (s', a' ~ pi), drops the ``drop`` largest per member to curb
overestimation, and subtracts the entropy term. Loss: quantile Huber.

Actor: maximizes ``rho(Z(s, a)) - alpha * log pi(a | s)`` where rho is the mean of the pooled
quantiles (risk neutral) or their lower-tail mean CVaR_eta (risk averse).

Temperature: alpha is tuned toward ``target_entropy = -act_dim * target_entropy_scale``.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from sactrader.agent.hlgauss import CategoricalCritic
from sactrader.agent.networks import QuantileCritic, RunningNorm, TanhGaussianActor


@dataclass
class SACConfig:
    hidden: int = 256
    critic_blocks: int = 2
    actor_hidden: int = 128
    actor_blocks: int = 1
    n_critics: int = 2
    n_quantiles: int = 25
    drop_quantiles: int = 2
    gamma: float = 0.99
    tau: float = 0.005
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 256
    init_alpha: float = 0.1
    target_entropy_scale: float = 1.0
    risk: str = "mean"  # "mean" or "cvar"
    cvar_eta: float = 0.25
    grad_clip: float = 10.0
    # Weight of a temporal-consistency penalty on the deterministic action between consecutive
    # states (CAPS-style). Oscillating targets pay spread and fees on every flip.
    smooth_coef: float = 0.0
    # Dropout rate inside critic residual blocks (0 disables).
    critic_dropout: float = 0.0
    # Std of Gaussian noise added to normalized observations in updates (0 disables).
    obs_noise: float = 0.0
    # "quantile": truncated quantile critics (TQC). "hlgauss": categorical critics trained by
    # cross-entropy on HL-Gauss projected targets (Farebrother et al., 2024), with clipped
    # double-Q targets over the ensemble.
    critic_type: str = "quantile"
    hl_bins: int = 101
    hl_vmax: float = 10.0
    # Position levels for categorical SAC; None keeps the continuous tanh-Gaussian policy.
    discrete_levels: tuple[float, ...] | None = None


class _QuantileHuber(torch.autograd.Function):
    """Quantile Huber loss with an analytic gradient.

    The pairwise residual tensor has shape (E, B, M, K); letting autograd record the half dozen
    elementwise ops on it costs more than the rest of the update, so the gradient
    ``-|tau - 1{u < 0}| * clip(u, -kappa, kappa) / kappa`` is formed in the forward pass instead.
    """

    @staticmethod
    def forward(ctx, pred: torch.Tensor, target: torch.Tensor, kappa: float) -> torch.Tensor:
        m = pred.shape[-1]
        tau = ((torch.arange(m, dtype=pred.dtype, device=pred.device) + 0.5) / m)[None, None, :, None]
        u = target[None, :, None, :] - pred[..., None]
        neg = u < 0
        w = torch.where(neg, 1.0 - tau, tau)
        clipped = u.clamp(-kappa, kappa)
        # huber(u) = clip * (|u| - |clip| / 2), which equals u^2/2 inside and kappa(|u| - kappa/2) outside.
        huber = clipped * (u - 0.5 * clipped)
        n = u.shape[1] * u.shape[2] * u.shape[3]
        loss = (w * huber).sum() / (n * kappa)
        ctx.save_for_backward(-(w * clipped).sum(-1) / (n * kappa))
        return loss

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (grad_pred,) = ctx.saved_tensors
        return grad_out * grad_pred, None, None


def quantile_huber(pred: torch.Tensor, target: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
    """pred (E, B, M), target (B, K) -> loss averaged over (B, M, K), summed over members.
    No gradient flows to ``target``."""
    return _QuantileHuber.apply(pred, target.detach(), kappa)


class TQCSAC:
    def __init__(self, obs_dim: int, act_dim: int = 1, cfg: SACConfig | None = None, seed: int = 0):
        self.cfg = cfg = cfg or SACConfig()
        torch.manual_seed(seed)
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.obs_norm = RunningNorm(obs_dim)
        self.actor = TanhGaussianActor(obs_dim, act_dim, cfg.actor_hidden, cfg.actor_blocks)
        self.critic = self._build_critic()
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
        self.log_alpha = nn.Parameter(torch.tensor(float(np.log(cfg.init_alpha))))
        self.target_entropy = -float(act_dim) * cfg.target_entropy_scale
        self._make_optimizers()
        self.updates = 0

    def _build_critic(self) -> nn.Module:
        c = self.cfg
        if c.critic_type == "hlgauss":
            return CategoricalCritic(self.obs_dim, self.act_dim, c.n_critics, c.hl_bins,
                                     -c.hl_vmax, c.hl_vmax, c.hidden, c.critic_blocks,
                                     c.critic_dropout)
        if c.critic_type != "quantile":
            raise ValueError(c.critic_type)
        return QuantileCritic(self.obs_dim, self.act_dim, c.n_critics, c.n_quantiles,
                              c.hidden, c.critic_blocks, c.critic_dropout)

    def _make_optimizers(self) -> None:
        c = self.cfg
        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=c.lr_actor,
                                           weight_decay=c.weight_decay, fused=True)
        self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=c.lr_critic,
                                            weight_decay=c.weight_decay, fused=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=c.lr_alpha)

    @property
    def alpha(self) -> float:
        return float(self.log_alpha.detach().exp())

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        x = self.obs_norm(torch.as_tensor(obs, dtype=torch.float32))
        a, _ = self.actor(x, deterministic=deterministic, with_logp=False)
        return a.numpy()

    def _risk(self, z: torch.Tensor) -> torch.Tensor:
        """z (E, B, M) -> (B,) risk-adjusted value of the pooled quantiles."""
        pooled = z.permute(1, 0, 2).reshape(z.shape[1], -1)
        if self.cfg.risk == "mean":
            return pooled.mean(-1)
        if self.cfg.risk == "cvar":
            n_tail = max(1, int(round(self.cfg.cvar_eta * pooled.shape[-1])))
            return pooled.sort(dim=-1).values[:, :n_tail].mean(-1)
        raise ValueError(self.cfg.risk)

    def update(self, obs: np.ndarray, action: np.ndarray, reward: np.ndarray,
               next_obs: np.ndarray, terminal: np.ndarray) -> dict[str, float]:
        c = self.cfg
        f32 = torch.float32
        obs_t = torch.as_tensor(obs, dtype=f32)
        next_t = torch.as_tensor(next_obs, dtype=f32)
        self.obs_norm.update(obs_t)
        o = self.obs_norm(obs_t)
        o2 = self.obs_norm(next_t)
        if c.obs_noise > 0:
            # Input jitter: a smoothness prior for features that are mostly noise.
            o = o + c.obs_noise * torch.randn_like(o)
            o2 = o2 + c.obs_noise * torch.randn_like(o2)
        a = torch.as_tensor(action, dtype=f32).reshape(-1, self.act_dim)
        r = torch.as_tensor(reward, dtype=f32).reshape(-1, 1)
        not_done = 1.0 - torch.as_tensor(terminal, dtype=f32).reshape(-1, 1)
        alpha = self.log_alpha.exp().detach()

        hl = c.critic_type == "hlgauss"
        with torch.no_grad():
            a2, logp2 = self.actor(o2)
            if hl:
                q2 = self.critic_target.q(o2, a2).min(dim=0).values  # (B,)
                target = r[:, 0] + not_done[:, 0] * c.gamma * (q2 - alpha * logp2)
            else:
                zt = self.critic_target(o2, a2)  # (E, B, M)
                b = zt.shape[1]
                pooled = zt.permute(1, 0, 2).reshape(b, -1).sort(dim=-1).values
                keep = pooled.shape[-1] - c.drop_quantiles * c.n_critics
                target = r + not_done * c.gamma * (pooled[:, :keep] - alpha * logp2[:, None])

        z = self.critic(o, a)
        critic_loss = self.critic.hl.loss(z, target) if hl else quantile_huber(z, target)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), c.grad_clip)
        self.critic_opt.step()

        a_pi, logp = self.actor(o)
        if hl:
            q_pi = self.critic.q(o, a_pi).min(dim=0).values
        else:
            q_pi = self._risk(self.critic(o, a_pi))
        actor_loss = (alpha * logp - q_pi).mean()
        if c.smooth_coef > 0:
            mu, _ = self.actor(o, deterministic=True)
            mu_next, _ = self.actor(o2, deterministic=True)
            actor_loss = actor_loss + c.smooth_coef * (mu - mu_next).pow(2).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), c.grad_clip)
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_target.parameters(), strict=True):
                pt.lerp_(p, c.tau)
        self.updates += 1
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": float(alpha),
            "entropy": -logp.mean().item(),
            "q_mean": (self.critic.hl.value(z) if hl else z).mean().item(),
        }

    def reset_networks(self, seed: int) -> None:
        """Re-initialize actor and critics (keeping observation statistics) to restore plasticity."""
        torch.manual_seed(seed)
        c = self.cfg
        self.actor = TanhGaussianActor(self.obs_dim, self.act_dim, c.actor_hidden, c.actor_blocks)
        self.critic = self._build_critic()
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
        self._make_optimizers()

    def state_dict(self) -> dict:
        return {
            "cfg": asdict(self.cfg), "obs_dim": self.obs_dim, "act_dim": self.act_dim,
            "obs_norm": copy.deepcopy(self.obs_norm.state_dict()),
            "actor": copy.deepcopy(self.actor.state_dict()),
            "critic": copy.deepcopy(self.critic.state_dict()),
            "critic_target": copy.deepcopy(self.critic_target.state_dict()),
            "log_alpha": float(self.log_alpha.detach()), "updates": self.updates,
        }

    def load_state_dict(self, state: dict) -> None:
        self.obs_norm.load_state_dict(state["obs_norm"])
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        with torch.no_grad():
            self.log_alpha.fill_(state["log_alpha"])
        self.updates = state["updates"]

    @classmethod
    def from_state_dict(cls, state: dict) -> TQCSAC:
        agent = cls(state["obs_dim"], state["act_dim"], SACConfig(**state["cfg"]))
        agent.load_state_dict(state)
        return agent
