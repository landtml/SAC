"""Categorical Soft Actor-Critic over a finite set of position levels.

The policy is a softmax over ``levels``. Each critic member outputs Q for every level, so the soft
value of the next state is an exact sum over levels rather than a sampled estimate (Christodoulou,
2019). Replayed continuous actions in [-1, 1] snap to the nearest level, and the critics train on
that level.

The public interface matches ``TQCSAC``. ``act`` returns the chosen level as float32 with shape
(B, 1); the deterministic policy takes the argmax level.

Temperature: alpha is tuned toward ``target_entropy = target_entropy_scale * 0.6 * log(K)``, a
fraction of the entropy of the uniform policy over K levels.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict

import numpy as np
import torch
from torch import nn

from sactrader.agent.networks import RunningNorm, SimbaNet
from sactrader.agent.sac import SACConfig


def _require_discrete(state: dict) -> None:
    if state.get("kind") != "discrete":
        raise ValueError("state was not written by DiscreteSAC (missing kind='discrete')")


class DiscreteSAC:
    def __init__(self, obs_dim: int, act_dim: int = 1, cfg: SACConfig | None = None, seed: int = 0,
                 levels: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)):
        if act_dim != 1:
            raise ValueError("DiscreteSAC picks one position level, so act_dim must be 1")
        if len(levels) < 2 or not all(map(math.isfinite, levels)):
            raise ValueError("levels must hold at least two finite values")
        if any(b <= a for a, b in zip(levels, levels[1:], strict=False)):
            raise ValueError("levels must be strictly increasing")
        self.cfg = cfg = cfg or SACConfig()
        self.levels = tuple(float(v) for v in levels)
        self._levels_t = torch.tensor(self.levels, dtype=torch.float32)
        torch.manual_seed(seed)
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.obs_norm = RunningNorm(obs_dim)
        self._build_networks()
        self.log_alpha = nn.Parameter(torch.tensor(float(np.log(cfg.init_alpha))))
        self.target_entropy = cfg.target_entropy_scale * 0.6 * math.log(len(self.levels))
        self._make_optimizers()
        self.updates = 0

    def _build_networks(self) -> None:
        c, k = self.cfg, len(self.levels)
        # The default output gain starts the logits near zero, so the initial policy is close to
        # uniform.
        self.actor = SimbaNet(1, self.obs_dim, k, hidden=c.actor_hidden, blocks=c.actor_blocks)
        # Q targets are unbounded, so the critic output keeps unit gain as in QuantileCritic.
        self.critic = SimbaNet(c.n_critics, self.obs_dim, k, hidden=c.hidden,
                               blocks=c.critic_blocks, out_gain=1.0, dropout=c.critic_dropout)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)

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

    def _level_index(self, action: torch.Tensor) -> torch.Tensor:
        """(B,) continuous actions -> (B,) index of the nearest level, the lower one on a tie."""
        return (action[:, None] - self._levels_t).abs().argmin(dim=-1)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        logits = self.actor(self.obs_norm(torch.as_tensor(obs, dtype=torch.float32)))[0]
        if deterministic:
            idx = logits.argmax(dim=-1)
        else:
            idx = torch.distributions.Categorical(logits=logits).sample()
        return self._levels_t[idx].numpy()[:, None]

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
            o = o + c.obs_noise * torch.randn_like(o)
            o2 = o2 + c.obs_noise * torch.randn_like(o2)
        idx = self._level_index(torch.as_tensor(action, dtype=f32).reshape(-1))
        r = torch.as_tensor(reward, dtype=f32).reshape(-1)
        not_done = 1.0 - torch.as_tensor(terminal, dtype=f32).reshape(-1)
        alpha = self.log_alpha.exp().detach()

        with torch.no_grad():
            logp2 = torch.log_softmax(self.actor(o2)[0], dim=-1)
            # Minimum over members (clipped double-Q) curbs overestimation, as TQC's truncation does.
            q_targ = self.critic_target(o2).min(dim=0).values  # (B, K)
            v2 = (logp2.exp() * (q_targ - alpha * logp2)).sum(dim=-1)
            target = r + not_done * c.gamma * v2

        z = self.critic(o)  # (E, B, K)
        taken = idx.view(1, -1, 1).expand(z.shape[0], -1, 1)
        q_taken = z.gather(-1, taken).squeeze(-1)  # (E, B)
        critic_loss = (q_taken - target).pow(2).mean(dim=1).sum()
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), c.grad_clip)
        self.critic_opt.step()

        logp = torch.log_softmax(self.actor(o)[0], dim=-1)
        pi = logp.exp()
        entropy = -(pi * logp).sum(dim=-1).detach()  # (B,), exact over levels
        q_pi = self.critic(o).min(dim=0).values  # (B, K)
        actor_loss = (pi * (alpha * logp - q_pi)).sum(dim=-1).mean()
        if c.smooth_coef > 0:
            # Discrete form of the temporal-consistency penalty: change in expected position.
            pi_next = torch.softmax(self.actor(o2)[0], dim=-1)
            gap = (pi - pi_next) @ self._levels_t
            actor_loss = actor_loss + c.smooth_coef * gap.pow(2).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), c.grad_clip)
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (self.target_entropy - entropy)).mean()
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
            "entropy": entropy.mean().item(),
            "q_mean": q_taken.mean().item(),
        }

    def reset_networks(self, seed: int) -> None:
        """Re-initialize actor and critics (keeping observation statistics) to restore plasticity."""
        torch.manual_seed(seed)
        self._build_networks()
        self._make_optimizers()

    def state_dict(self) -> dict:
        return {
            "kind": "discrete", "levels": list(self.levels),
            "cfg": asdict(self.cfg), "obs_dim": self.obs_dim, "act_dim": self.act_dim,
            "obs_norm": copy.deepcopy(self.obs_norm.state_dict()),
            "actor": copy.deepcopy(self.actor.state_dict()),
            "critic": copy.deepcopy(self.critic.state_dict()),
            "critic_target": copy.deepcopy(self.critic_target.state_dict()),
            "log_alpha": float(self.log_alpha.detach()), "updates": self.updates,
        }

    def load_state_dict(self, state: dict) -> None:
        _require_discrete(state)
        if tuple(float(v) for v in state["levels"]) != self.levels:
            raise ValueError("state levels do not match this agent's levels")
        self.obs_norm.load_state_dict(state["obs_norm"])
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        with torch.no_grad():
            self.log_alpha.fill_(state["log_alpha"])
        self.updates = state["updates"]

    @classmethod
    def from_state_dict(cls, state: dict) -> DiscreteSAC:
        _require_discrete(state)
        agent = cls(state["obs_dim"], state["act_dim"], SACConfig(**state["cfg"]),
                    levels=tuple(state["levels"]))
        agent.load_state_dict(state)
        return agent
