"""Training loop: vectorized rollouts, counterfactual replay, validation-based checkpointing."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch

from sactrader.agent.factory import make_agent
from sactrader.agent.replay import IndexReplay, materialize, relabel
from sactrader.agent.sac import TQCSAC, SACConfig
from sactrader.env.vec_env import EnvConfig, TapeSet, VecTradingEnv, rollout
from sactrader.tape import MarketTape


@dataclass
class TrainConfig:
    total_steps: int = 200_000  # environment transitions
    warmup_steps: int = 5_000  # uniform random actions before learning starts
    utd: float = 0.25  # gradient updates per environment transition
    relabel_frac: float = 0.5
    replay_capacity: int = 1_000_000
    reward_scale: float | None = None  # None: derive from training tapes
    eval_every: int = 2_000  # gradient updates between validation runs
    reset_every: int | None = None  # gradient updates between network resets
    # Anneal the entropy target from sac.target_entropy_scale to this value over training.
    target_entropy_end_scale: float | None = None
    # Restore the best checkpoint when the mean |Q| exceeds this (scaled rewards keep |Q| near
    # 1 / (1 - gamma) at most) or a loss is not finite.
    q_abs_limit: float = 100.0
    # "cosine" decays actor and critic learning rates to lr_min_frac of their start value.
    lr_schedule: str | None = None
    lr_min_frac: float = 0.1
    seed: int = 0
    threads: int = 4
    env: EnvConfig = field(default_factory=EnvConfig)
    sac: SACConfig = field(default_factory=SACConfig)


@dataclass
class TrainResult:
    agent: TQCSAC
    best_state: dict
    best_score: float
    history: list[dict]
    feature_names: tuple[str, ...]
    reward_scale: float


def auto_reward_scale(tapes: list[MarketTape], gamma: float) -> float:
    """Scale that gives an unlevered position's discounted return a standard deviation near 1.

    Uses the median over tapes of a robust per-bar log-return scale (1.4826 * MAD), so one tape with
    jumps (a binary settlement) does not set the scale for all.
    """
    scales = []
    for tp in tapes:
        r = np.log1p(np.maximum(tp.long_ret[:-1], -0.999))
        mad = np.median(np.abs(r - np.median(r))) * 1.4826
        if mad > 0:
            scales.append(mad)
    sigma = float(np.median(scales)) if scales else 0.01
    return float(np.sqrt(1.0 - gamma**2) / sigma)


def sharpe(returns: np.ndarray, bars_per_year: float) -> float:
    if len(returns) < 2:
        return 0.0
    sd = returns.std(ddof=1)
    return 0.0 if sd == 0 else float(returns.mean() / sd * np.sqrt(bars_per_year))


def validation_score(agent: TQCSAC, val: TapeSet) -> tuple[float, dict]:
    """Mean annualized Sharpe of the deterministic policy across validation tapes."""
    out = rollout(val, lambda o: agent.act(o, deterministic=True))
    sh = [sharpe(o["ret"], tp.bars_per_year) for o, tp in zip(out, val.tapes, strict=True)]
    growth = [float(np.log1p(np.maximum(o["ret"], -0.999)).sum()) for o in out]
    turnover = [float(o["turnover"].mean()) for o in out]
    return float(np.mean(sh)), {
        "val_sharpe": float(np.mean(sh)),
        "val_log_growth": float(np.mean(growth)),
        "val_turnover": float(np.mean(turnover)),
    }


def train(train_tapes: list[MarketTape], val_tapes: list[MarketTape] | None, cfg: TrainConfig,
          log: Callable[[dict], None] | None = None, init_state: dict | None = None,
          feature_names: tuple[str, ...] | None = None) -> TrainResult:
    """Train one agent. ``init_state`` warm-starts from an earlier agent (its warm-up steps then
    follow that agent's policy instead of uniform actions); ``feature_names`` pins the
    observation layout so the warm-started networks see the same inputs."""
    torch.set_num_threads(cfg.threads)
    rng = np.random.default_rng(cfg.seed)
    ts = TapeSet(train_tapes, cfg.env.window, feature_names)
    val = TapeSet(val_tapes, cfg.env.window, ts.feature_names) if val_tapes else None
    env = VecTradingEnv(ts, cfg.env, seed=cfg.seed)
    agent = make_agent(ts.obs_dim, cfg.sac, cfg.seed)
    if init_state is not None:
        agent.load_state_dict(init_state)
    replay = IndexReplay(cfg.replay_capacity)
    scale = cfg.reward_scale or auto_reward_scale(train_tapes, cfg.sac.gamma)
    planned_updates = max(1.0, (cfg.total_steps - cfg.warmup_steps) * cfg.utd)
    ent_start = cfg.sac.target_entropy_scale
    ent_end = cfg.target_entropy_end_scale
    n_updates = 0
    restores = 0

    history: list[dict] = []
    best_score, best_state = -np.inf, agent.state_dict()
    owed = 0.0
    steps = 0
    last_eval = 0
    t0 = time.time()
    stats: dict[str, float] = {}
    while steps < cfg.total_steps:
        if steps < cfg.warmup_steps and init_state is None:
            action = rng.uniform(-1.0, 1.0, cfg.env.n_envs)
        else:
            action = agent.act(env.observe())[:, 0]
        replay.add(env.step(action))
        steps += cfg.env.n_envs
        if steps < cfg.warmup_steps:
            continue
        owed += cfg.utd * cfg.env.n_envs
        while owed >= 1.0:
            owed -= 1.0
            sample = replay.sample(cfg.sac.batch_size, rng)
            relabel(sample, ts, cfg.relabel_frac, rng)
            obs, next_obs = materialize(sample, ts)
            frac = min(1.0, n_updates / planned_updates)
            if ent_end is not None:
                agent.target_entropy = -(ent_start + (ent_end - ent_start) * frac)
            if cfg.lr_schedule == "cosine":
                mult = cfg.lr_min_frac + (1 - cfg.lr_min_frac) * 0.5 * (1 + np.cos(np.pi * frac))
                for opt, base in ((agent.actor_opt, cfg.sac.lr_actor),
                                  (agent.critic_opt, cfg.sac.lr_critic)):
                    for group in opt.param_groups:
                        group["lr"] = base * mult
            stats = agent.update(obs, sample["action"], sample["reward"] * scale, next_obs,
                                 sample["terminal"])
            n_updates += 1
            if not np.isfinite(stats["critic_loss"]) or abs(stats["q_mean"]) > cfg.q_abs_limit:
                # Value divergence: growing |Q| does not recover on its own (arXiv 2609.32819).
                agent.load_state_dict(best_state)
                agent._make_optimizers()
                restores += 1
            if cfg.reset_every and agent.updates % cfg.reset_every == 0:
                agent.reset_networks(cfg.seed + agent.updates)
        if agent.updates - last_eval >= cfg.eval_every:
            last_eval = agent.updates
            rec = {"steps": steps, "updates": agent.updates, "time": time.time() - t0,
                   "restores": restores, "target_entropy": agent.target_entropy, **stats}
            recent = env.completed_returns[-200:]
            rec["train_ep_log_growth"] = float(np.mean(recent)) if recent else float("nan")
            if val is not None:
                score, vstats = validation_score(agent, val)
                rec.update(vstats)
                if score > best_score:
                    best_score, best_state = score, agent.state_dict()
            history.append(rec)
            if log:
                log(rec)
    if val is None:
        best_state = agent.state_dict()
    return TrainResult(agent, best_state, float(best_score), history, ts.feature_names, scale)
