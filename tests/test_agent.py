import numpy as np
import pytest
import torch

from sactrader.agent.networks import RunningNorm, TanhGaussianActor
from sactrader.agent.sac import TQCSAC, SACConfig, quantile_huber


def _reference_quantile_huber(pred, target, kappa=1.0):
    m = pred.shape[-1]
    tau = (torch.arange(m, dtype=pred.dtype) + 0.5) / m
    u = target[None, :, None, :] - pred[..., None]
    huber = torch.where(u.abs() <= kappa, 0.5 * u**2, kappa * (u.abs() - 0.5 * kappa))
    w = (tau[None, None, :, None] - (u.detach() < 0).to(pred.dtype)).abs()
    return (w * huber / kappa).mean(dim=(1, 2, 3)).sum()


@pytest.mark.parametrize("kappa", [1.0, 0.3])
def test_quantile_huber_matches_autograd_reference(kappa):
    g = torch.Generator().manual_seed(0)
    pred = torch.randn(3, 16, 5, generator=g, dtype=torch.float64, requires_grad=True)
    target = 2 * torch.randn(16, 9, generator=g, dtype=torch.float64)
    ref = _reference_quantile_huber(pred, target, kappa)
    (g_ref,) = torch.autograd.grad(ref, pred)
    got = quantile_huber(pred, target, kappa)
    (g_got,) = torch.autograd.grad(got, pred)
    assert float(got) == pytest.approx(float(ref), rel=1e-10)
    torch.testing.assert_close(g_got, g_ref)


def test_running_norm_matches_numpy():
    x = np.random.default_rng(0).normal(3.0, 2.0, size=(1000, 4))
    rn = RunningNorm(4)
    for chunk in np.array_split(x, 7):
        rn.update(torch.as_tensor(chunk))
    np.testing.assert_allclose(rn.mean.numpy(), x.mean(0), rtol=1e-10)
    np.testing.assert_allclose((rn.m2 / (rn.count - 1)).numpy(), x.var(0, ddof=1), rtol=1e-10)


def test_tanh_gaussian_logp_matches_change_of_variables():
    torch.manual_seed(0)
    actor = TanhGaussianActor(3, 1, hidden=16, blocks=1)
    obs = torch.randn(4096, 3)
    a, logp = actor(obs)
    assert (a.abs() < 1).all()
    out = actor.net(obs)[0]
    mean, log_std = out.split(1, dim=-1)
    lo, hi = actor.log_std_min, actor.log_std_max
    std = (lo + 0.5 * (hi - lo) * (torch.tanh(log_std) + 1)).exp()
    z = torch.atanh(a.clamp(-1 + 1e-6, 1 - 1e-6))
    ref = torch.distributions.Normal(mean, std).log_prob(z).sum(-1) - torch.log1p(-a.pow(2) + 1e-12).sum(-1)
    torch.testing.assert_close(logp, ref, atol=1e-3, rtol=1e-3)


def test_update_runs_and_state_roundtrip():
    rng = np.random.default_rng(0)
    cfg = SACConfig(hidden=32, critic_blocks=1, actor_hidden=16, batch_size=32, risk="cvar")
    agent = TQCSAC(6, 1, cfg, seed=0)
    obs = rng.normal(size=(32, 6)).astype(np.float32)
    for _ in range(5):
        stats = agent.update(obs, rng.uniform(-1, 1, 32), rng.normal(size=32), obs, np.zeros(32, bool))
    assert all(np.isfinite(v) for v in stats.values())
    clone = TQCSAC.from_state_dict(agent.state_dict())
    np.testing.assert_allclose(clone.act(obs, deterministic=True), agent.act(obs, deterministic=True))


def test_smoothness_penalty_runs_and_shrinks_action_gap():
    rng = np.random.default_rng(1)
    obs = rng.normal(size=(64, 4)).astype(np.float32)
    nxt = (obs + rng.normal(scale=2.0, size=obs.shape)).astype(np.float32)
    gaps = {}
    for coef in (0.0, 5.0):
        cfg = SACConfig(hidden=32, critic_blocks=1, actor_hidden=16, smooth_coef=coef, lr_actor=3e-3)
        agent = TQCSAC(4, 1, cfg, seed=0)
        for _ in range(150):
            agent.update(obs, rng.uniform(-1, 1, 64), rng.normal(size=64), nxt, np.zeros(64, bool))
        gaps[coef] = float(np.abs(agent.act(obs, True) - agent.act(nxt, True)).mean())
    assert gaps[5.0] < gaps[0.0]


def test_critic_learns_constant_reward_value():
    """With reward 1 every step, no termination and a near-zero entropy bonus, the soft value
    approaches 1 / (1 - gamma)."""
    cfg = SACConfig(hidden=32, critic_blocks=1, actor_hidden=16, gamma=0.5, init_alpha=1e-4,
                    lr_alpha=0.0, lr_critic=3e-3, drop_quantiles=0, tau=0.05)
    agent = TQCSAC(2, 1, cfg, seed=0)
    obs = np.random.default_rng(0).normal(size=(64, 2)).astype(np.float32)
    for _ in range(600):
        stats = agent.update(obs, np.zeros(64), np.ones(64), obs, np.zeros(64, bool))
    assert stats["q_mean"] == pytest.approx(2.0, abs=0.15)


def test_dropout_and_obs_noise_options_train():
    rng = np.random.default_rng(2)
    cfg = SACConfig(hidden=32, critic_blocks=1, actor_hidden=16, critic_dropout=0.1, obs_noise=0.1)
    agent = TQCSAC(4, 1, cfg, seed=0)
    assert any(isinstance(m, torch.nn.Dropout) for m in agent.critic.modules())
    obs = rng.normal(size=(32, 4)).astype(np.float32)
    for _ in range(5):
        stats = agent.update(obs, rng.uniform(-1, 1, 32), rng.normal(size=32), obs, np.zeros(32, bool))
    assert all(np.isfinite(v) for v in stats.values())


def test_hlgauss_critic_learns_constant_reward_value():
    cfg = SACConfig(hidden=32, critic_blocks=1, actor_hidden=16, gamma=0.5, init_alpha=1e-4,
                    lr_alpha=0.0, lr_critic=3e-3, tau=0.05, critic_type="hlgauss", hl_bins=51,
                    hl_vmax=5.0)
    agent = TQCSAC(2, 1, cfg, seed=0)
    obs = np.random.default_rng(0).normal(size=(64, 2)).astype(np.float32)
    for _ in range(600):
        stats = agent.update(obs, np.zeros(64), np.ones(64), obs, np.zeros(64, bool))
    assert stats["q_mean"] == pytest.approx(2.0, abs=0.2)
    clone = TQCSAC.from_state_dict(agent.state_dict())
    np.testing.assert_allclose(clone.act(obs, True), agent.act(obs, True))
