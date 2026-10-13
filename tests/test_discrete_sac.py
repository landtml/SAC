import inspect
import math

import numpy as np
import pytest
import torch

from sactrader.agent.discrete_sac import DiscreteSAC
from sactrader.agent.sac import TQCSAC, SACConfig

LEVELS = (-1.0, -0.5, 0.0, 0.5, 1.0)


def _small_cfg(**overrides) -> SACConfig:
    return SACConfig(**(dict(hidden=32, critic_blocks=1, actor_hidden=16, batch_size=32) | overrides))


@pytest.mark.parametrize("name", ["act", "update", "reset_networks", "state_dict",
                                  "load_state_dict", "_make_optimizers", "from_state_dict"])
def test_signatures_match_tqcsac(name):
    got = inspect.signature(getattr(DiscreteSAC, name)).parameters
    want = inspect.signature(getattr(TQCSAC, name)).parameters
    assert list(got) == list(want)


def test_attributes_match_tqcsac():
    tq, dq = TQCSAC(4, 1, _small_cfg()), DiscreteSAC(4, 1, _small_cfg())
    for name in ["cfg", "obs_dim", "act_dim", "obs_norm", "actor", "critic", "critic_target",
                 "log_alpha", "target_entropy", "updates"]:
        assert hasattr(tq, name) and hasattr(dq, name), name
    assert isinstance(TQCSAC.alpha, property)
    assert isinstance(DiscreteSAC.alpha, property)
    assert dq.alpha == pytest.approx(_small_cfg().init_alpha)


def test_target_entropy_is_scaled_fraction_of_uniform_entropy():
    agent = DiscreteSAC(3, cfg=_small_cfg(target_entropy_scale=0.5), levels=(-1.0, 0.0, 1.0))
    assert agent.target_entropy == pytest.approx(0.5 * 0.6 * math.log(3))


@pytest.mark.parametrize("levels", [(0.0,), (1.0, 0.5), (0.0, 0.0, 1.0), (0.0, float("nan"))])
def test_invalid_levels_are_rejected(levels):
    with pytest.raises(ValueError):
        DiscreteSAC(3, cfg=_small_cfg(), levels=levels)


def test_actions_map_to_nearest_level():
    agent = DiscreteSAC(3, cfg=_small_cfg(), levels=LEVELS)
    actions = torch.tensor([-1.0, -0.7, -0.2, 0.2, 0.26, 1.7, -3.0])
    assert agent._level_index(actions).tolist() == [0, 1, 2, 2, 3, 4, 0]


def test_nearest_level_respects_custom_levels():
    agent = DiscreteSAC(3, cfg=_small_cfg(), levels=(0.0, 0.1, 1.0))
    assert agent._level_index(torch.tensor([0.6, 0.04])).tolist() == [2, 0]


@pytest.mark.parametrize("deterministic", [False, True])
def test_act_returns_level_values(deterministic):
    agent = DiscreteSAC(4, cfg=_small_cfg(), seed=0, levels=LEVELS)
    obs = np.random.default_rng(0).normal(size=(64, 4)).astype(np.float32)
    out = agent.act(obs, deterministic=deterministic)
    assert out.shape == (64, 1)
    assert np.isin(out, np.float32(LEVELS)).all()


def test_stochastic_act_samples_every_level_from_near_uniform_policy():
    agent = DiscreteSAC(4, cfg=_small_cfg(), seed=0, levels=LEVELS)
    obs = np.random.default_rng(0).normal(size=(512, 4)).astype(np.float32)
    assert len(np.unique(agent.act(obs))) == len(LEVELS)


def test_deterministic_act_is_argmax_level():
    agent = DiscreteSAC(4, cfg=_small_cfg(), seed=0)
    obs = np.random.default_rng(0).normal(size=(32, 4)).astype(np.float32)
    with torch.no_grad():
        logits = agent.actor(agent.obs_norm(torch.as_tensor(obs)))[0]
    expected = np.float32(agent.levels)[logits.argmax(dim=-1).numpy()][:, None]
    np.testing.assert_array_equal(agent.act(obs, deterministic=True), expected)


def test_update_returns_finite_stats():
    rng = np.random.default_rng(1)
    agent = DiscreteSAC(4, cfg=_small_cfg(), seed=0)
    obs = rng.normal(size=(32, 4)).astype(np.float32)
    for _ in range(5):
        stats = agent.update(obs, rng.uniform(-1, 1, 32), rng.normal(size=32), obs,
                             rng.random(32) < 0.1)
    assert set(stats) == {"critic_loss", "actor_loss", "alpha", "entropy", "q_mean"}
    assert all(np.isfinite(v) for v in stats.values())
    assert agent.updates == 5


def test_state_dict_roundtrip_gives_identical_deterministic_actions():
    rng = np.random.default_rng(2)
    agent = DiscreteSAC(4, cfg=_small_cfg(), seed=0, levels=(-1.0, 0.0, 1.0))
    obs = rng.normal(size=(32, 4)).astype(np.float32)
    for _ in range(5):
        agent.update(obs, rng.uniform(-1, 1, 32), rng.normal(size=32), obs, np.zeros(32, bool))
    state = agent.state_dict()
    assert state["kind"] == "discrete"
    assert state["levels"] == [-1.0, 0.0, 1.0]
    clone = DiscreteSAC.from_state_dict(state)
    assert clone.levels == agent.levels
    np.testing.assert_array_equal(clone.act(obs, deterministic=True),
                                  agent.act(obs, deterministic=True))


def test_from_state_dict_rejects_tqcsac_state():
    with pytest.raises(ValueError):
        DiscreteSAC.from_state_dict(TQCSAC(4, 1, _small_cfg()).state_dict())


def test_bandit_learns_the_rewarded_level():
    agent = DiscreteSAC(3, cfg=_small_cfg(), seed=0)
    rng = np.random.default_rng(0)
    obs = np.ones((64, 3), np.float32)
    done = np.ones(64, bool)
    for _ in range(400):
        action = rng.uniform(-1, 1, 64)
        reward = (action > 0.75).astype(np.float64)  # nearest level is 1.0 above the 0.75 midpoint
        agent.update(obs, action, reward, obs, done)
    np.testing.assert_array_equal(agent.act(obs, deterministic=True), np.ones((64, 1), np.float32))
