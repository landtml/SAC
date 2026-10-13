"""Pick the agent class from the config, and rebuild agents from saved state."""

from __future__ import annotations

from softedge.agent.sac import TQCSAC, SACConfig


def make_agent(obs_dim: int, cfg: SACConfig, seed: int):
    """TQC-SAC with a tanh-Gaussian policy, or categorical SAC when ``cfg.discrete_levels``."""
    if cfg.discrete_levels:
        from softedge.agent.discrete_sac import DiscreteSAC

        return DiscreteSAC(obs_dim, 1, cfg, seed=seed, levels=tuple(cfg.discrete_levels))
    return TQCSAC(obs_dim, 1, cfg, seed=seed)


def agent_from_state(state: dict):
    if state.get("kind") == "discrete":
        from softedge.agent.discrete_sac import DiscreteSAC

        return DiscreteSAC.from_state_dict(state)
    return TQCSAC.from_state_dict(state)
