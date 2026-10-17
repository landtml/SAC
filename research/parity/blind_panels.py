"""Weekly multi-asset panels for blind testing. Numpy only.

panel(family, seed, n_assets=10, n_weeks=1040) returns (logret, cash):
  logret  float array [n_assets, n_weeks], total weekly log return; NaN before listing.
  cash    float array [n_weeks], weekly cash log return.
Deterministic in seed through numpy.random.default_rng(seed).
"""

import numpy as np

FAMILIES = ("mom_rebound", "vol_premium", "slow_state", "tails_contagion")

_RHO_COMMON = 0.4
_VOL_LO, _VOL_HI = 0.01, 0.12
_LISTED_FRACTION = 0.3
_C_MOM = 0.02
_C_VOL = 0.03
_C_SLOW = 0.02

_PARAMS = {
    "mom_rebound": dict(df=4.0, jump_mean=0.03, jump_base=0.015, jump_feedback=0.08,
                        jump_decay=0.8),
    "vol_premium": dict(df=4.0, jump_mean=0.03, jump_base=0.015, jump_feedback=0.08,
                        jump_decay=0.8),
    "slow_state": dict(df=4.0, jump_mean=0.03, jump_base=0.015, jump_feedback=0.08,
                       jump_decay=0.8),
    "tails_contagion": dict(df=3.0, jump_mean=0.05, jump_base=0.015, jump_feedback=0.15,
                            jump_decay=0.8),
}


def _std_t(rng, df, size):
    return rng.standard_t(df, size) * np.sqrt((df - 2.0) / df)


def _ar1_unit(innov, phi):
    # Starts from the stationary law, so every entry has unit variance.
    out = np.empty(innov.size)
    out[0] = innov[0]
    for t in range(1, innov.size):
        out[t] = phi * out[t - 1] + np.sqrt(1.0 - phi * phi) * innov[t]
    return out


def panel(family, seed, n_assets=10, n_weeks=1040):
    if family not in _PARAMS:
        raise ValueError(f"family must be one of {FAMILIES}, got {family!r}")
    p = _PARAMS[family]
    rng = np.random.default_rng(seed)
    n, T = n_assets, n_weeks

    # Draw order is fixed, so a seed always gives the same panel.
    f = _std_t(rng, p["df"], T)
    u = _std_t(rng, p["df"], (T, n))
    eps = np.sqrt(_RHO_COMMON) * f[:, None] + np.sqrt(1.0 - _RHO_COMMON) * u
    vol_innov = rng.standard_normal(T)
    jump_u = rng.random(T)
    jump_sign = np.where(rng.random(T) < 0.5, 1.0, -1.0)
    jump_size = rng.exponential(p["jump_mean"], T)
    state_innov = rng.standard_normal(T)
    cash_innov = rng.standard_normal(T)
    scale = np.exp(rng.uniform(np.log(0.02), np.log(0.07), n))
    load = rng.uniform(0.5, 1.5, n)
    sens = rng.uniform(-1.0, 1.0, n)
    late_assets = rng.choice(n, size=round(_LISTED_FRACTION * n), replace=False)
    late_weeks = rng.integers(1, T // 2, size=late_assets.size)

    # Log volatility multiplier. It reacts to last week's common shock, so vol is predetermined.
    lv = np.zeros(T)
    for t in range(1, T):
        lv[t] = 0.92 * lv[t - 1] + 0.2 * (abs(f[t - 1]) - np.sqrt(0.5)) + 0.08 * vol_innov[t - 1]
    vol = np.clip(scale[None, :] * np.exp(lv)[:, None], _VOL_LO, _VOL_HI)

    # Self-exciting common jumps, sign-symmetric. Jump size is the same sign for all assets
    # up to each asset's loading.
    jump = np.zeros(T)
    kcount = 0.0
    for t in range(T):
        prob = p["jump_base"] + p["jump_feedback"] * kcount
        hit = float(jump_u[t] < prob)
        jump[t] = jump_sign[t] * jump_size[t] * hit
        kcount = p["jump_decay"] * kcount + hit
    jump_ret = load[None, :] * jump[:, None]

    slow = _ar1_unit(state_innov, 0.99)
    cash_state = _ar1_unit(cash_innov, 0.99)
    cash = np.log1p((0.03 + 0.03 * np.tanh(cash_state)) / 52.0)

    exc = np.zeros((T, n))
    mkt = np.zeros(T)
    for t in range(T):
        if family == "mom_rebound":
            past = exc[max(0, t - 12):max(0, t - 1)].sum(axis=0)
            mom = past / (np.sqrt(11.0) * scale)
            recent = mkt[max(0, t - 4):t].sum()
            longer = mkt[max(0, t - 52):t].sum()
            flip = -1.0 if (recent > 0.05 and longer < 0.0) else 1.0
            mu = _C_MOM * vol[t] * flip * np.tanh(mom)
        elif family == "vol_premium":
            window = mkt[max(0, t - 8):t]
            rv = np.sqrt(np.mean(window ** 2)) if window.size else 0.03
            mu = _C_VOL * vol[t] * np.clip(1.0 - rv / 0.04, -1.0, 1.0)
        elif family == "slow_state":
            mu = _C_SLOW * vol[t] * sens * slow[t]
        else:
            mu = np.zeros(n)
        exc[t] = mu + vol[t] * eps[t] + jump_ret[t]
        mkt[t] = exc[t].mean()

    logret = (exc + cash[:, None]).T.copy()
    for a, w in zip(late_assets, late_weeks):
        logret[a, :w] = np.nan
    return logret, cash
