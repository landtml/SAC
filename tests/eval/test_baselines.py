import numpy as np

from sactrader.eval.baselines import buy_and_hold, flat, random_exposures, vol_target_trend

from .helpers import make_tape


def _walk(n: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))


def test_flat_and_buy_and_hold_have_one_value_per_row():
    tape = make_tape(_walk(50))
    assert np.all(flat(tape) == 0.0) and flat(tape).shape == (len(tape),)
    bh = buy_and_hold(tape, leverage=0.5)
    assert bh.shape == (len(tape),)
    assert np.all(bh == 0.5)


def test_vol_target_trend_is_zero_in_warmup_and_positive_on_a_rising_path():
    close = 100.0 * 1.001 ** np.arange(1000)
    exp = vol_target_trend(make_tape(close))
    lookback = 24 * 30
    assert np.all(exp[:lookback] == 0.0)
    assert np.all(exp[lookback:] > 0.0)
    assert np.all(np.abs(exp) <= 1.0)  # spot leverage limit


def test_vol_target_trend_is_negative_on_a_falling_path():
    close = 100.0 * 0.999 ** np.arange(1000)
    exp = vol_target_trend(make_tape(close))
    assert np.all(exp[24 * 30 :] < 0.0)


def test_vol_target_trend_ignores_future_closes():
    close = _walk(1000)
    changed = close.copy()
    changed[901:] *= 1.3
    before = vol_target_trend(make_tape(close))
    after = vol_target_trend(make_tape(changed))
    np.testing.assert_array_equal(before[:901], after[:901])
    assert np.any(before[901:] != after[901:])


def test_random_exposures_hold_in_blocks_and_depend_on_seed():
    tape = make_tape(_walk(200))
    exp = random_exposures(tape, seed=3, hold=24)
    assert exp.shape == (len(tape),)
    assert set(np.unique(exp)) <= {-1.0, 1.0}
    for start in range(0, len(exp), 24):
        block = exp[start : start + 24]
        assert np.all(block == block[0])
    np.testing.assert_array_equal(exp, random_exposures(tape, seed=3, hold=24))
    assert not np.array_equal(exp, random_exposures(tape, seed=4, hold=24))
