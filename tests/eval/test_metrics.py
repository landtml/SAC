import numpy as np
import pytest

from sactrader.eval.metrics import sharpe, summarize


def test_sharpe_annualizes_with_sqrt_bars_per_year():
    # Mean 0.02, deviations +-0.01, ddof=1 variance 2e-4: per-bar Sharpe is sqrt(2).
    s = summarize([0.03, 0.01], bars_per_year=4)
    assert s["sharpe"] == pytest.approx(2 * np.sqrt(2))
    # Std sqrt(2e-4) scaled by sqrt(4) gives sqrt(8e-4).
    assert s["ann_vol"] == pytest.approx(np.sqrt(8e-4))


def test_sharpe_helper_is_zero_for_constant_rows():
    x = np.array([[0.03, 0.01], [0.0, 0.0]])
    np.testing.assert_allclose(sharpe(x, 1.0), [np.sqrt(2), 0.0])


def test_sortino_uses_downside_deviation_over_all_bars():
    # Mean 0.01. Downside deviation sqrt(mean([0, 1e-4])) = sqrt(5e-5), so Sortino is sqrt(2).
    s = summarize([0.03, -0.01], bars_per_year=1)
    assert s["sortino"] == pytest.approx(np.sqrt(2))
    # Std: deviations +-0.02 give ddof=1 variance 8e-4, so Sharpe is 0.01 / sqrt(8e-4).
    assert s["sharpe"] == pytest.approx(1.0 / (2 * np.sqrt(2)))


def test_drawdown_and_total_return_on_known_path():
    # Equity 1, 1.1, 0.55, 0.66. Peak 1.1, trough 0.55 gives -50%.
    s = summarize([0.1, -0.5, 0.2], bars_per_year=1)
    assert s["max_drawdown"] == pytest.approx(-0.5)
    assert s["total_return"] == pytest.approx(-0.34)
    assert s["hit_rate"] == pytest.approx(2 / 3)
    assert s["n_bars"] == 3.0


def test_cagr_uses_bars_over_bars_per_year():
    # Two bars at two bars per year is one year, so CAGR equals total return 1.21 - 1.
    s = summarize([0.1, 0.1], bars_per_year=2)
    assert s["cagr"] == pytest.approx(0.21)


def test_calmar_is_cagr_over_absolute_max_drawdown():
    s = summarize([0.1, -0.5, 0.2], bars_per_year=1)
    assert s["cagr"] == pytest.approx(0.66 ** (1 / 3) - 1)
    assert s["calmar"] == pytest.approx(s["cagr"] / 0.5)


def test_skew_and_excess_kurtosis_of_symmetric_three_point_series():
    # Biased moments: m2 = 2/3 * 1e-4, m4 = 2/3 * 1e-8, so m4 / m2**2 - 3 = -1.5.
    s = summarize([-0.01, 0.0, 0.01], bars_per_year=1)
    assert s["skew"] == pytest.approx(0.0, abs=1e-12)
    assert s["kurtosis"] == pytest.approx(-1.5)


def test_zero_dispersion_gives_zero_ratios():
    s = summarize(np.zeros(5), bars_per_year=252)
    for key in ("sharpe", "sortino", "ann_vol", "calmar", "max_drawdown", "hit_rate", "skew"):
        assert s[key] == 0.0, key

    c = summarize(np.full(5, 0.01), bars_per_year=252)
    assert c["sharpe"] == 0.0
    assert c["sortino"] == 0.0


def test_summarize_rejects_short_or_nonfinite_series():
    with pytest.raises(ValueError):
        summarize([0.1], bars_per_year=252)
    with pytest.raises(ValueError):
        summarize([0.1, np.nan], bars_per_year=252)
