from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from softedge.data import fred_markets, tips
from softedge.data.schema import validate_frame
from softedge.features.causal import compute_features
from softedge.tape import CostSpec, build_tape

ROOT = Path(__file__).resolve().parents[2]


class FakeResponse:
    def __init__(self, content: bytes):
        self.content = content


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("SOFTEDGE_DATA", str(tmp_path))
    monkeypatch.setattr(fred_markets, "REQUEST_INTERVAL_S", 0.0)


def utc(day: str) -> pd.Timestamp:
    return pd.Timestamp(day, tz="UTC")


def monthly_cpi(first: str = "2008-01", last: str = "2011-12") -> pd.Series:
    idx = pd.date_range(f"{first}-01", f"{last}-01", freq="MS", tz="UTC")
    k = np.arange(len(idx), dtype=np.float64)
    # Distinct values per month, so a wrong month shows up in the output.
    return pd.Series(100.0 + 0.1 * k + 0.05 * (k % 3), index=idx)


def daily(first: str, last: str, value_fn) -> pd.Series:
    idx = pd.bdate_range(first, last, tz="UTC")
    return pd.Series(value_fn(np.arange(len(idx), dtype=np.float64)), index=idx)


def build(cpi: pd.Series, nominal=None, real=None, breakeven=None, **kw) -> pd.DataFrame:
    days = pd.bdate_range("2010-01-01", "2011-06-30", tz="UTC")
    n = len(days)
    if nominal is None:
        nominal = pd.Series(4.0 + 0.5 * np.sin(np.arange(n) / 20.0), index=days)
    if real is None:
        real = pd.Series(2.0 + 0.3 * np.cos(np.arange(n) / 25.0), index=days)
    if breakeven is None:
        breakeven = pd.Series(2.5 + 0.2 * np.sin(np.arange(n) / 15.0), index=days)
    kw.setdefault("duration", 8.6)
    kw.setdefault("convexity", 84.6)
    kw.setdefault("symbol", "BE10")
    return tips.breakeven_frame(nominal, real, breakeven, cpi, **kw)


def test_reference_cpi_interpolates_between_three_and_two_month_lags():
    cpi = monthly_cpi()
    # May 2010 uses February (three months back) and March (two months back).
    feb, mar = cpi[utc("2010-02-01")], cpi[utc("2010-03-01")]
    dates = pd.DatetimeIndex([utc("2010-05-01"), utc("2010-05-16"), utc("2010-05-31")])
    got = tips.reference_cpi(cpi, dates)
    np.testing.assert_allclose(got, [feb, feb + 15 / 31 * (mar - feb), feb + 30 / 31 * (mar - feb)])


def test_accrual_does_not_read_the_current_month_cpi():
    cpi = monthly_cpi()
    dates = pd.bdate_range("2010-05-03", "2010-05-28", tz="UTC")
    base = tips.reference_cpi(cpi, dates)
    same_month = cpi.copy()
    same_month[utc("2010-05-01")] *= 1.05
    np.testing.assert_array_equal(tips.reference_cpi(same_month, dates), base)
    # Positive control: the lagged months do feed the accrual.
    lagged = cpi.copy()
    lagged[utc("2010-02-01")] *= 1.05
    assert not np.allclose(tips.reference_cpi(lagged, dates), base)


def test_accrual_does_not_read_the_previous_months_cpi():
    cpi = monthly_cpi()
    # May reads February and March only, so April's print cannot reach the accrual on any day.
    dates = pd.bdate_range("2010-05-03", "2010-05-14", tz="UTC")
    base = tips.reference_cpi(cpi, dates)
    april = cpi.copy()
    april[utc("2010-04-01")] *= 1.02
    np.testing.assert_array_equal(tips.reference_cpi(april, dates), base)


def test_missing_cpi_month_carries_the_previous_value_forward():
    cpi = monthly_cpi().drop(utc("2010-03-01"))
    dates = pd.bdate_range("2010-05-03", "2010-05-28", tz="UTC")
    # March 2010 has no print, so May's reference CPI stays at February's value. Interpolating
    # toward a later print would read data not yet released.
    feb = monthly_cpi()[utc("2010-02-01")]
    np.testing.assert_allclose(tips.reference_cpi(cpi, dates), feb)


def test_prior_reads_cpi_only_from_its_release_stamp():
    cpi = monthly_cpi()
    dates = pd.DatetimeIndex([utc("2010-05-15"), utc("2010-05-16")])
    base = tips.trailing_inflation(cpi, dates)
    # April 2010 CPI is stamped on 16 May 2010, so the 15th may not see it and the 16th may.
    april = cpi.copy()
    april[utc("2010-04-01")] *= 1.02
    after = tips.trailing_inflation(april, dates)
    assert after[0] == base[0]
    assert after[1] != base[1]


def test_prior_does_not_read_the_current_month_cpi():
    cpi = monthly_cpi()
    dates = pd.DatetimeIndex([utc("2010-05-31")])
    base = tips.trailing_inflation(cpi, dates)
    same_month = cpi.copy()
    same_month[utc("2010-05-01")] *= 1.05
    np.testing.assert_array_equal(tips.trailing_inflation(same_month, dates), base)


def test_trailing_inflation_is_twelve_month_rate_in_percent():
    cpi = monthly_cpi()
    got = tips.trailing_inflation(cpi, pd.DatetimeIndex([utc("2010-05-16")]))
    # On 16 May the latest release is April 2010, compared with April 2009.
    expected = 100 * (cpi[utc("2010-04-01")] / cpi[utc("2009-04-01")] - 1)
    np.testing.assert_allclose(got, [expected])


def test_frame_rows_do_not_depend_on_data_after_their_bar():
    cpi = monthly_cpi()
    base = build(cpi)
    cut = utc("2011-01-14")
    nominal = pd.Series(4.0 + 0.5 * np.sin(np.arange(len(base)) / 20.0), index=base.index)
    real = pd.Series(2.0 + 0.3 * np.cos(np.arange(len(base)) / 25.0), index=base.index)
    be = pd.Series(2.5 + 0.2 * np.sin(np.arange(len(base)) / 15.0), index=base.index)
    # Change every observation after the cut, including CPI prints not yet released there.
    late_nom, late_real, late_be = nominal.copy(), real.copy(), be.copy()
    late_nom[late_nom.index > cut] += 1.0
    late_real[late_real.index > cut] -= 0.7
    late_be[late_be.index > cut] *= 1.3
    late_cpi = cpi.copy()
    late_cpi[late_cpi.index >= utc("2010-12-01")] *= 1.1
    changed = build(late_cpi, late_nom, late_real, late_be)
    before = base.index <= cut
    pd.testing.assert_frame_equal(base.loc[before], changed.loc[before])


def test_spread_return_is_carry_minus_accrual_with_no_yield_change():
    days = pd.DatetimeIndex([utc("2010-01-04"), utc("2010-01-05"), utc("2010-01-06")])
    flat = pd.Series([4.0, 4.0, 4.0], index=days)
    real = pd.Series([2.0, 2.0, 2.0], index=days)
    be = pd.Series([2.0, 2.0, 2.0], index=days)
    # Constant CPI: no accrual, so the spread earns (4% - 2%) per year, one calendar day at a time.
    cpi = pd.Series(100.0, index=pd.date_range("2008-01-01", "2010-12-01", freq="MS", tz="UTC"))
    df = tips.breakeven_frame(flat, real, be, cpi, duration=8.6, convexity=84.6, symbol="BE10")
    dt = 1 / 365
    np.testing.assert_allclose(df["close"].iloc[1] / df["close"].iloc[0] - 1, 0.02 * 1 * dt)
    np.testing.assert_allclose(df["close"].iloc[2] / df["close"].iloc[1] - 1, 0.02 * 1 * dt)


def test_short_spread_pays_the_carry_and_no_borrow_is_credited(monkeypatch):
    # Constant yields and CPI: the long spread earns 4% nominal less 2% real per year, and nothing
    # else. A short spread holds the negative of that return, so its tape pays the carry with
    # no borrow credit. The tape uses config costs: borrow_apr 0 on the spot branch.
    days = pd.bdate_range("2003-01-02", "2010-06-30")
    cpi_idx = pd.date_range("2001-01-01", "2010-06-01", freq="MS")
    serve(monkeypatch, {
        "DGS10": fred_csv("DGS10", days, np.full(len(days), 4.0)),
        "DFII10": fred_csv("DFII10", days, np.full(len(days), 2.0)),
        "T10YIE": fred_csv("T10YIE", days, np.full(len(days), 2.0)),
        "CPIAUCNS": fred_csv("CPIAUCNS", cpi_idx, np.full(len(cpi_idx), 100.0)),
    })
    df = tips.load_tips_breakeven("10", start="2008-01", end="2008-12")
    dt = df.index.to_series().diff().dt.total_seconds().to_numpy()[1:] / (365 * 86400)
    costs = CostSpec(fee_bps=2.0, half_spread_bps=2.0, borrow_apr=0.0, max_leverage=4.0)
    tape = build_tape(df, pd.DataFrame({"x": 0.0}, index=df.index), costs)
    n = len(dt)
    np.testing.assert_allclose(tape.long_ret[:n], 0.02 * dt, rtol=1e-9)
    np.testing.assert_allclose(tape.short_ret[:n], -0.02 * dt, rtol=1e-9)
    np.testing.assert_array_equal(tape.carry_short[:n], 0.0)


def test_breakeven_change_costs_about_the_duration_per_point():
    days = pd.bdate_range("2010-01-01", "2010-01-29", tz="UTC")
    cpi = pd.Series(100.0, index=pd.date_range("2008-01-01", "2010-12-01", freq="MS", tz="UTC"))
    nominal = pd.Series(4.0, index=days)
    real = pd.Series(2.0, index=days)
    # Real yields fall 0.10 point on the last day with nominal unchanged: breakeven rises 0.10.
    real_dn = real.copy()
    real_dn.iloc[-1] -= 0.10
    df = tips.breakeven_frame(nominal, real_dn, nominal - real, cpi, duration=8.6,
                              convexity=84.6, symbol="BE10")
    flat = tips.breakeven_frame(nominal, real, nominal - real, cpi, duration=8.6,
                                convexity=84.6, symbol="BE10")
    # A 0.10 point rise in breakeven, at duration 8.6, costs the spread about 0.86 percent.
    gap = np.log(df["close"].iloc[-1]) - np.log(flat["close"].iloc[-1])
    assert gap == pytest.approx(-8.6 * 0.001, rel=0.05)


def test_frame_is_a_valid_spot_frame_with_prior_column():
    df = build(monthly_cpi())
    validate_frame(df)
    assert df.attrs["kind"] == "spot"
    assert df.attrs["bar"] == "1D"
    assert df.attrs["symbol"] == "BE10"
    assert np.isfinite(df["tips_prior"]).all()


def test_prior_feature_passes_through_compute_features():
    df = build(monthly_cpi())
    feats = compute_features(df)
    assert "tips_prior" in feats.columns
    expected = np.clip(df["tips_prior"].to_numpy(), -5.0, 5.0)
    np.testing.assert_allclose(feats["tips_prior"].to_numpy(), expected, rtol=1e-6)


def serve(monkeypatch, csv_by_id: dict[str, str]) -> None:
    def fake_get(url, *, params=None, **kwargs):
        assert url == fred_markets.FRED_CSV_URL
        return FakeResponse(csv_by_id[params["id"]].encode())

    monkeypatch.setattr(fred_markets, "http_get", fake_get)


def fred_csv(name: str, dates: pd.DatetimeIndex, values) -> str:
    rows = ["observation_date," + name]
    rows += [f"{d.date()},{v}" for d, v in zip(dates, values, strict=True)]
    return "\n".join(rows) + "\n"


def test_loader_reads_the_four_fred_series(monkeypatch):
    days = pd.bdate_range("2003-01-02", "2010-06-30")
    cpi_idx = pd.date_range("2001-01-01", "2010-06-01", freq="MS")
    serve(monkeypatch, {
        "DGS10": fred_csv("DGS10", days, np.full(len(days), 4.0)),
        "DFII10": fred_csv("DFII10", days, np.full(len(days), 2.0)),
        "T10YIE": fred_csv("T10YIE", days, np.full(len(days), 2.0)),
        "CPIAUCNS": fred_csv("CPIAUCNS", cpi_idx, np.linspace(170.0, 220.0, len(cpi_idx))),
    })
    df = tips.load_tips_breakeven("10", start="2008-01", end="2008-12")
    validate_frame(df)
    assert df.attrs["symbol"] == "BE10"
    assert df.index.min() >= utc("2008-01-01")
    assert df.index.max() <= utc("2008-12-31")
    assert np.isfinite(df["tips_prior"]).all()


def test_unknown_tenor_is_rejected():
    with pytest.raises(ValueError):
        tips.load_tips_breakeven("7")


@pytest.mark.parametrize("name", ["tips_breakeven_1d", "tips_breakeven_1d_prior"])
def test_config_charges_each_unit_two_legs_at_one_bp_fee_and_half_spread(name):
    with open(ROOT / "configs" / "tips" / f"{name}.yaml") as f:
        costs = yaml.safe_load(f)["costs"]
    # The tape charges one fee and one half spread per unit of spread turnover. A unit trades
    # two bonds, each at 1 bp fee plus 1 bp half spread, so the per-unit charge is 4 bp.
    assert costs["fee_bps"] + costs["half_spread_bps"] == pytest.approx(4.0)
