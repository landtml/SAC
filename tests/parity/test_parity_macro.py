import math
from pathlib import Path

import numpy as np
import pandas as pd
import parity_macro as pm
import pytest

UST2_PERCENT = [
    ("2005-11-25", 4.00), ("2005-11-28", 4.10), ("2005-11-29", 4.05), ("2005-11-30", 4.20),
    ("2005-12-01", 4.00), ("2005-12-02", 4.00), ("2005-12-05", 4.10), ("2005-12-06", 4.05),
    ("2005-12-07", 4.20), ("2005-12-08", 4.00), ("2005-12-09", 4.00),
]


def write_fred(root: Path, series_id: str, dated_values) -> None:
    root.mkdir(parents=True, exist_ok=True)
    lines = [f"observation_date,{series_id}"]
    lines += [f"{date},{float(value):.6f}" for date, value in dated_values]
    (root / f"{series_id}.csv").write_text("\n".join(lines) + "\n")


def write_root(root: Path, after_end: float) -> str:
    days = pd.bdate_range("2005-11-01", "2006-01-13")
    steps = np.arange(len(days))
    late = np.asarray(days > pm.END)
    values = {
        "NASDAQCOM": 100.0 * 1.001 ** steps,
        "NIKKEI225": 100.0 * 1.0005 ** steps,
        "DCOILWTICO": 50.0 + 0.1 * np.sin(steps),
        "DCOILBRENTEU": 55.0 + 0.1 * np.cos(steps),
        "DGS10": 4.5 + 0.02 * np.sin(steps),
        "DGS5": 4.2 + 0.02 * np.cos(steps),
        "DGS2": 4.0 + 0.03 * np.sin(steps / 2),
        "DTB3": 3.0 + 0.001 * steps,
    }
    for series_id, series in values.items():
        series = np.where(late, after_end, series)
        write_fred(root, series_id, zip(days.strftime("%Y-%m-%d"), series, strict=True))
    return str(root)


def test_treasury_week_follows_the_daily_formula(tmp_path):
    write_fred(tmp_path, "DGS2", UST2_PERCENT)
    write_fred(tmp_path, "DTB3", [(d, 3.0) for d, _ in UST2_PERCENT])
    panel = pm.weekly_panel("UST2", str(tmp_path))
    duration = pm.SERIES["UST2"][2]
    path = [pct / 100.0 for _, pct in UST2_PERCENT]
    expected = 0.0
    # Mon 12-05 to Fri 12-09, each against the day before
    for prev, cur in zip(path[5:10], path[6:11], strict=True):
        expected += math.log1p(prev / 252 - duration * (cur - prev))
    assert panel.loc["2005-12-09", "risky"] == pytest.approx(expected, rel=1e-12)


def test_rows_after_end_change_nothing(tmp_path):
    a = write_root(tmp_path / "a", after_end=1.0)
    b = write_root(tmp_path / "b", after_end=1e6)
    for name in pm.SERIES:
        pd.testing.assert_frame_equal(pm.weekly_panel(name, a), pm.weekly_panel(name, b))
    assert pm.weekly_panel("NASDAQ", a).index.max() == pm.END


def test_cash_accrues_over_seven_days_from_the_starting_friday(tmp_path):
    days = pd.bdate_range("2005-11-25", "2005-12-09")
    write_fred(tmp_path, "NASDAQCOM", [(d.strftime("%Y-%m-%d"), 100 + i)
                                       for i, d in enumerate(days)])
    write_fred(tmp_path, "DTB3", [(d.strftime("%Y-%m-%d"), 3.65 if d <= pd.Timestamp("2005-12-02")
                                   else 5.00) for d in days])
    panel = pm.weekly_panel("NASDAQ", str(tmp_path))
    accrual = math.log1p(0.0365 * 7 / 365)
    assert panel.loc["2005-12-09", "cash"] == pytest.approx(accrual, rel=1e-12)
    assert panel.loc["2005-12-02", "cash"] == pytest.approx(accrual, rel=1e-12)


def test_panel_starts_when_cash_first_exists(tmp_path):
    days = pd.bdate_range("2005-11-25", "2005-12-16")
    write_fred(tmp_path, "NASDAQCOM", [(d.strftime("%Y-%m-%d"), 100 + i)
                                       for i, d in enumerate(days)])
    write_fred(tmp_path, "DTB3", [(d.strftime("%Y-%m-%d"), 3.0)
                                  for d in days if d >= pd.Timestamp("2005-12-02")])
    panel = pm.weekly_panel("NASDAQ", str(tmp_path))
    assert panel.index[0] == pd.Timestamp("2005-12-09")
    assert panel[["risky", "cash"]].notna().all().all()


def test_stale_flag_marks_a_friday_whose_last_observation_is_over_four_days_old(tmp_path):
    days = pd.bdate_range("2005-11-25", "2005-12-02")
    prices = [(d.strftime("%Y-%m-%d"), 100 + i) for i, d in enumerate(days)]
    write_fred(tmp_path, "NASDAQCOM", prices + [("2005-12-16", 120.0)])
    write_fred(tmp_path, "DTB3", [(d.strftime("%Y-%m-%d"), 3.0)
                                  for d in pd.bdate_range("2005-11-25", "2005-12-16")])
    panel = pm.weekly_panel("NASDAQ", str(tmp_path))
    assert panel.index.tolist() == [pd.Timestamp(d) for d in ("2005-12-02", "2005-12-09",
                                                              "2005-12-16")]
    assert panel["stale"].tolist() == [False, True, False]
    assert panel["stale"].dtype == bool


def test_coverage_reports_every_series_with_dates_and_counts(tmp_path):
    cov = pm.coverage(write_root(tmp_path, after_end=1.0))
    assert cov["series"].tolist() == list(pm.SERIES)
    assert set(cov["last_week"]) == {"2005-12-30"}
    assert set(cov["first_week"]) == {"2005-11-11"}
    assert set(cov["weeks"]) == {8}
    assert set(cov["stale_weeks"]) == {0}


def test_costs_cover_every_series():
    assert set(pm.COSTS) == set(pm.SERIES)
    for name, (_, kind, _) in pm.SERIES.items():
        expected = 0.0002 if kind == "yield" else 0.0005
        assert pm.COSTS[name] == expected
