from datetime import date

import numpy as np
import pandas as pd
import pytest

from softedge.data import fomc, fred_markets
from softedge.data.schema import make_frame, validate_frame
from softedge.features.causal import compute_features
from softedge.pipeline import load_frames, to_tapes

HISTORICAL_HTML = """<html><body>
<a href="/x">February 1-2 Meeting - 2005</a>
<a href="/y">March 22 Meeting - 2005</a>
<a href="/z">January 31-February 1 Meeting - 2006</a>
<a href="/c">March 15 Conference Call - 2020</a>
<a href="/n">Notation Vote - 2020</a>
</body></html>"""

CURRENT_HTML = """<html><body>
<h4><a id="1">2025 FOMC Meetings</a></h4>
<div class="row fomc-meeting">
  <div class="fomc-meeting__month col-xs-5"><strong>January</strong></div>
  <div class="fomc-meeting__date col-xs-4">28-29</div>
</div>
<div class="row fomc-meeting">
  <div class="fomc-meeting--shaded fomc-meeting__month col-xs-5"><strong>August</strong></div>
  <div class="fomc-meeting__date col-xs-4">22 (notation vote)</div>
</div>
<h4><a id="2">2023 FOMC Meetings</a></h4>
<div class="row fomc-meeting">
  <div class="fomc-meeting__month col-xs-5"><strong>Jan/Feb</strong></div>
  <div class="fomc-meeting__date col-xs-4">31-1</div>
</div>
<div class="row fomc-meeting">
  <div class="fomc-meeting--shaded fomc-meeting__month col-xs-5"><strong>March</strong></div>
  <div class="fomc-meeting__date col-xs-4">21-22*</div>
</div>
</body></html>"""


class FakeResponse:
    def __init__(self, content: bytes):
        self.content = content


@pytest.fixture(autouse=True)
def isolated(request, tmp_path, monkeypatch):
    # Network tests keep the caller's cache directory.
    if request.node.get_closest_marker("network"):
        return
    monkeypatch.setenv("SOFTEDGE_DATA", str(tmp_path))
    monkeypatch.setattr(fred_markets, "REQUEST_INTERVAL_S", 0.0)


def _spot_frame(days: pd.DatetimeIndex, close: np.ndarray, symbol: str = "T"):
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    return make_frame(days, open_, high, low, close, symbol=symbol, kind="spot", bar="1D")


def _random_closes(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))


def test_historical_page_reads_scheduled_meetings_on_their_last_day():
    assert fomc.parse_calendar(HISTORICAL_HTML) == [
        date(2005, 2, 2), date(2005, 3, 22), date(2006, 2, 1)]


def test_historical_headings_with_abbreviated_and_slashed_months():
    html = ('<h5 class="x">Jan/Feb 31-1 Meeting - 2017</h5>'
            '<h5 class="x">Jul/Aug 31-1 Meeting - 2018</h5>'
            '<h5 class="x">Dec 15-16 Meeting - 2003</h5>')
    assert fomc.parse_calendar(html) == [date(2003, 12, 16), date(2017, 2, 1), date(2018, 8, 1)]


def test_a_two_day_meeting_listed_day_by_day_is_announced_on_its_second_day():
    # The 2003 page lists the September meeting as an entry for each day; the statement and
    # minutes are dated the 16th, and the 15th carries only the agenda.
    html = ("<h5>September 15 Meeting - 2003</h5><h5>September 16 Meeting - 2003</h5>"
            "<h5>October 28 Meeting - 2003</h5>")
    assert fomc.parse_calendar(html) == [date(2003, 9, 16), date(2003, 10, 28)]


def test_current_page_reads_each_year_and_skips_notation_votes():
    assert fomc.parse_calendar(CURRENT_HTML) == [
        date(2023, 2, 1), date(2023, 3, 22), date(2025, 1, 29)]


def test_flag_marks_the_bar_before_the_announcement_session():
    index = pd.bdate_range("2024-01-29", "2024-02-02", tz="UTC")  # Mon to Fri
    flags = fomc.fomc_next_flags(index, [date(2024, 1, 31)])  # Wednesday
    np.testing.assert_array_equal(flags, [0, 1, 0, 0, 0])


def test_announcement_on_a_day_without_a_bar_is_taken_by_the_next_bar():
    # Monday 15 January 2024 is a market holiday: its announcement falls in Tuesday's session.
    index = pd.DatetimeIndex(["2024-01-12", "2024-01-16", "2024-01-17"], tz="UTC")
    np.testing.assert_array_equal(fomc.fomc_next_flags(index, [date(2024, 1, 15)]), [1, 0, 0])


def test_flag_depends_on_the_calendar_and_not_on_prices():
    days = pd.bdate_range("2020-01-01", periods=300, tz="UTC")
    close = _random_closes(300, seed=3)
    k = 120
    alt = close.copy()
    alt[k + 1:] *= 1.5  # a different price path after row k
    announcements = [date(2020, 1, 29)] + [d.date() for d in days[k + 1:k + 2]]
    before = compute_features(fomc.attach_fomc(_spot_frame(days, close), announcements))
    after = compute_features(fomc.attach_fomc(_spot_frame(days, alt), announcements))
    np.testing.assert_array_equal(before["fomc_next"].to_numpy(), after["fomc_next"].to_numpy())
    assert before["fomc_next"].iloc[k] == 1.0
    # Guards against a vacuous pass: the price features after row k must actually change.
    assert not np.allclose(before["ret_h1"].iloc[k + 1:], after["ret_h1"].iloc[k + 1:])


def test_feature_is_a_column_only_when_the_frame_carries_it():
    # Past the 256-bar warm-up of the default features, so the tape keeps the late announcement.
    days = pd.bdate_range("2020-01-01", periods=400, tz="UTC")
    frame = _spot_frame(days, _random_closes(400, seed=1))
    assert "fomc_next" not in compute_features(frame).columns
    tagged = fomc.attach_fomc(frame, [days[300].date()])
    feats = compute_features(tagged)
    assert "fomc_next" in feats.columns and feats["fomc_next"].dtype == np.float32
    tape = to_tapes([tagged])[0]
    assert "fomc_next" in tape.feature_names
    col = tape.feature_names.index("fomc_next")
    assert tape.features[:, col].sum() == 1.0


def _fake_fred(monkeypatch, n_days: int = 120) -> None:
    days = pd.bdate_range("2023-12-01", periods=n_days)
    bodies = {}
    for i, series in enumerate(("SP500", "NASDAQCOM", "NIKKEI225")):
        vals = _random_closes(n_days, seed=i)
        bodies[series] = "observation_date," + series + "\n" + "\n".join(
            f"{d.date()},{v:.4f}" for d, v in zip(days, vals, strict=True)) + "\n"

    def fake_get(url, *, params=None, **kwargs):
        return FakeResponse(bodies[params["id"]].encode())

    monkeypatch.setattr(fred_markets, "http_get", fake_get)


def test_macro_fomc_universe_is_the_equity_indices_with_the_calendar_flag(monkeypatch):
    _fake_fred(monkeypatch)
    monkeypatch.setattr(fomc, "announcement_dates",
                        lambda start, end: [date(2024, 1, 31), date(2024, 3, 20)])
    frames = load_frames("macro_fomc", "2024-01", "2024-03")
    assert [f.attrs["symbol"] for f in frames] == ["SPX", "NDX", "NKY"]
    for frame in frames:
        validate_frame(frame)
        assert frame["fomc_next"].sum() == 2
    plain = load_frames("macro", "2024-01", "2024-03", symbols=["SPX"])
    assert "fomc_next" not in plain[0].columns


def test_scheduled_calendar_pages_are_fetched_once_and_then_cached(monkeypatch):
    calls: list[str] = []

    def fake_get(url, *, params=None, **kwargs):
        calls.append(url)
        return FakeResponse(HISTORICAL_HTML.encode() if "2005" in url else b"<html></html>")

    monkeypatch.setattr(fomc, "http_get", fake_get)
    assert fomc.announcement_dates(2005, 2005) == [date(2005, 2, 2), date(2005, 3, 22)]
    assert fomc.announcement_dates(2005, 2005) == [date(2005, 2, 2), date(2005, 3, 22)]
    assert calls == [fomc.HISTORICAL_URL.format(year=2005)]


def test_configs_build_the_flag_only_into_the_prior_tapes(monkeypatch):
    from pathlib import Path

    from softedge.env.vec_env import TapeSet
    from softedge.experiment import build_all_tapes, build_train_config, load_config

    root = Path(__file__).resolve().parents[2]
    _fake_fred(monkeypatch, n_days=600)
    late = pd.bdate_range("2023-12-01", periods=600)[300].date()  # past the warm-up
    monkeypatch.setattr(fomc, "announcement_dates", lambda start, end: [late])
    prior_cfg = load_config(root / "configs/macro/fomc_1d_prior.yaml")
    base_cfg = load_config(root / "configs/macro/fomc_1d_vt.yaml")
    prior = build_all_tapes(prior_cfg)
    base = build_all_tapes(base_cfg)
    assert [tp.symbol for tp in prior] == ["SPX", "NDX", "NKY"]
    assert all("fomc_next" in tp.feature_names for tp in prior)
    assert not any("fomc_next" in tp.feature_names for tp in base)
    env = build_train_config(prior_cfg, 0).env
    ts = TapeSet(prior, env.window, action_map=env.action_map)
    # One announcement, and each of the three indices has one bar before it.
    assert set(np.unique(ts.prior)) <= {0.0, 1.0} and ts.prior.sum() == 3.0


@pytest.mark.network
def test_live_calendar_lists_known_meetings_and_no_conference_calls():
    dates = fomc.announcement_dates(2003, 2026)
    assert date(2003, 9, 16) in dates and date(2003, 9, 15) not in dates
    assert date(2020, 1, 29) in dates and date(2020, 3, 15) not in dates
    assert date(2026, 1, 28) in dates and date(2026, 9, 16) in dates
