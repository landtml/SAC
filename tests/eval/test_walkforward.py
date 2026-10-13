import pandas as pd
import pytest

from softedge.eval.walkforward import walk_forward_splits

START, END = "2020-01-01", "2021-01-01"  # 366 days, 2020 is a leap year


def _assert_well_formed(splits, embargo_days):
    gap = pd.Timedelta(days=embargo_days)
    for s in splits:
        for lo, hi in (s.train, s.val, s.test):
            assert lo < hi
        assert s.train[1] + gap == s.val[0]
        assert s.val[1] + gap == s.test[0]
        assert s.train[1] <= s.val[0] and s.val[1] <= s.test[0]
        assert s.test[1] <= pd.Timestamp(END, tz="UTC")


def test_rolling_splits_are_ordered_embargoed_and_non_overlapping():
    splits = walk_forward_splits(START, END, train="180D", val="30D", test="30D",
                                 embargo="5D")
    # The first split needs 250 days, and each later origin moves 30 days.
    # Origins 0, 30, 60, 90 fit. The next one ends after day 366.
    assert len(splits) == 4
    _assert_well_formed(splits, embargo_days=5)
    for a, b in zip(splits, splits[1:], strict=False):
        assert b.train[0] - a.train[0] == pd.Timedelta(days=30)
        assert b.test[0] - a.test[0] == pd.Timedelta(days=30)


def test_default_step_equals_test_window():
    splits = walk_forward_splits(START, END, train="100D", val="10D", test="20D")
    assert len(splits) > 2
    for a, b in zip(splits, splits[1:], strict=False):
        assert b.train[0] - a.train[0] == pd.Timedelta(days=20)


def test_anchored_keeps_train_start_and_grows_train_window():
    splits = walk_forward_splits(START, END, train="120D", val="20D", test="20D",
                                 embargo="2D", anchored=True)
    assert len(splits) >= 2
    starts = {s.train[0] for s in splits}
    assert starts == {pd.Timestamp(START, tz="UTC")}
    ends = [s.train[1] for s in splits]
    assert all(b > a for a, b in zip(ends, ends[1:], strict=False))
    _assert_well_formed(splits, embargo_days=2)


def test_last_test_window_ends_inside_the_range():
    splits = walk_forward_splits(START, END, train="100D", val="10D", test="50D")
    assert splits[-1].test[1] <= pd.Timestamp(END, tz="UTC")
    # One more step would push the test window past the end.
    assert splits[-1].test[1] + pd.Timedelta(days=50) > pd.Timestamp(END, tz="UTC")


def test_no_split_when_range_is_too_short():
    assert walk_forward_splits(START, "2020-02-01", train="100D", val="10D", test="10D") == []


@pytest.mark.parametrize("bad", ["0D", "-5D"])
def test_nonpositive_step_is_rejected(bad):
    with pytest.raises(ValueError):
        walk_forward_splits(START, END, train="100D", val="10D", test="10D", step=bad)
