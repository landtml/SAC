import itertools
import math

import pandas as pd
import pytest

from sactrader.eval.cpcv import assemble_paths, cpcv_splits, group_bounds, n_paths

START, END = "2020-01-01", "2021-01-01"  # 366 days, 2020 is a leap year
T0 = pd.Timestamp(START, tz="UTC")
T1 = pd.Timestamp(END, tz="UTC")
DAY = pd.Timedelta(days=1)
GRID = [(6, 2), (5, 1), (8, 3)]


def _overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def _zones(split, purge, embargo):
    """Purge zones before and embargo zones after each test window, clipped to the range."""
    zones = []
    for a, b in split.test:
        zones.append((max(T0, a - purge), a))
        zones.append((b, min(T1, b + embargo)))
    return [z for z in zones if z[0] < z[1]]


def _tiles(windows, lo, hi):
    """True when the windows, once sorted, abut exactly from ``lo`` to ``hi``."""
    cursor = lo
    for a, b in sorted(windows):
        if a != cursor:
            return False
        cursor = b
    return cursor == hi


@pytest.mark.parametrize("n_groups, n_test", GRID)
def test_split_count_is_binomial_in_combinations_order(n_groups, n_test):
    splits = cpcv_splits(START, END, n_groups, n_test)
    assert len(splits) == math.comb(n_groups, n_test)
    assert [s.index for s in splits] == list(range(len(splits)))
    assert [s.test_groups for s in splits] == list(
        itertools.combinations(range(n_groups), n_test)
    )


def test_groups_tile_the_range_with_equal_durations():
    bounds = group_bounds(START, END, 6)
    assert len(bounds) == 6
    assert bounds[0][0] == T0 and bounds[-1][1] == T1
    assert all(a[1] == b[0] for a, b in zip(bounds, bounds[1:], strict=False))
    assert all(hi - lo == pd.Timedelta(days=61) for lo, hi in bounds)


def test_groups_of_a_span_not_divisible_by_n_groups_differ_by_at_most_one_ns():
    bounds = group_bounds("2020-01-01", "2020-01-02", 7)  # 86400 s over 7 groups
    assert bounds[0][0] == pd.Timestamp("2020-01-01", tz="UTC")
    assert bounds[-1][1] == pd.Timestamp("2020-01-02", tz="UTC")
    assert all(a[1] == b[0] for a, b in zip(bounds, bounds[1:], strict=False))
    durations = [hi - lo for lo, hi in bounds]
    assert max(durations) - min(durations) <= pd.Timedelta(1, unit="ns")


def test_purge_and_embargo_zones_are_removed_from_training():
    # Six one-day groups from day 0 to day 6. Holding out group 2 gives test [2D, 3D).
    # Purge 12h removes [1.5D, 2D). Embargo 6h removes [3D, 3.25D).
    base = pd.Timestamp("2020-01-01", tz="UTC")
    split = cpcv_splits("2020-01-01", "2020-01-07", 6, 1, purge="12h", embargo="6h")[2]
    assert split.test_groups == (2,)
    assert split.test == ((base + pd.Timedelta("48h"), base + pd.Timedelta("72h")),)
    assert split.train == (
        (base, base + pd.Timedelta("36h")),
        (base + pd.Timedelta("78h"), base + pd.Timedelta("144h")),
    )


def test_zones_outside_the_range_are_clipped_away():
    # Group 0 starts at the range start, so its purge zone is empty.
    # Group 5 ends at the range end, so its embargo zone is empty.
    base = pd.Timestamp("2020-01-01", tz="UTC")
    splits = cpcv_splits("2020-01-01", "2020-01-07", 6, 1, purge="12h", embargo="12h")
    assert splits[0].train == ((base + pd.Timedelta("36h"), base + pd.Timedelta("144h")),)
    assert splits[5].train == ((base, base + pd.Timedelta("108h")),)


def test_adjacent_test_groups_merge_into_one_window():
    bounds = group_bounds(START, END, 6)
    by_groups = {s.test_groups: s for s in cpcv_splits(START, END, 6, 2)}
    assert by_groups[(1, 2)].test == ((bounds[1][0], bounds[2][1]),)
    assert by_groups[(0, 2)].test == ((bounds[0][0], bounds[0][1]), (bounds[2][0], bounds[2][1]))
    bounds8 = group_bounds(START, END, 8)
    triple = {s.test_groups: s for s in cpcv_splits(START, END, 8, 3)}[(0, 1, 2)]
    assert triple.test == ((bounds8[0][0], bounds8[2][1]),)


@pytest.mark.parametrize("n_groups, n_test", GRID)
def test_train_avoids_test_purge_and_embargo_zones(n_groups, n_test):
    purge, embargo = pd.Timedelta("10D"), pd.Timedelta("5D")
    for s in cpcv_splits(START, END, n_groups, n_test, purge="10D", embargo="5D"):
        blocked = _zones(s, purge, embargo) + list(s.test)
        for win in s.train:
            assert T0 <= win[0] < win[1] <= T1
            assert not any(_overlaps(win, zone) for zone in blocked)
        for a, b in zip(s.train, s.train[1:], strict=False):
            assert a[1] < b[0]  # maximal: touching windows would have merged
        # Blocked windows do not overlap one another here, so their lengths add up.
        blocked_total = sum((b - a for a, b in blocked), pd.Timedelta(0))
        train_total = sum((b - a for a, b in s.train), pd.Timedelta(0))
        assert blocked_total + train_total == T1 - T0


@pytest.mark.parametrize("n_groups, n_test", GRID)
def test_without_purge_or_embargo_train_and_test_tile_the_range(n_groups, n_test):
    for s in cpcv_splits(START, END, n_groups, n_test):
        assert _tiles(list(s.train) + list(s.test), T0, T1)


@pytest.mark.parametrize("n_groups, n_test", [(6, 0), (6, 6), (6, 7), (1, 1)])
def test_bad_group_counts_raise(n_groups, n_test):
    with pytest.raises(ValueError):
        cpcv_splits(START, END, n_groups, n_test)


@pytest.mark.parametrize("purge, embargo", [("-1D", "0D"), ("0D", "-1h")])
def test_negative_purge_or_embargo_raises(purge, embargo):
    with pytest.raises(ValueError):
        cpcv_splits(START, END, 6, 2, purge=purge, embargo=embargo)


def test_empty_or_reversed_range_raises():
    with pytest.raises(ValueError):
        cpcv_splits(END, START, 6, 2)
    with pytest.raises(ValueError):
        group_bounds(START, START, 6)


def test_span_shorter_than_one_ns_per_group_raises():
    with pytest.raises(ValueError):
        group_bounds("2020-01-01T00:00:00", "2020-01-01T00:00:00.000000005", 6)


@pytest.mark.parametrize("n_groups, n_test", GRID)
def test_n_paths_equals_n_test_times_splits_over_n_groups(n_groups, n_test):
    total = n_test * math.comb(n_groups, n_test)
    assert total % n_groups == 0
    assert n_paths(n_groups, n_test) == total // n_groups


def test_six_groups_two_test_gives_five_paths():
    assert n_paths(6, 2) == 5


@pytest.mark.parametrize("n_groups, n_test", [(6, 0), (6, 6), (1, 1)])
def test_n_paths_rejects_bad_arguments(n_groups, n_test):
    with pytest.raises(ValueError):
        n_paths(n_groups, n_test)


@pytest.mark.parametrize("n_groups, n_test", GRID)
def test_path_count_matches_n_paths(n_groups, n_test):
    splits = cpcv_splits(START, END, n_groups, n_test)
    assert len(assemble_paths(splits, n_groups)) == n_paths(n_groups, n_test)


def test_paths_cover_each_group_once_and_use_each_split_test_slot_once():
    splits = cpcv_splits(START, END, 6, 2)
    paths = assemble_paths(splits, 6)
    assert len(paths) == 5
    by_index = {s.index: s for s in splits}
    slots = []
    for path in paths:
        assert len(path) == 6
        for g, split_index in enumerate(path):
            # Path entry g must come from a split that holds group g out.
            assert g in by_index[split_index].test_groups
            slots.append((split_index, g))
    # Every (split, test group) pair is used by exactly one path.
    expected = [(s.index, g) for s in splits for g in s.test_groups]
    assert sorted(slots) == sorted(expected)
    for s in splits:
        assert sum(split_index == s.index for path in paths for split_index in path) == 2


def test_assemble_paths_rejects_splits_missing_a_group():
    splits = cpcv_splits(START, END, 6, 2)
    with pytest.raises(ValueError):
        assemble_paths(splits[:-1], 6)
