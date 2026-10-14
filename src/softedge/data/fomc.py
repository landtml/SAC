"""Scheduled FOMC announcement dates from federalreserve.gov, and the flag a prior is built from.

Calendars come from the historical year pages (2000 to 2020) and from the current calendar page
(2021 onwards, which also lists the next year's tentative dates). Only regularly scheduled
meetings are read. Notation votes and the conference calls of the historical pages are not
listed as meetings, so an unscheduled meeting never sets the flag.

A meeting is announced on its last day: the second day of a two-day meeting, or the only day
of a one-day meeting. The statement is released at 14:00 ET on that day. The calendar is the final
revised schedule from the historical and current pages, not the schedule as first published, so a
meeting that was rescheduled is dated the day it was held.

``fomc_next_flags`` marks row ``t`` with 1 when the bar after it (the session ``t + 1``, which
runs from the close of ``t`` to the close of ``t + 1``) holds the reaction to an announcement.
A US bar closes after 14:00 ET on the announcement day, so the reaction is in the bar dated
that day. A Tokyo bar closes about 02:00 ET, before the release, so the reaction is in the bar
dated the next day; ``EARLY_CLOSE_SYMBOLS`` lists the symbols that close early. The flag marks
the whole announcement session, a daily bar that contains the release. It marks no window before
the release day, so it does not measure a pre-announcement drift. The flag reads the calendar and
the bar dates only, never prices.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable
from datetime import date

import numpy as np
import pandas as pd

from softedge.data.cache import cache_dir, http_get

HISTORICAL_URL = "https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"
CURRENT_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
# The historical year pages end in 2020; the current calendar page starts with 2021.
LAST_HISTORICAL_YEAR = 2020
# The current page is refreshed once a day, since it adds the next year's dates over time.
CURRENT_MAX_AGE_S = 86400

# Symbols whose daily bar closes before the 14:00 ET release on an announcement day (Tokyo).
EARLY_CLOSE_SYMBOLS = frozenset({"NKY"})

MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December")
_ABBREV_NUM = {m[:3]: i + 1 for i, m in enumerate(MONTHS)}
# Full names first, so the alternation takes "March" whole and "Mar" only where it is alone.
_MONTH = "|".join([*MONTHS, *(m[:3] for m in MONTHS)])

# "February 1-2 Meeting - 2005", "January 31-February 1 Meeting - 2006" and the abbreviated
# "Jan/Feb 31-1 Meeting - 2017" (historical pages).
_HISTORICAL_MEETING = re.compile(
    rf"(?P<m1>{_MONTH})(?:/(?P<m1b>{_MONTH}))?\s+(?P<d1>\d{{1,2}})"
    rf"(?:\s*-\s*(?:(?P<m2>{_MONTH})\s+)?(?P<d2>\d{{1,2}}))?"
    r"\s+Meeting\s*-\s*(?P<year>\d{4})"
)
# Section headings of the current page, e.g. '<a id="42828">2026 FOMC Meetings</a>'.
_CURRENT_YEAR = re.compile(r"(\d{4}) FOMC Meetings")
# A row of the current page: month cell ("January" or "Jan/Feb") and date cell ("27-28", "31-1").
_CURRENT_ROW = re.compile(
    r'<div class="[^"]*fomc-meeting__month[^"]*"><strong>(?P<months>[A-Za-z/]+)</strong></div>\s*'
    r'<div class="[^"]*fomc-meeting__date[^"]*">(?P<dates>[^<]*)</div>'
)
_DAY_RANGE = re.compile(r"(?P<d1>\d{1,2})(?:-(?P<d2>\d{1,2}))?\*?")


def page_url(year: int) -> str:
    return HISTORICAL_URL.format(year=year) if year <= LAST_HISTORICAL_YEAR else CURRENT_URL


def parse_calendar(html: str) -> list[date]:
    """Announcement dates of the scheduled meetings on one calendar page, sorted."""
    if "fomc-meeting__month" in html:
        found = _parse_current(html)
    else:
        found = _parse_historical(html)
    return _last_days(sorted(set(found)))


def _last_days(days: list[date]) -> list[date]:
    """Drop each day that is followed by the next calendar day in the list.

    The historical pages list a two-day meeting once per day. The statement and the minutes
    carry the second day, so only that day is an announcement.
    """
    return [d for i, d in enumerate(days)
            if not (i + 1 < len(days) and days[i + 1].toordinal() - d.toordinal() == 1)]


def _parse_historical(html: str) -> list[date]:
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    out = []
    for m in _HISTORICAL_MEETING.finditer(text):
        year = int(m.group("year"))
        start = _ABBREV_NUM[m.group("m1")[:3]]
        if m.group("d2"):
            end_name = m.group("m2") or m.group("m1b") or m.group("m1")
            end = _ABBREV_NUM[end_name[:3]]
            day = int(m.group("d2"))
        else:
            end, day = start, int(m.group("d1"))
        out.append(_announcement(year, end, day, start))
    return out


def _parse_current(html: str) -> list[date]:
    headings = list(_CURRENT_YEAR.finditer(html))
    out = []
    for i, head in enumerate(headings):
        stop = headings[i + 1].start() if i + 1 < len(headings) else len(html)
        section = html[head.end():stop]
        year = int(head.group(1))
        for row in _CURRENT_ROW.finditer(section):
            day_range = _DAY_RANGE.fullmatch(row.group("dates").strip())
            if day_range is None:  # for example "22 (notation vote)"
                continue
            # The current page writes some months as "Jan/Feb" or "Apr/May", so match on the prefix.
            months = [_ABBREV_NUM[name[:3]] for name in row.group("months").split("/")]
            start, end = months[0], months[-1]
            day = int(day_range.group("d2") or day_range.group("d1"))
            out.append(_announcement(year, end, day, start))
    return out


def _announcement(year: int, end_month: int, day: int, start_month: int) -> date:
    # A meeting that runs into the next calendar year (December to January) is dated by its end.
    if end_month < start_month:
        year += 1
    return date(year, end_month, day)


def _fetch(url: str, name: str, max_age: float | None) -> str:
    path = cache_dir("fomc") / name
    if path.exists() and (max_age is None or time.time() - path.stat().st_mtime <= max_age):
        return path.read_text(encoding="utf-8")
    body = http_get(url).content.decode("utf-8", errors="replace")
    # Write under a temporary name so an interrupted download never leaves a short cache file.
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(path)
    return body


def announcement_dates(start_year: int, end_year: int) -> list[date]:
    """Scheduled announcement dates in years ``start_year`` to ``end_year``, both inclusive.

    One page per historical year, and one fetch of the current page when a year after 2020 is
    asked for. Pages are cached under ``SOFTEDGE_DATA/fomc``.
    """
    years = range(start_year, end_year + 1)
    dates: set[date] = set()
    for year in years:
        if year <= LAST_HISTORICAL_YEAR:
            html = _fetch(page_url(year), f"fomchistorical{year}.html", None)
            dates.update(d for d in parse_calendar(html) if d.year == year)
    if end_year > LAST_HISTORICAL_YEAR:
        html = _fetch(CURRENT_URL, "fomccalendars.html", CURRENT_MAX_AGE_S)
        dates.update(d for d in parse_calendar(html) if start_year <= d.year <= end_year)
    return sorted(dates)


def fomc_next_flags(index: pd.DatetimeIndex, announcements: Iterable[date], *,
                    closes_before_release: bool = False) -> np.ndarray:
    """Flag per bar: 1.0 where the next bar's session holds a scheduled announcement's reaction.

    Session ``t + 1`` covers the calendar days after the bar date of row ``t`` up to and including
    the bar date of row ``t + 1``. An announcement on day ``D`` falls in that session when the bar
    dated ``D`` closes after the 14:00 ET release (the default, a US close). With
    ``closes_before_release`` the bar dated ``D`` closes first, so the session must be dated after
    ``D``. The last row has no next session and is 0.
    """
    days = pd.DatetimeIndex(index)
    if days.tz is not None:
        days = days.tz_convert("UTC").tz_localize(None)
    days = days.normalize().to_numpy(dtype="datetime64[ns]")
    ann = np.sort(pd.DatetimeIndex(list(announcements)).normalize().to_numpy(dtype="datetime64[ns]"))
    flags = np.zeros(len(days), dtype=np.float64)
    if len(days) < 2 or len(ann) == 0:
        return flags
    prev, nxt = days[:-1], days[1:]
    # Count announcements in (prev, nxt] for a bar that closes after the release, and in
    # [prev, nxt) for one that closes before it.
    side = "left" if closes_before_release else "right"
    inside = np.searchsorted(ann, nxt, side=side) - np.searchsorted(ann, prev, side=side)
    flags[:-1] = (inside > 0).astype(np.float64)
    return flags


def attach_fomc(frame: pd.DataFrame, announcements: Iterable[date] | None = None) -> pd.DataFrame:
    """Copy of a daily bar frame with the column ``fomc_next`` (see ``fomc_next_flags``).

    ``announcements`` defaults to the scheduled calendar for the years the frame spans. The bar
    close is taken from ``frame.attrs["symbol"]`` and ``EARLY_CLOSE_SYMBOLS``.
    """
    if announcements is None:
        years = pd.DatetimeIndex(frame.index).year
        announcements = announcement_dates(int(years.min()), int(years.max()))
    early = frame.attrs.get("symbol") in EARLY_CLOSE_SYMBOLS
    out = frame.copy()
    out["fomc_next"] = fomc_next_flags(frame.index, announcements, closes_before_release=early)
    out.attrs = dict(frame.attrs)
    return out
