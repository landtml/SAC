"""Add ``bars_per_year`` to the result.json of runs saved before the key was written.

For each ``<runs>/<name>/result.json`` without the key: rebuild the run's tapes from its
``config.yaml`` (no training), compute the bar rate the way the run did, copy result.json to
result.json.bak and add the key. Nothing else in the file changes.

Runs are never downloaded from. Network calls are refused while the script runs; a run that
reaches for the network, for example because its data is not cached, is skipped and listed.
``--allow-network RUN ...`` lifts the block for the named runs only, for runs whose data is not
cached and may be fetched. The rate comes from the bar timestamps alone (estimate_bars_per_year
in softedge.tape), so refetched prices do not change it.
Cached data is read whatever its age: the cache is what the run read, so a stale file is not
refetched.
Runs whose result.json was written less than ``--min-age`` seconds ago are skipped, since a
run that is still being written may be incomplete.
"""

from __future__ import annotations

import argparse
import contextlib
import http.client
import json
import os
import shutil
import socket
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from softedge.cpcv_experiment import _utc
from softedge.data import fomc, fred_markets, fx, rates, vol_indices
from softedge.experiment import build_all_tapes, load_config, make_splits
from softedge.pipeline import split_tapes, window_tapes


class NetworkBlocked(RuntimeError):
    pass


_blocked_calls = 0


def _refuse(*args, **kwargs):
    global _blocked_calls
    _blocked_calls += 1
    raise NetworkBlocked("network access is blocked while backfilling")


@contextlib.contextmanager
def no_network() -> Iterator[None]:
    """Refuse every network call made inside the block, then restore the originals."""
    try:
        import requests
    except ImportError:
        requests = None
    patched = [
        (urllib.request, "urlopen"), (socket, "create_connection"),
        (socket.socket, "connect"), (socket.socket, "connect_ex"),
        (http.client.HTTPConnection, "connect"),
    ]
    if requests is not None:
        patched += [(requests, "get"), (requests.Session, "request")]
    saved = [(owner, name, getattr(owner, name)) for owner, name in patched]
    try:
        for owner, name in patched:
            setattr(owner, name, _refuse)
        yield
    finally:
        for owner, name, original in saved:
            setattr(owner, name, original)


# Cache-age limits of the loaders that keep downloads on disk: a cached file older than its limit
# is refetched. Every age check in src/softedge/data reads one of these.
CACHE_AGE_LIMITS = [
    (rates, "MAX_CACHE_AGE_S"), (fx, "MAX_CACHE_AGE_S"), (vol_indices, "MAX_CACHE_AGE_S"),
    (fred_markets, "MAX_CACHE_AGE_S"), (fomc, "CURRENT_MAX_AGE_S"),
]


@contextlib.contextmanager
def fresh_caches() -> Iterator[None]:
    """Treat every cached file as fresh, whatever its age, and restore the limits afterwards."""
    saved = [(owner, name, getattr(owner, name)) for owner, name in CACHE_AGE_LIMITS]
    try:
        for owner, name, _ in saved:
            setattr(owner, name, float("inf"))
        yield
    finally:
        for owner, name, original in saved:
            setattr(owner, name, original)


def run_bars_per_year(cfg: dict) -> float:
    """Bar rate a run annualises with: the median over the tapes it tests.

    Mirrors ``run_experiment`` (a fold counts when its train, validation and test parts are all
    non-empty) and ``run_cpcv`` (the tapes inside ``[start, end)``).
    """
    tapes = build_all_tapes(cfg)
    if "cpcv" in cfg:
        c = cfg["cpcv"]
        test = window_tapes(tapes, [(_utc(c["start"]), _utc(c["end"]))])
    else:
        test = []
        for split in make_splits(cfg):
            parts = split_tapes(tapes, split)
            if parts["train"] and parts["val"] and parts["test"]:
                test.extend(parts["test"])
    if not test:
        raise ValueError("the run tests no tapes")
    return float(np.median([tp.bars_per_year for tp in test]))


def backfill_run(path: Path, min_age: float) -> tuple[str, float | None, str]:
    """Process one result.json; returns (run, bars per year, status)."""
    run = path.parent.name
    raw = path.read_text()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = None  # possibly a run still being written; the age check below decides
    if result is not None and "bars_per_year" in result:
        return run, float(result["bars_per_year"]), "already set"
    if time.time() - path.stat().st_mtime < min_age:
        return run, None, f"skipped: result.json younger than {min_age:.0f} s"
    if result is None:
        return run, None, "skipped: result.json is not valid JSON"
    if json.dumps(result, indent=2, default=str) != raw:
        return run, None, "skipped: result.json not in the writer's format"
    if not (path.parent / "config.yaml").exists():
        return run, None, "skipped: no config.yaml"
    cfg = load_config(path.parent / "config.yaml")
    before = _blocked_calls
    try:
        bpy = run_bars_per_year(cfg)
    except NetworkBlocked:
        return run, None, "skipped: needs network (data not cached)"
    except Exception as exc:  # report every failing run, then carry on with the rest
        return run, None, f"skipped: {type(exc).__name__}: {str(exc).splitlines()[0]}"
    if _blocked_calls != before:
        # A loader may swallow the refusal and build from a partial cache, so the rate is not trusted.
        return run, None, "skipped: needs network (data not cached)"

    shutil.copy2(path, path.with_name("result.json.bak"))
    result["bars_per_year"] = bpy
    tmp = path.with_name("result.json.tmp")
    tmp.write_text(json.dumps(result, indent=2, default=str))
    os.replace(tmp, path)
    return run, bpy, "filled"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", default="runs", help="runs directory")
    ap.add_argument("--min-age", type=float, default=600.0,
                    help="skip runs whose result.json is younger than this many seconds")
    ap.add_argument("--allow-network", nargs="+", default=[], metavar="RUN",
                    help="runs allowed to fetch their data; every other run is blocked")
    args = ap.parse_args(argv)

    rows = []
    with fresh_caches():
        for path in sorted(Path(args.runs).glob("*/result.json")):
            allowed = path.parent.name in args.allow_network
            guard = contextlib.nullcontext() if allowed else no_network()
            with guard:
                rows.append(backfill_run(path, args.min_age))

    width = max((len(r[0]) for r in rows), default=3)
    print(f"{'run':<{width}}  {'bars_per_year':>14}  status")
    for run, bpy, status in rows:
        shown = "-" if bpy is None else f"{bpy:.6f}"
        print(f"{run:<{width}}  {shown:>14}  {status}")
    counts: dict[str, int] = {}
    for _, _, status in rows:
        key = status.split(":")[0] if status.startswith("skipped") else status
        counts[key] = counts.get(key, 0) + 1
    print("\n" + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
