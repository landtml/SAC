"""Command line: ``softedge download <universe>``, ``softedge run <config.yaml>``,
``softedge dsr``, ``softedge leaderboard`` and ``softedge compare <base> <runs>...``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="softedge")
    sub = parser.add_subparsers(dest="cmd", required=True)

    dl = sub.add_parser("download", help="fill the data cache for a universe")
    dl.add_argument("universe", choices=["crypto_perp", "crypto_spot", "crypto_basis", "fx", "macro",
                                         "polymarket", "kalshi"])
    dl.add_argument("--start", default="2020-01")
    dl.add_argument("--end", default="2026-09")
    dl.add_argument("--symbols", nargs="*")
    dl.add_argument("--max-markets", type=int, default=300)

    run = sub.add_parser("run", help="train, select on validation, evaluate on test")
    run.add_argument("config")
    run.add_argument("--out", default="runs")

    dsr = sub.add_parser("dsr", help="recompute deflated Sharpe ratios over all saved runs")
    dsr.add_argument("--runs", default="runs")
    dsr.add_argument("--out", default=None, help="file to write (default: RUNS/dsr.md)")

    lb = sub.add_parser("leaderboard", help="rank every saved run by deflated Sharpe ratio")
    lb.add_argument("--runs", default="runs")
    lb.add_argument("--top", type=int, default=None)

    cmp = sub.add_parser("compare", help="paired Sharpe tests of runs against a base run")
    cmp.add_argument("base")
    cmp.add_argument("names", nargs="+")
    cmp.add_argument("--runs", default="runs")
    cmp.add_argument("--strategy", default="sac_banded")

    sm = sub.add_parser("summary", help="SAC against the best baseline of every finished run")
    sm.add_argument("--runs", default="runs")
    sm.add_argument("--out", default=None, help="file to write (default: RUNS/summary.md)")

    pbo = sub.add_parser("pbo", help="probability of backtest overfitting across a universe")
    pbo.add_argument("universe")
    pbo.add_argument("--runs", default="runs")

    args = parser.parse_args(argv)
    if args.cmd == "summary":
        from softedge.eval.report import to_markdown
        from softedge.eval.summary import run_summary

        text = to_markdown(run_summary(args.runs))
        _write_report(args.out or Path(args.runs) / "summary.md", text)
        print(text)
        return 0
    if args.cmd == "pbo":
        from softedge.eval.compare import universe_pbo

        print(json.dumps(universe_pbo(args.runs, args.universe)))
        return 0
    if args.cmd == "compare":
        from softedge.eval.compare import compare_runs
        from softedge.eval.report import to_markdown

        print(to_markdown(compare_runs(args.runs, args.base, args.names, args.strategy)))
        return 0
    if args.cmd == "leaderboard":
        from softedge.eval.leaderboard import collect, leaderboard_markdown

        text = leaderboard_markdown(collect(args.runs), args.top)
        (__import__("pathlib").Path(args.runs) / "leaderboard.md").write_text(text + "\n")
        print(text)
        return 0
    if args.cmd == "dsr":
        from softedge.eval.report import to_markdown
        from softedge.experiment import recompute_dsr

        table = recompute_dsr(args.runs)
        cols = ["run", "strategy", "sharpe", "psr", "n_trials", "sr0", "dsr", "min_trl_years",
                "n_bars", "skew", "kurt"]
        text = to_markdown(table[cols])
        _write_report(args.out or Path(args.runs) / "dsr.md", text)
        print(text)
        return 0
    if args.cmd == "download":
        from softedge.pipeline import load_frames

        frames = load_frames(args.universe, args.start, args.end, symbols=args.symbols,
                             max_markets=args.max_markets)
        for fr in frames:
            print(f"{fr.attrs['symbol']}\t{fr.attrs['kind']}\t{len(fr)} bars\t"
                  f"{fr.index[0]} .. {fr.index[-1]}")
        return 0
    if args.cmd == "run":
        from softedge.experiment import load_config, run_experiment

        result = run_experiment(load_config(args.config), args.out)
        # Walk-forward results carry "summary"; CPCV results carry "aggregate".
        print(json.dumps(result.get("summary", result.get("aggregate")), indent=2))
        return 0
    return 2


def _write_report(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")


if __name__ == "__main__":
    sys.exit(main())
