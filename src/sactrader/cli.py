"""Command line: ``sactrader download <universe>`` and ``sactrader run <config.yaml>``."""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sactrader")
    sub = parser.add_subparsers(dest="cmd", required=True)

    dl = sub.add_parser("download", help="fill the data cache for a universe")
    dl.add_argument("universe", choices=["crypto_perp", "crypto_spot", "fx", "polymarket"])
    dl.add_argument("--start", default="2020-01")
    dl.add_argument("--end", default="2026-09")
    dl.add_argument("--symbols", nargs="*")
    dl.add_argument("--max-markets", type=int, default=300)

    run = sub.add_parser("run", help="train, select on validation, evaluate on test")
    run.add_argument("config")
    run.add_argument("--out", default="runs")

    args = parser.parse_args(argv)
    if args.cmd == "download":
        from sactrader.pipeline import load_frames

        frames = load_frames(args.universe, args.start, args.end, symbols=args.symbols,
                             max_markets=args.max_markets)
        for fr in frames:
            print(f"{fr.attrs['symbol']}\t{fr.attrs['kind']}\t{len(fr)} bars\t"
                  f"{fr.index[0]} .. {fr.index[-1]}")
        return 0
    if args.cmd == "run":
        from sactrader.experiment import load_config, run_experiment

        result = run_experiment(load_config(args.config), args.out)
        print(json.dumps(result["summary"], indent=2))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
