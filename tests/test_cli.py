import json

import softedge.experiment as experiment
from softedge.cli import main


def test_run_prints_cpcv_aggregate(monkeypatch, capsys, tmp_path):
    # CPCV results carry an "aggregate" block instead of a walk-forward "summary".
    monkeypatch.setattr(experiment, "load_config", lambda path: {})
    monkeypatch.setattr(experiment, "run_experiment",
                        lambda cfg, out: {"mode": "cpcv", "aggregate": {"sac": 1.0}})
    assert main(["run", str(tmp_path / "c.yaml"), "--out", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out) == {"sac": 1.0}


def test_dsr_and_summary_write_to_out_path(monkeypatch, tmp_path):
    import pandas as pd

    import softedge.eval.report as report
    import softedge.eval.summary as summary
    import softedge.experiment as experiment

    cols = ["run", "strategy", "sharpe", "psr", "n_trials", "sr0", "dsr", "min_trl_years",
            "n_bars", "skew", "kurt"]
    monkeypatch.setattr(experiment, "recompute_dsr", lambda root: pd.DataFrame(columns=cols))
    monkeypatch.setattr(summary, "run_summary", lambda root: pd.DataFrame())
    monkeypatch.setattr(report, "to_markdown", lambda df, floatfmt=".3f": "TABLE")
    runs = tmp_path / "runs"
    runs.mkdir()
    target = tmp_path / "analysis"

    assert main(["dsr", "--runs", str(runs), "--out", str(target / "dsr.md")]) == 0
    assert main(["summary", "--runs", str(runs), "--out", str(target / "summary.md")]) == 0
    assert (target / "dsr.md").read_text() == "TABLE\n"
    assert (target / "summary.md").read_text() == "TABLE\n"
    assert not (runs / "dsr.md").exists() and not (runs / "summary.md").exists()

    # Without --out the files still land in the runs root.
    assert main(["dsr", "--runs", str(runs)]) == 0
    assert main(["summary", "--runs", str(runs)]) == 0
    assert (runs / "dsr.md").read_text() == "TABLE\n"
    assert (runs / "summary.md").read_text() == "TABLE\n"


def _pipeline_universes() -> set[str]:
    """Universe names that ``load_frames`` compares against, read from its source."""
    import ast
    from pathlib import Path

    import softedge.pipeline as pipeline

    tree = ast.parse(Path(pipeline.__file__).read_text())
    (func,) = [n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "load_frames"]
    names: set[str] = set()
    for node in ast.walk(func):
        if (isinstance(node, ast.Compare) and isinstance(node.left, ast.Name)
                and node.left.id == "universe"):
            for comp in node.comparators:
                values = comp.elts if isinstance(comp, ast.Tuple) else [comp]
                names.update(v.value for v in values if isinstance(v, ast.Constant))
    return names


def test_download_choices_match_every_universe_load_frames_supports():
    from softedge.cli import DOWNLOAD_UNIVERSES

    assert set(DOWNLOAD_UNIVERSES) == _pipeline_universes()
    assert {"pegs", "delivery_basis", "hl_basis", "perp_spread", "fx_em"} <= set(DOWNLOAD_UNIVERSES)


def test_download_accepts_each_universe_and_rejects_others(monkeypatch, capsys):
    import pytest

    import softedge.pipeline as pipeline
    from softedge.cli import DOWNLOAD_UNIVERSES

    seen = []
    monkeypatch.setattr(pipeline, "load_frames", lambda universe, *a, **k: seen.append(universe) or [])
    for name in DOWNLOAD_UNIVERSES:
        assert main(["download", name]) == 0
    assert seen == list(DOWNLOAD_UNIVERSES)
    with pytest.raises(SystemExit):
        main(["download", "not_a_universe"])
