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
