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
