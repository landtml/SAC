import numpy as np
import pandas as pd
import pytest

from softedge.eval.compare import COLUMNS, compare_runs


def _write(root, name, frame):
    (root / name).mkdir()
    frame.to_csv(root / name / "pooled_returns.csv")


def test_paired_comparison_on_common_bars(tmp_path):
    rng = np.random.default_rng(0)
    idx = pd.date_range("2022-01-01", periods=2000, freq="1h", tz="UTC", name="time")
    noise = rng.normal(0, 0.01, len(idx))
    base = pd.DataFrame({"sac_seed0": noise, "sac_banded": noise,
                         "oracle": np.abs(noise)}, index=idx)
    better = base.copy()
    better["sac_banded"] = noise + 0.002
    better["sac_seed0"] = noise + 0.001
    _write(tmp_path, "base", base)
    _write(tmp_path, "better", better.iloc[100:])  # fewer bars: compared on the overlap only
    table = compare_runs(tmp_path, "base", ["better"], n_boot=300)
    assert list(table.columns) == COLUMNS
    b, v = table.set_index("run").loc["base"], table.set_index("run").loc["better"]
    assert v["n_bars"] == 1900 and b["diff_vs_base"] == 0.0
    assert v["diff_vs_base"] > 0 and v["diff_lo"] > 0 and v["p_value"] < 0.05
    assert v["sharpe"] > v["seed_mean_sharpe"] > b["sharpe"]
    assert 0 < v["oracle_share"] < 1


def test_unfinished_run_raises(tmp_path):
    (tmp_path / "base").mkdir()
    with pytest.raises(FileNotFoundError):
        compare_runs(tmp_path, "base", ["other"])


def test_universe_pbo_pools_sac_columns_of_one_universe(tmp_path):
    from softedge.eval.compare import universe_pbo

    rng = np.random.default_rng(1)
    idx = pd.date_range("2022-01-01", periods=640, freq="1h", tz="UTC", name="time")
    for name, universe in (("a", "u"), ("b", "u"), ("c", "other")):
        frame = pd.DataFrame({"sac_seed0": rng.normal(0, 0.01, 640),
                              "sac_banded": rng.normal(0, 0.01, 640),
                              "buy_and_hold": rng.normal(0, 0.01, 640)}, index=idx)
        _write(tmp_path, name, frame)
        (tmp_path / name / "config.yaml").write_text(f"universe: {universe}\n")
        (tmp_path / name / "result.json").write_text("{}")
    out = universe_pbo(tmp_path, "u")
    assert out["n_variants"] == 4 and out["n_bars"] == 640 and 0.0 <= out["pbo"] <= 1.0
    with pytest.raises(ValueError):
        universe_pbo(tmp_path, "missing")
