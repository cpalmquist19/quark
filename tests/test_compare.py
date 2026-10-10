import json

import pytest
from pydantic import ValidationError

from quark.compare import CompareSpec, arm_config, run_compare, summarize, validate, write_report
from quark.components import ComponentError

from .conftest import tiny_config


@pytest.fixture
def base_yaml(tmp_path):
    path = tmp_path / "base.yaml"
    tiny_config(max_tokens=20 * 4 * 32).to_yaml(path)  # 20 steps per run
    return str(path)


def spec(base, **updates) -> CompareSpec:
    fields = dict(
        name="t",
        base=base,
        seeds=[1, 2],
        arms={"baseline": {}, "relu2": {"model": {"slots": {"mlp": "relu2"}}}},
    )
    fields.update(updates)
    return CompareSpec(**fields)


def quiet(_):
    pass


def test_arm_config_merges_overrides_and_fixes_budget(base_yaml):
    s = spec(base_yaml, max_tokens=4096, arms={"a": {}, "b": {"model": {"slots": {"mlp": "relu2"}}, "train": {"lr": 1e-3}}})
    cfg = arm_config(s, "b", seed=7)
    assert cfg.model.slots["mlp"] == "relu2" and cfg.model.slots["position"] == "rope"  # merged, not replaced
    assert cfg.train.lr == 1e-3 and cfg.train.seed == 7 and cfg.train.max_tokens == 4096
    assert not cfg.train.early_stopping.enabled and cfg.train.keep_last == 0
    assert s.ref == "a"
    with_data = spec(base_yaml, data={"source": "some/dataset", "tokenizer": {"size": 300}})
    assert arm_config(with_data, "baseline", 1).data.source == "some/dataset"
    assert arm_config(with_data, "baseline", 1).data.tokenizer.size == 300
    assert arm_config(s, "a", 1).config_hash() != arm_config(s, "a", 2).config_hash()  # seeds differ


@pytest.mark.parametrize(
    "updates, message",
    [
        ({"arms": {"bad name": {}}}, "names may only use"),
        ({"reference": "nope"}, "not one of the arms"),
        ({"seeds": [1, 1]}, "distinct"),
        ({"arms": {}}, "at least one arm"),
    ],
)
def test_spec_validation(base_yaml, updates, message):
    with pytest.raises(ValidationError, match=message):
        spec(base_yaml, **updates)


def test_invalid_arms_fail_before_training(base_yaml):
    s = spec(base_yaml, arms={"baseline": {}, "typo": {"model": {"slots": {"mlp": "swigloo"}}}})
    with pytest.raises(ComponentError, match=r"typo / mlp:swigloo: unknown mlp 'swigloo'"):
        validate(s)


def test_run_compare_end_to_end(base_yaml, data_dir, tmp_path):
    out = tmp_path / "exp"
    s = spec(base_yaml)
    summary = run_compare(s, data_dir, out, device="cpu", log=quiet)

    assert summary["runs_done"] == summary["runs_total"] == 4
    relu2 = summary["arms"]["relu2"]
    assert relu2["paired"] == 2 and relu2["delta"] is not None and relu2["delta_se"] is not None
    assert relu2["verdict"] in ("better", "worse", "within noise")
    assert summary["arms"]["baseline"]["verdict"] == "reference"
    for arm in s.arms:
        for seed in s.seeds:
            run = out / "runs" / arm / f"seed{seed}"
            result = json.loads((run / "result.json").read_text())
            assert result["stop_reason"] == "completed" and result["final_val"] is not None
            assert not list(run.glob("*.pt"))  # checkpoints removed once a run is done
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "| relu2 |" in report and "Runs finished: 4 / 4" in report and "2.56K tokens per run" in report
    assert (out / "curves.png").exists() and (out / "spec.yaml").exists()

    # Re-running skips finished runs entirely.
    sizes = {p: p.stat().st_size for p in out.rglob("metrics.jsonl")}
    run_compare(s, data_dir, out, device="cpu", log=quiet)
    assert {p: p.stat().st_size for p in out.rglob("metrics.jsonl")} == sizes

    # Changing a finished arm is refused instead of silently mixing configs.
    changed = spec(base_yaml, arms={"baseline": {}, "relu2": {"model": {"slots": {"mlp": "relu2"}}, "train": {"lr": 1e-4}}})
    with pytest.raises(ValueError, match="arm 'relu2' changed"):
        run_compare(changed, data_dir, out, device="cpu", log=quiet)


def test_interrupted_runs_resume(base_yaml, data_dir, tmp_path):
    from quark.train import Trainer

    out = tmp_path / "exp"
    s = spec(base_yaml, seeds=[1], arms={"baseline": {}})
    run_dir = out / "runs" / "baseline" / "seed1"
    Trainer(arm_config(s, "baseline", 1), data_dir, run_dir, device="cpu", max_steps=10, log=quiet).fit()
    assert (run_dir / "latest.pt").exists() and not (run_dir / "result.json").exists()

    logged = []
    summary = run_compare(s, data_dir, out, device="cpu", log=logged.append)
    assert any("(resuming)" in line for line in logged)
    assert any("resumed from" in line and "step 10" in line for line in logged)
    assert summary["runs_done"] == 1
    assert json.loads((run_dir / "result.json").read_text())["steps"] == 20


def test_report_with_no_finished_runs(base_yaml, tmp_path):
    s = spec(base_yaml)
    summary = write_report(s, tmp_path)
    assert summary["runs_done"] == 0
    assert summarize(s, tmp_path)["arms"]["relu2"]["verdict"] == "pending"
    assert "Runs finished: 0 / 4" in (tmp_path / "report.md").read_text(encoding="utf-8")
