import pytest
from pydantic import ValidationError

from quark.config import ModelConfig, QuarkConfig, TokenizerConfig


def test_yaml_round_trip(tmp_path):
    cfg = QuarkConfig(name="tiny", model=ModelConfig(d_model=256, n_heads=4, n_kv_heads=2))
    path = tmp_path / "config.yaml"
    cfg.to_yaml(path)
    loaded = QuarkConfig.from_yaml(path)
    assert loaded == cfg
    assert QuarkConfig.from_yaml(cfg.to_yaml()) == cfg  # from a YAML string too


def test_hash_is_stable_and_tracks_training_changes():
    a, b = QuarkConfig(), QuarkConfig()
    assert a.config_hash() == b.config_hash()
    assert len(a.config_hash()) == 12

    changed = QuarkConfig(train={"lr": 1e-3})
    assert changed.config_hash() != a.config_hash()


def test_hash_ignores_name_and_hardware():
    # Checkpoints are portable across GPUs, so these must not break resume.
    base = QuarkConfig().config_hash()
    assert QuarkConfig(name="other", hardware="colab-t4").config_hash() == base


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"d_model": 100, "n_heads": 3}, "divisible by n_heads"),
        ({"n_heads": 6, "n_kv_heads": 4}, "divisible by n_kv_heads"),
        ({"slots": {"position": "rope"}}, "missing slot choices"),
    ],
)
def test_model_shape_validation(kwargs, message):
    with pytest.raises(ValidationError, match=message):
        ModelConfig(**kwargs)


def test_unknown_keys_are_rejected():
    with pytest.raises(ValidationError):
        QuarkConfig.model_validate({"model": {"d_modle": 512}})


def test_tokenizer_defaults_follow_plan_rule():
    tok = TokenizerConfig()
    assert (tok.type, tok.size) == ("train", 8192)
    with pytest.raises(ValidationError, match="name is required"):
        TokenizerConfig(type="pretrained")


def test_new_fields_leave_old_hashes_alone():
    # Fields added after Phase 1 don't change the hash while unset, and do once set.
    base = QuarkConfig()
    assert base.config_hash() == QuarkConfig(data={"subset": None, "token_limit": None}).config_hash()
    assert QuarkConfig(data={"subset": "sample-10BT"}).config_hash() != base.config_hash()
    assert QuarkConfig(model={"params": {"position": {"base": 5e5}}}).config_hash() != base.config_hash()
