from typer.testing import CliRunner

from quark.cli import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "quark 0.1.0" in result.output
    assert "torch" in result.output


def test_gpu_list_shows_presets():
    result = runner.invoke(app, ["gpu", "list"])
    assert result.exit_code == 0
    assert "colab-t4" in result.output
    assert "rtx3070ti-laptop" in result.output


def test_components_list_and_show():
    result = runner.invoke(app, ["components", "list"])
    assert result.exit_code == 0
    for key in ("position:rope", "attention:mha", "block:standard", "optimizer:adamw", "schedule:cosine"):
        assert key in result.output
    result = runner.invoke(app, ["components", "show", "position:rope"])
    assert result.exit_code == 0
    assert "arXiv:2104.09864" in result.output and "base = 10000.0" in result.output
    result = runner.invoke(app, ["components", "show", "position:ropee"])
    assert result.exit_code == 1 and "did you mean 'rope'" in result.output
