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
