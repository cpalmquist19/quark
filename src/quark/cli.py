"""`quark` command-line entry point.

Phase 0 provides `version` and `gpu detect|list`. The remaining commands in plan §13 (plan, build,
data, notebook, train, generate, compare, export, models) are added in later phases.
"""

from __future__ import annotations

import platform

import typer
from rich.console import Console
from rich.table import Table

import quark
from quark.config import HardwareProfile

app = typer.Typer(help="Quark: plan, build and train small transformers.", no_args_is_help=True)
gpu_app = typer.Typer(help="Detect and list GPU hardware profiles.", no_args_is_help=True)
app.add_typer(gpu_app, name="gpu")

console = Console()


@app.command()
def version() -> None:
    """Show Quark, Python and PyTorch versions."""
    import torch

    cuda = torch.version.cuda or "none (CPU-only build)"
    console.print(f"quark {quark.__version__}")
    console.print(f"python {platform.python_version()}")
    console.print(f"torch {torch.__version__} (CUDA {cuda})")


@gpu_app.command("detect")
def gpu_detect(
    device: int = typer.Option(0, help="CUDA device index."),
    save: bool = typer.Option(False, "--save", help="Write the profile to ./hardware/<name>.yaml."),
) -> None:
    """Build a hardware profile from the current GPU."""
    from quark.hardware import NoGpuError, detect_profile, save_profile

    try:
        profile = detect_profile(device)
    except NoGpuError as err:
        console.print(f"[red]cuda: no[/red]  {err}")
        raise typer.Exit(1) from err

    console.print(_profile_table(profile))
    if save:
        path = save_profile(profile)
        console.print(f"Saved to {path}")


@gpu_app.command("list")
def gpu_list() -> None:
    """List known hardware profiles (built-in presets plus ./hardware/)."""
    from quark.hardware import list_profiles

    table = Table()
    table.add_column("name", no_wrap=True)  # profile names are what users type, so never truncate them
    for column in ("gpu", "VRAM (usable)", "cc", "precision", "SDPA kernels", "source"):
        table.add_column(column)
    for profile, path in list_profiles():
        source = "preset" if "presets" in path.parts else str(path)
        table.add_row(
            profile.name,
            profile.gpu_name,
            f"{profile.vram_gb:g} GB ({profile.usable_vram_gb:g})",
            profile.compute_capability,
            profile.precision,
            ", ".join(profile.sdpa_kernels),
            source,
        )
    console.print(table)


def _profile_table(profile: HardwareProfile) -> Table:
    def yes(flag: bool) -> str:
        return "[green]yes[/green]" if flag else "[yellow]no[/yellow]"

    table = Table(title=f"Hardware profile: {profile.name}", show_header=False)
    table.add_row("cuda", yes(True))
    table.add_row("gpu", profile.gpu_name)
    table.add_row("VRAM", f"{profile.vram_gb:g} GB total, {profile.usable_vram_gb:g} GB usable")
    table.add_row("compute capability", profile.compute_capability)
    table.add_row("bf16 (native)", yes(profile.bf16))
    table.add_row("tf32", yes(profile.tf32))
    table.add_row("training precision", f"{profile.precision}" + ("" if profile.bf16 else " + GradScaler"))
    table.add_row("SDPA kernels", ", ".join(profile.sdpa_kernels) or "none")
    table.add_row("flash-attn package", yes(profile.flash_attn_pkg))
    table.add_row("triton", yes(profile.triton))
    table.add_row("storage", profile.storage)
    return table


if __name__ == "__main__":
    app()
