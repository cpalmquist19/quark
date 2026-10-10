"""`quark` command-line entry point.

Available: version, gpu detect|list, components list|show, new, data prepare, train, generate. The rest of plan §13
(plan, build, notebook, compare, export, models) arrives in later phases.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

import quark
from quark.config import HardwareProfile

app = typer.Typer(help="Quark: plan, build and train small transformers.", no_args_is_help=True)
gpu_app = typer.Typer(help="Detect and list GPU hardware profiles.", no_args_is_help=True)
app.add_typer(gpu_app, name="gpu")
data_app = typer.Typer(help="Prepare datasets.", no_args_is_help=True)
app.add_typer(data_app, name="data")
components_app = typer.Typer(help="Browse the component registry.", no_args_is_help=True)
app.add_typer(components_app, name="components")

# Windows consoles may use a legacy code page (e.g. cp1252); never crash on an unprintable character.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure") and (_stream.encoding or "").lower() not in ("utf-8", "utf8"):
        _stream.reconfigure(errors="replace")

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


@components_app.command("list")
def components_list(slot: str | None = typer.Argument(None, help="Only this slot, e.g. attention.")) -> None:
    """List registered components by slot."""
    from quark.components import components, slots

    known = [s.name for s in slots()]
    if slot is not None and slot not in known:
        console.print(f"[red]unknown slot {slot!r} (known: {', '.join(known)})[/red]")
        raise typer.Exit(1)
    table = Table()
    table.add_column("component", no_wrap=True)
    for column in ("maturity", "summary"):
        table.add_column(column)
    for spec in slots():
        if slot is not None and spec.name != slot:
            continue
        for comp in components(spec.name):
            table.add_row(comp.key(), comp.maturity, comp.summary)
    console.print(table)


@components_app.command("show")
def components_show(key: str = typer.Argument(..., help="slot:name, e.g. position:rope.")) -> None:
    """Show one component: paper, rules, hardware needs and parameters."""
    from dataclasses import asdict

    from quark.components import get

    slot, _, name = key.partition(":")
    try:
        comp = get(slot, name)
    except KeyError as err:
        console.print(f"[red]{err.args[0]}[/red]")
        raise typer.Exit(1) from err
    table = Table(title=comp.key(), show_header=False)
    table.add_row("summary", comp.summary)
    table.add_row("paper", comp.paper or "-")
    table.add_row("maturity", comp.maturity)
    if implemented_by := getattr(comp, "implemented_by", ""):
        table.add_row("implemented by", implemented_by)
    table.add_row("requires", ", ".join(comp.requires) or "-")
    table.add_row("conflicts", ", ".join(comp.conflicts) or "-")
    hw = {k: v for k, v in asdict(comp.hw).items() if v}
    table.add_row("hardware", ", ".join(f"{k}={v}" for k, v in hw.items()) or "any")
    table.add_row("supports toggles", ", ".join(sorted(comp.supports)) or "-")
    for rule in comp.rules:
        table.add_row(f"rule ({rule.severity})", rule.reason)
    if comp.prior is not None:
        p = comp.prior
        lo, hi = p.validated_params_m
        table.add_row("prior", f"x{p.multiplier:g} compute ({p.confidence} confidence, {lo:g}-{hi:g}M params) {p.source}")
    fields = comp.Params.model_fields
    table.add_row("params", ", ".join(f"{k} = {f.default!r}" for k, f in fields.items()) or "-")
    console.print(table)


@app.command()
def new(
    name: str = typer.Argument(..., help="Model name; creates models/<name>/config.yaml."),
    preset: str = typer.Option("tinystories-small", help="Starting config: smoke | tinystories-small."),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing config.yaml."),
) -> None:
    """Create a model folder from a preset config."""
    from quark.config.presets import get_preset
    from quark.project import create_model

    try:
        directory = create_model(name, get_preset(preset), force=force)
    except (KeyError, FileExistsError) as err:
        console.print(f"[red]{err}[/red]")
        raise typer.Exit(1) from err
    console.print(f"Created {directory / 'config.yaml'} from preset {preset!r}. Edit it, then:")
    console.print(f"  quark data prepare --source <dataset> && quark train {directory}")


@data_app.command("prepare")
def data_prepare(
    source: str = typer.Option("roneneldan/TinyStories", help="Hugging Face dataset name, or a local file/folder."),
    out: Path | None = typer.Option(None, help="Output folder (default: data/<source>-<tokenizer>)."),
    tokenizer: str = typer.Option("train:8192", help="train:<vocab size> or pretrained:<name> (e.g. smollm2)."),
    split: str = typer.Option("train", help="Hugging Face split to read."),
    text_field: str = typer.Option("text", help="Field/column holding the text."),
    val_fraction: float = typer.Option(0.005, help="Fraction of documents held out for validation."),
    max_docs: int | None = typer.Option(None, help="Only use the first N documents (quick tests)."),
    tokenizer_sample_docs: int = typer.Option(200_000, help="Documents used to train a BPE tokenizer."),
) -> None:
    """Tokenize a dataset once into memory-mapped train/val files."""
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

    from quark.config import DataConfig, TokenizerConfig
    from quark.data import default_data_dir, prepare_dataset

    kind, _, value = tokenizer.partition(":")
    if kind == "train" and value.isdigit():
        tok = TokenizerConfig(type="train", size=int(value))
    elif kind == "pretrained" and value:
        tok = TokenizerConfig(type="pretrained", name=value, size=None)
    else:
        console.print("[red]--tokenizer must look like train:8192 or pretrained:smollm2[/red]")
        raise typer.Exit(1)
    data = DataConfig(source=source, split=split, text_field=text_field, tokenizer=tok, val_fraction=val_fraction)
    out = out or default_data_dir(data)

    with Progress(SpinnerColumn("line"), TextColumn("{task.description}"), TimeElapsedColumn(), console=console) as bar:
        task = bar.add_task("loading source and building tokenizer...")

        def progress(docs: int, tokens: int) -> None:
            bar.update(task, description=f"tokenized {docs:,} docs, {tokens / 1e6:,.1f}M tokens")

        meta = prepare_dataset(data, out, max_docs=max_docs, tokenizer_sample_docs=tokenizer_sample_docs, progress=progress)

    console.print(
        f"Wrote {out}: vocab {meta['vocab_size']:,}, train {meta['train_tokens'] / 1e6:,.1f}M tokens "
        f"({meta['train_docs']:,} docs), val {meta['val_tokens'] / 1e6:,.2f}M tokens ({meta['val_docs']:,} docs)"
    )


@app.command()
def train(
    model: str = typer.Argument(..., help="Model name or folder (with config.yaml)."),
    data: Path | None = typer.Option(None, help="Prepared data folder (default: the resumed run's, if resuming)."),
    resume: str | None = typer.Option(None, help="latest | best | path to a checkpoint."),
    allow_config_change: bool = typer.Option(False, "--allow-config-change", help="Resume even if the config changed."),
    max_steps: int | None = typer.Option(None, help="Stop after this many steps (for quick tests)."),
) -> None:
    """Train a model (new run, or resume an existing one)."""
    import json

    from quark.project import load_model_config, model_dir, new_run_dir, resolve_checkpoint
    from quark.train import ConfigMismatchError, Trainer

    directory = model_dir(model)
    cfg = load_model_config(directory)
    checkpoint = None
    if resume:
        try:
            checkpoint = resolve_checkpoint(directory, resume)
        except FileNotFoundError as err:
            console.print(f"[red]{err}[/red]")
            raise typer.Exit(1) from err
        run_dir = checkpoint.parent
        if data is None:
            data = Path(json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["data_dir"])
    else:
        if data is None:
            console.print("[red]pass --data <prepared data folder> (see `quark data prepare`)[/red]")
            raise typer.Exit(1)
        run_dir = new_run_dir(directory)

    try:
        trainer = Trainer(cfg, data, run_dir, resume_from=checkpoint, allow_config_change=allow_config_change, max_steps=max_steps)
    except ConfigMismatchError as err:
        console.print(f"[red]{err}[/red]")
        raise typer.Exit(1) from err
    result = trainer.fit()
    console.print(f"best val loss {result.best_val:.4f}; checkpoints in {result.run_dir}")


@app.command()
def generate(
    model: str = typer.Argument(..., help="Model name or folder."),
    ckpt: str = typer.Option("best", help="best | latest | path to a checkpoint."),
    prompt: str = typer.Option("Once upon a time", help="Text to continue."),
    max_new: int = typer.Option(200, help="Maximum new tokens."),
    temperature: float = typer.Option(0.8, help="0 = greedy."),
    top_p: float = typer.Option(0.9, help="Nucleus sampling threshold (1 = off)."),
    top_k: int = typer.Option(0, help="Top-k sampling (0 = off)."),
    samples: int = typer.Option(1, help="Number of samples."),
    seed: int | None = typer.Option(None, help="Random seed for reproducible samples."),
) -> None:
    """Generate text from a trained checkpoint."""
    from quark.infer import generate_text, load_for_inference
    from quark.project import model_dir, resolve_checkpoint

    try:
        path = resolve_checkpoint(model_dir(model), ckpt)
    except FileNotFoundError as err:
        console.print(f"[red]{err}[/red]")
        raise typer.Exit(1) from err
    net, tokenizer, _ = load_for_inference(path)
    console.print(f"[dim]{path}[/dim]")
    for i in range(samples):
        text = generate_text(
            net, tokenizer, prompt, max_new, temperature, top_k, top_p, None if seed is None else seed + i
        )
        if samples > 1:
            console.print(f"\n--- sample {i + 1} ---", style="bold")
        console.print(text, markup=False, highlight=False)


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
