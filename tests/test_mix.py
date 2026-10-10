"""Data mixtures: domain filters, resumable collection, preparation, the mixture loader and training on a mix."""

import json
import random

import numpy as np
import pytest
from pydantic import ValidationError

from quark.config import DataConfig, ModelConfig, QuarkConfig, TrainConfig
from quark.data import MixData, MixtureLoader, load_data
from quark.data.domains import DocFilter
from quark.data.mix import MixRecipe, collect, collect_status, iter_collected, mix_dir, prepare_mix

FILLER = "the cat sat on the mat and the dog ran to the big red tree under a blue sky".split()
EDU = "teachers classroom curriculum lesson plans literacy educators".split()
KIDS = "preschool toddlers kindergarten phonics young children alphabet".split()


def doc(rng: random.Random, topic: list[str] | None, n_words: int = 120) -> str:
    words = [rng.choice(FILLER) for _ in range(n_words)]
    if topic:
        for i in range(0, n_words, 8):  # dense enough to pass the strict threshold
            words[i] = rng.choice(topic)
    return " ".join(words)


@pytest.fixture(scope="module")
def stream_file(tmp_path_factory):
    rng = random.Random(0)
    rows = []
    for i in range(3000):
        topic = EDU if i % 5 == 0 else KIDS if i % 7 == 0 else None
        rows.append({"text": doc(rng, topic) + f" doc{i}", "url": f"https://site{i % 4}.example.org/p{i}", "kind": "a" if i % 2 else "b"})
    path = tmp_path_factory.mktemp("src") / "stream.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def recipe(stream_file, **sources) -> MixRecipe:
    return MixRecipe.model_validate({
        "name": "t", "tokenizer": {"type": "train", "size": 300}, "val_fraction": 0.1,
        "sources": sources or {
            "general": {"source": str(stream_file), "tokens": 40_000},
            "education": {"source": str(stream_file), "tokens": 15_000, "filter": {"domain": "education"}},
            "kids": {"source": str(stream_file), "tokens": 10_000, "filter": {"domain": "early_childhood"}},
        },
    })


@pytest.fixture(scope="module")
def mix_root(stream_file, tmp_path_factory):
    root = tmp_path_factory.mktemp("mix")
    r = recipe(stream_file)
    collect(r, root, log=lambda _: None, flush_docs=200)
    prepare_mix(r, root)
    return root


# --- filters and recipes ------------------------------------------------------------------------


def test_doc_filter_rules():
    rng = random.Random(1)
    about = {"text": doc(rng, EDU), "url": "https://ok.org/x"}
    assert DocFilter("education").accepts(about)
    assert not DocFilter("education").accepts({"text": doc(rng, None) + " teachers", "url": "https://ok.org"})  # passing mention
    assert not DocFilter("education", exclude_sites=("ok.org",)).accepts({**about, "url": "https://www.sub.ok.org/y"})
    assert not DocFilter("education", exclude_patterns=(r"\bcurriculum\b",)).accepts(about)
    assert DocFilter(columns={"kind": ["a"]}).accepts({"text": "x", "kind": "a"})
    assert not DocFilter(columns={"kind": ["a"]}).accepts({"text": "x", "kind": "b"})


@pytest.mark.parametrize(
    "source, message",
    [
        ({"source": "x", "tokens": 10, "filter": {"domain": "astrology"}}, "unknown domain"),
        ({"source": "x", "tokens": 10, "filter": {"exclude_patterns": ["(unclosed"]}}, "missing \\)"),
        ({"source": "x", "tokens": 0}, "greater than 0"),
    ],
)
def test_recipe_validation(source, message):
    with pytest.raises((ValidationError, ValueError), match=message):
        MixRecipe.model_validate({"name": "m", "sources": {"s": source}})


def test_shipped_recipe_is_valid():
    r = MixRecipe.from_yaml("mixes/chatbot-v1.yaml")
    assert {"general", "neurodivergence", "early_childhood", "education", "children_stories"} == set(r.sources)
    nd = r.sources["neurodivergence"].filter.build()
    assert not nd.accepts({"text": "x", "url": "https://www.autismspeaks.org/a"})


# --- collect and prepare ------------------------------------------------------------------------


def test_collect_fills_each_source_in_one_pass(stream_file, mix_root):
    r = recipe(stream_file)
    status = collect_status(r, mix_root)
    assert all(s["done"] for s in status.values())
    edu = list(iter_collected(r, "education", mix_root))
    assert edu and all(sum(w in t.split() for w in EDU) >= 3 for t in edu)  # only on-topic documents
    states = list((mix_dir(r, mix_root) / "raw").glob("*.state.json"))
    assert len(states) == 1  # three sources, one shared stream, one scan


def test_collect_resumes_without_duplicates(stream_file, tmp_path):
    r = recipe(stream_file, general={"source": str(stream_file), "tokens": 10**9})  # more than the file holds
    collect(r, tmp_path, log=lambda _: None, flush_docs=100, max_docs=700)
    raw = mix_dir(r, tmp_path) / "raw"
    first = len(list(iter_collected(r, "general", tmp_path)))
    assert first == 700
    # A shard written after the last saved state (a crash mid-flush) is discarded on resume.
    (raw / "general" / "part-999999999999.jsonl").write_text('{"text": "orphan"}\n', encoding="utf-8")
    collect(r, tmp_path, log=lambda _: None, flush_docs=100)
    texts = list(iter_collected(r, "general", tmp_path))
    assert len(texts) == 3000 == len(set(texts)) and "orphan" not in texts
    assert collect_status(r, tmp_path)["general"]["done"]  # stream exhausted


def test_prepare_needs_complete_collection(stream_file, tmp_path):
    with pytest.raises(RuntimeError, match="not fully collected"):
        prepare_mix(recipe(stream_file), tmp_path)


def test_prepared_mix_layout(stream_file, mix_root):
    data = load_data(mix_dir(recipe(stream_file), mix_root))
    assert isinstance(data, MixData) and set(data.sources) == {"general", "education", "kids"}
    assert data.meta["kind"] == "mix" and data.vocab_size <= 300
    for s in data.sources.values():
        assert len(s.train) > 0 and len(s.val) > 0
    assert DataConfig(mix="t", tokenizer={"type": "train", "size": 300}).model_dump()["mix"] == "t"


# --- loader -------------------------------------------------------------------------------------


def test_weights_resolution(stream_file, mix_root):
    data = load_data(mix_dir(recipe(stream_file), mix_root))
    natural = data.resolve_weights({})
    assert natural["general"] > natural["education"] and sum(natural.values()) == pytest.approx(1)
    assert data.resolve_weights({"general": 3, "kids": 1}) == {"general": 0.75, "kids": 0.25}
    with pytest.raises(ValueError, match="unknown sources"):
        data.resolve_weights({"nope": 1})


def test_mixture_loader_follows_weights_and_resumes(stream_file, mix_root):
    data = load_data(mix_dir(recipe(stream_file), mix_root))
    weights = {"general": 0.5, "education": 0.3, "kids": 0.2}
    loader = MixtureLoader(data, weights, seq_len=16, batch_size=8, seed=3)
    counts = {n: 0 for n in weights}
    for _ in range(300):
        before = {n: s.epoch_progress for n, s in loader.streams.items()}
        loader.next_batch()
        for n, s in loader.streams.items():
            counts[n] += round((s.epoch_progress - before[n]) * s.n_windows)
    total = sum(counts.values())
    for n, w in weights.items():
        assert counts[n] / total == pytest.approx(w, abs=0.04)
    assert loader.epoch_progress == max(loader.source_epochs.values()) > 0

    a = MixtureLoader(data, weights, 16, 8, seed=3)
    for _ in range(5):
        a.next_batch()
    state = a.state_dict()
    b = MixtureLoader(data, weights, 16, 8, seed=3)
    b.load_state_dict(state)
    for _ in range(3):
        xa, _ = a.next_batch()
        xb, _ = b.next_batch()
        assert np.array_equal(xa.numpy(), xb.numpy())


# --- training -----------------------------------------------------------------------------------


def mix_config(**data) -> QuarkConfig:
    return QuarkConfig(
        name="mix",
        model=ModelConfig(vocab_size=320, d_model=32, n_layers=2, n_heads=4, seq_len=32),
        train=TrainConfig(lr=3e-3, warmup_steps=5, max_tokens=20 * 4 * 32, micro_batch=4, log_every_steps=5,
                          eval_every_steps=10, eval_batches=2, checkpoint_every_steps=10, precision="fp32"),
        data=DataConfig(mix="t", tokenizer={"type": "train", "size": 300}, **data),
    )


def test_trainer_on_a_mix_reports_per_source_validation(stream_file, mix_root, tmp_path):
    from quark.train import Trainer

    data_dir = mix_dir(recipe(stream_file), mix_root)
    cfg = mix_config(weights={"general": 0.6, "education": 0.2, "kids": 0.2}, eval_weights={"general": 1, "kids": 1})
    trainer = Trainer(cfg, data_dir, tmp_path / "run", device="cpu", max_steps=10, log=lambda _: None)
    trainer.fit()
    evals = [json.loads(line) for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    last = [r for r in evals if r["type"] == "eval"][-1]
    assert {"val_loss/general", "val_loss/education", "val_loss/kids", "epochs/general"} <= set(last)
    # The headline is the eval-weighted mean: here general and kids, equally.
    assert last["val_loss"] == pytest.approx((last["val_loss/general"] + last["val_loss/kids"]) / 2)

    resumed = Trainer(cfg, data_dir, tmp_path / "run", device="cpu", resume_from=tmp_path / "run" / "latest.pt",
                      log=lambda _: None)
    assert resumed.step == 10 and resumed.train_loader.draws == trainer.train_loader.draws
    assert resumed.fit().stop_reason == "completed"


def test_trainer_warns_about_repetition_and_mismatches(stream_file, mix_root, data_dir, tmp_path):
    from quark.train import Trainer

    mix = mix_dir(recipe(stream_file), mix_root)
    logged = []
    cfg = mix_config(weights={"kids": 1.0})
    cfg = cfg.model_copy(update={"train": cfg.train.model_copy(update={"max_tokens": 5000 * 4 * 32})})
    Trainer(cfg, mix, tmp_path / "rep", device="cpu", log=logged.append)
    assert any("source 'kids' would be repeated" in line for line in logged)

    with pytest.raises(ValueError, match="is a mix but data.mix is None"):
        Trainer(mix_config().model_copy(update={"data": DataConfig()}), mix, tmp_path / "a", device="cpu",
                log=lambda _: None)
    with pytest.raises(ValueError, match="is a single source"):
        Trainer(mix_config(), data_dir, tmp_path / "b", device="cpu", log=lambda _: None)
    with pytest.raises(ValidationError, match="only apply to a mix"):
        DataConfig(weights={"a": 1})


def test_compare_reports_per_source_deltas(stream_file, mix_root, tmp_path):
    from quark.compare import CompareSpec, run_compare

    base = tmp_path / "base.yaml"
    mix_config().to_yaml(base)
    spec = CompareSpec(name="m", base=str(base), seeds=[1, 2],
                       arms={"natural": {}, "more-kids": {"data": {"weights": {"general": 0.5, "kids": 0.5}}}})
    summary = run_compare(spec, mix_dir(recipe(stream_file), mix_root), tmp_path / "exp", device="cpu",
                          log=lambda _: None)
    kids = summary["arms"]["more-kids"]["sources"]["kids"]
    assert kids["delta"] is not None and kids["delta_se"] is not None
    report = (tmp_path / "exp" / "report.md").read_text(encoding="utf-8")
    assert "Δ val loss by source" in report and "| arm | education | general | kids |" in report
