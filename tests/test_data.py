import json

import numpy as np
import pytest

from quark.config import DataConfig, TokenizerConfig
from quark.data import EOT, BatchLoader, TokenData, iter_documents, prepare_dataset
from quark.data.prepare import default_data_dir, is_val


def test_prepared_files(data_dir):
    meta = json.loads((data_dir / "meta.json").read_text())
    data = TokenData(data_dir)
    assert meta["vocab_size"] <= 300
    assert len(data.train) == meta["train_tokens"] and len(data.val) == meta["val_tokens"]
    assert meta["train_docs"] + meta["val_docs"] == 1500
    assert 0.05 < meta["val_docs"] / 1500 < 0.15  # ~val_fraction 0.1
    # Every document ends with EOT, and the text round-trips through the tokenizer.
    tok = data.tokenizer()
    assert data.train[-1] == meta["eot_id"]
    first_doc = np.asarray(data.train[: list(data.train).index(meta["eot_id"])])
    assert tok.decode(first_doc.tolist()).startswith(next(iter_documents(meta["source"]))[:20])


def test_split_is_by_content_and_deterministic():
    assert is_val("same text", 0.3) == is_val("same text", 0.3)
    assert sum(is_val(f"doc {i}", 0.2) for i in range(5000)) == pytest.approx(1000, rel=0.15)


def test_local_sources(tmp_path):
    (tmp_path / "a.txt").write_text(f"one{EOT}two", encoding="utf-8")
    (tmp_path / "b.jsonl").write_text('{"text": "three"}\n{"text": "four"}\n', encoding="utf-8")
    assert sorted(iter_documents(str(tmp_path))) == ["four", "one", "three", "two"]
    assert list(iter_documents(str(tmp_path), max_docs=1)) == ["one"]


def test_empty_val_split_fails(tmp_path):
    (tmp_path / "c.txt").write_text("only one document", encoding="utf-8")
    data = DataConfig(source=str(tmp_path / "c.txt"), tokenizer=TokenizerConfig(size=260), val_fraction=0.001)
    with pytest.raises(ValueError, match="empty split"):
        prepare_dataset(data, tmp_path / "out")


def test_default_data_dir():
    assert default_data_dir(DataConfig()).as_posix() == "data/tinystories-bpe8192"


def test_loader_epochs_and_resume(data_dir):
    tokens = TokenData(data_dir).train
    loader = BatchLoader(tokens, seq_len=16, batch_size=8, seed=3)
    x, y = loader.next_batch()
    assert x.shape == y.shape == (8, 16)
    assert (x[:, 1:] == y[:, :-1]).all()  # targets are inputs shifted by one

    for _ in range(5):
        loader.next_batch()
    state = loader.state_dict()
    expected = [loader.next_batch()[0] for _ in range(3)]

    resumed = BatchLoader(tokens, seq_len=16, batch_size=8, seed=0)
    resumed.load_state_dict(state)
    assert all((a == b).all() for a, b in zip(expected, (resumed.next_batch()[0] for _ in range(3))))

    # Running past the end starts a new, differently shuffled epoch.
    steps_per_epoch = loader.n_windows // 8
    for _ in range(steps_per_epoch):
        loader.next_batch()
    assert loader.epoch >= 1


def test_fixed_batches_are_repeatable(data_dir):
    loader = BatchLoader(TokenData(data_dir).val, seq_len=16, batch_size=4, shuffle=False)
    first = [x for x, _ in loader.fixed_batches(3)]
    second = [x for x, _ in loader.fixed_batches(3)]
    assert len(first) == 3 and all((a == b).all() for a, b in zip(first, second))
