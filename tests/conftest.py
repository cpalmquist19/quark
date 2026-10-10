import random

import pytest

from quark.config import DataConfig, ModelConfig, QuarkConfig, TokenizerConfig, TrainConfig
from quark.data import prepare_dataset

WORDS = "the a cat dog bird sat ran jumped on under over mat log tree happy sad big small red blue".split()


def make_corpus(path, n_docs=1500, seed=0):
    rng = random.Random(seed)
    docs = []
    for _ in range(n_docs):
        sentences = [" ".join(rng.choice(WORDS) for _ in range(rng.randint(4, 9))).capitalize() + "."
                     for _ in range(rng.randint(2, 5))]
        docs.append(" ".join(sentences))
    path.write_text("<|endoftext|>".join(docs), encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    corpus = make_corpus(root / "corpus.txt")
    data = DataConfig(source=str(corpus), tokenizer=TokenizerConfig(type="train", size=300), val_fraction=0.1)
    prepare_dataset(data, root / "prepared")
    return root / "prepared"


def tiny_config(**train_overrides) -> QuarkConfig:
    train = dict(
        lr=3e-3,
        warmup_steps=5,
        max_tokens=40 * 4 * 32,  # 40 steps
        micro_batch=4,
        log_every_steps=5,
        eval_every_steps=10,
        eval_batches=2,
        checkpoint_every_steps=10,
        precision="fp32",
    )
    train.update(train_overrides)
    return QuarkConfig(
        name="tiny",
        model=ModelConfig(vocab_size=512, d_model=32, n_layers=2, n_heads=4, seq_len=32),
        train=TrainConfig(**train),
    )
