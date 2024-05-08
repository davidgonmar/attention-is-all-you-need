from pathlib import Path
from tokenizers import Tokenizer
from datasets import Dataset
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from enum import Enum


class SpecialTokens(Enum):
    UNK = "<unk>"
    BOS = "<s>"
    EOS = "</s>"
    PAD = "<pad>"


def train_tokenizer(ds: Dataset, lang: str, vocab_size: int):
    cached_path = get_tokenizer_path(lang)
    if Path.exists(cached_path):
        raise ValueError(
            f"Tokenizer for {lang} already exists. If you still want to train a new one, delete the existing one first"
        )
    tok = Tokenizer(WordPiece(unk_token=SpecialTokens.UNK.value))
    tok.pre_tokenizer = Whitespace()
    trainer = WordPieceTrainer(
        special_tokens=[t.value for t in SpecialTokens],
        vocab_size=vocab_size,
    )

    def iter_ds(ds: Dataset):
        for ex in ds:
            yield ex["translation"][lang]

    tok.train_from_iterator(iter_ds(ds), trainer)
    # We don't use padding/truncation in the tokenizer, we'll do it manually
    tok.save(str(cached_path))


def get_tokenizer_path(lang: str) -> Path:
    return Path("data") / f"tokenizer_{lang}.json"


def get_tokenizer(lang: str):
    cached_path = get_tokenizer_path(lang)
    assert Path.exists(
        cached_path
    ), f"Tokenizer for {lang} does not exist, train it first"
    return Tokenizer.from_file(str(cached_path))
