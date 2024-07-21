from pathlib import Path
from tokenizers import Tokenizer
from datasets import Dataset
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from enum import Enum
from config import get_config_and_parser


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


if __name__ == "__main__":
    from dataset import get_raw_dataset

    (
        ds_config,
        model_config,
        training_config,
        eval_config,
        parser,
    ) = get_config_and_parser()
    # We only want the 'train' split to train the tokenizer
    dataset = (
        get_raw_dataset(ds_config)["train"]
        if "train" in get_raw_dataset(ds_config).keys()
        else get_raw_dataset(ds_config)
    )
    # Train tokenizer for source language
    train_tokenizer(dataset, ds_config.src_lang, model_config.src_vocab_size)
    # Train tokenizer for target language
    train_tokenizer(dataset, ds_config.tgt_lang, model_config.tgt_vocab_size)
    print("Tokenizers trained successfully")
