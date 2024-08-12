from pathlib import Path
from tokenizers import Tokenizer
from datasets import Dataset
from tokenizers.models import WordPiece as WordPieceModel
from tokenizers.trainers import WordPieceTrainer
from tokenizers.decoders import WordPiece as WordPieceDecoder
from tokenizers.pre_tokenizers import Whitespace
from enum import Enum
from config import get_config_and_parser


class SpecialTokens(Enum):
    UNK = "<unk>"
    BOS = "<s>"
    EOS = "</s>"
    PAD = "<pad>"


def get_tokenizer_path() -> Path:
    return Path("data") / "tokenizer.json"


def get_tokenizer() -> Tokenizer:
    cached_path = get_tokenizer_path()
    assert Path.exists(cached_path), "Tokenizer does not exist, train it first"
    return Tokenizer.from_file(str(cached_path))


def train_tokenizer(ds: Dataset, vocab_size: int):
    cached_path = get_tokenizer_path()
    if Path.exists(cached_path):
        raise ValueError(
            "Tokenizer already exists. If you still want to train a new one, delete the existing one first"
        )
    tok = Tokenizer(WordPieceModel(unk_token=SpecialTokens.UNK.value))
    tok.pre_tokenizer = Whitespace()
    trainer = WordPieceTrainer(
        special_tokens=[t.value for t in SpecialTokens],
        vocab_size=vocab_size,
    )
    langs = ds[0]["translation"].keys()
    assert len(langs) == 2, "expected two langs, got: " + str(langs)

    def iter_ds(ds: Dataset):
        for ex in ds:
            # one sentence in each lang each time!
            for lang in langs:
                yield ex["translation"][lang]

    tok.train_from_iterator(iter_ds(ds), trainer)
    tok.decoder = WordPieceDecoder()
    tok.save(str(cached_path))


if __name__ == "__main__":
    from dataset import get_raw_dataset

    (
        ds_config,
        model_config,
        training_config,
        parser,
    ) = get_config_and_parser()
    # We only want the 'train' split to train the tokenizer
    dataset = (
        get_raw_dataset(ds_config)["train"]
        if "train" in get_raw_dataset(ds_config).keys()
        else get_raw_dataset(ds_config)
    )
    # Train tokenizer for both langs
    train_tokenizer(dataset, model_config.vocab_size)
    print("Tokenizer trained successfully")
