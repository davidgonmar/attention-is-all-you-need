from datasets import load_dataset, Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from torch.utils.data import random_split
from typing import Tuple
import torch
from config import DatasetConfig, ModelConfig


def get_dataset(
    config: DatasetConfig,
    model_config: ModelConfig,
) -> Tuple["TranslationDataset", "TranslationDataset"]:
    """
    Returns a tuple of training and test datasets

    Args:
        config: loaded configuration

    Returns:
        Tuple of training and test datasets
    """
    try:
        datasetdict = load_dataset(
            config.ds_name, f"{config.src_lang}-{config.tgt_lang}", cache_dir="data"
        )
    except:  # noqa
        datasetdict = load_dataset(
            config.ds_name, f"{config.tgt_lang}-{config.src_lang}", cache_dir="data"
        )

    # if it has a train and test split, we use those
    if "train" in datasetdict.keys() and "test" in datasetdict.keys():
        train_ds = datasetdict["train"]
        test_ds = datasetdict["test"]
        return TranslationDataset(
            train_ds,
            config.src_lang,
            config.tgt_lang,
            config.seq_len,
            model_config.src_vocab_size,
            model_config.tgt_vocab_size,
        ), TranslationDataset(
            test_ds,
            config.src_lang,
            config.tgt_lang,
            config.seq_len,
            model_config.src_vocab_size,
            model_config.tgt_vocab_size,
        )
    # if it doesn't have a test split, we split it ourselves
    else:
        assert (
            "train" in datasetdict.keys()
        ), "Dataset must have a train split, got datasetdict: {datasetdict}"
        dataset = datasetdict["train"]
        train_size = int(config.split * len(dataset))
        test_size = len(dataset) - train_size
        train_ds, test_ds = random_split(
            dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )  # so the dataset remains the same
        return TranslationDataset(
            train_ds,
            config.src_lang,
            config.tgt_lang,
            config.seq_len,
            model_config.src_vocab_size,
            model_config.tgt_vocab_size,
        ), TranslationDataset(
            test_ds,
            config.src_lang,
            config.tgt_lang,
            config.seq_len,
            model_config.src_vocab_size,
            model_config.tgt_vocab_size,
        )


def _get_sentences_iter(ds: Dataset, lang: str):
    for item in ds:
        yield item["translation"][lang]


def get_tokenizer(ds: Dataset, lang: str, vocab_size):
    cached_path = Path("data") / f"ds_{lang}.json"
    if not Path.exists(cached_path):
        tok = Tokenizer(WordPiece(unk_token="<unk>"))
        tok.pre_tokenizer = Whitespace()
        trainer = WordPieceTrainer(
            special_tokens=["<unk>", "<s>", "</s>", "<pad>"], vocab_size=vocab_size
        )
        tok.train_from_iterator(_get_sentences_iter(ds, lang), trainer=trainer)
        tok.save(str(cached_path))
        return tok
    else:
        return Tokenizer.from_file(str(cached_path))


class TranslationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        ds: Dataset,
        src_lang: str,
        tgt_lang: str,
        seq_len: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
    ):
        self.raw_ds = ds
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.src_tok = get_tokenizer(self.raw_ds, src_lang, src_vocab_size)
        self.tgt_tok = get_tokenizer(self.raw_ds, tgt_lang, tgt_vocab_size)

        self.src_tok.enable_truncation(max_length=seq_len)
        self.tgt_tok.enable_truncation(max_length=seq_len)
        self.src_tok.enable_padding(
            length=seq_len, pad_id=self.src_tok.token_to_id("<pad>")
        )
        self.tgt_tok.enable_padding(
            length=seq_len, pad_id=self.tgt_tok.token_to_id("<pad>")
        )
        self.sos = self._load_sp_tok("<s>")
        self.eos = self._load_sp_tok("</s>")
        self.pad = self._load_sp_tok("<pad>")
        self.unk = self._load_sp_tok("<unk>")

        self.seq_len = seq_len

    def _load_sp_tok(self, tok_str: str):
        return torch.tensor(self.src_tok.token_to_id(tok_str), dtype=torch.int64)

    def __len__(self):
        return len(self.raw_ds)

    def __getitem__(self, key):
        both = self.raw_ds[key]
        src = "<s> " + both["translation"][self.src_lang] + " </s>"
        tgt_shifted = "<s> " + both["translation"][self.tgt_lang]
        tgt_labels = both["translation"][self.tgt_lang] + " </s>"
        src = self.src_tok.encode(src).ids
        tgt_shifted = self.tgt_tok.encode(tgt_shifted).ids
        tgt_labels = self.tgt_tok.encode(tgt_labels).ids

        return {
            "src": torch.tensor(src, dtype=torch.int64),
            "tgt_shifted": torch.tensor(tgt_shifted, dtype=torch.int64),
            "tgt_labels": torch.tensor(tgt_labels, dtype=torch.int64),
        }
