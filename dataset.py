from datasets import load_dataset, Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from torch.utils.data import random_split
from typing import Tuple
import torch
from config import DatasetConfig


def get_dataset(
    config: DatasetConfig,
) -> Tuple["TranslationDataset", "TranslationDataset"]:
    """
    Returns a tuple of training and validation datasets

    Args:
        config: loaded configuration

    Returns:
        Tuple of training and validation datasets
    """
    dataset = load_dataset(
        config.ds_name, f"{config.src_lang}-{config.tgt_lang}", split="train"
    )  # We'll manually split the dataset

    train_size = int(config.split * len(dataset))
    valid_size = len(dataset) - train_size
    train_ds, valid_ds = random_split(dataset, [train_size, valid_size])
    return TranslationDataset(
        train_ds, config.src_lang, config.tgt_lang, config.seq_len
    ), TranslationDataset(valid_ds, config.src_lang, config.tgt_lang, config.seq_len)


def _get_sentences_iter(ds: Dataset, lang: str):
    for item in ds:
        yield item["translation"][lang]


def _get_tokenizer(ds: Dataset, lang: str) -> Tokenizer:
    cached_path = Path("data") / f"ds_{lang}.json"
    if not Path.exists(cached_path):
        tok = Tokenizer(WordLevel(unk_token="<unk>"))
        tok.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["<unk>", "<s>", "</s>", "<pad>"])
        tok.train_from_iterator(_get_sentences_iter(ds, lang), trainer=trainer)
        tok.save(str(cached_path))
        return tok
    else:
        return Tokenizer.from_file(str(cached_path))


class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, ds: Dataset, src_lang: str, tgt_lang: str, seq_len: int = 128):
        self.raw_ds = ds
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.src_tok = _get_tokenizer(self.raw_ds, src_lang)
        self.tgt_tok = _get_tokenizer(self.raw_ds, tgt_lang)

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
