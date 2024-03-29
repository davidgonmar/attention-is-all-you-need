from datasets import load_dataset, Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from torch.utils.data import random_split
from typing import Tuple
import torch
from dataclasses import dataclass


@dataclass
class Config:
    ds_name: str = "opus_books"
    src_lang: str = "en"
    tgt_lang: str = "es"
    split: int = 0.9
    seq_len: int = 400
    model_path: str = "transformer.pth"
    train: bool = True


def get_dataset(config: Config) -> Tuple[Dataset, Dataset]:
    dataset = load_dataset(
        config.ds_name, f"{config.src_lang}-{config.tgt_lang}", split="train"
    )  # only train is available, we will split it later

    # filter dataset from long sentences
    dataset = dataset.filter(
        lambda x: len(x["translation"][config.src_lang].split()) < config.seq_len
        and len(x["translation"][config.tgt_lang].split()) < config.seq_len
    )
    train_size = int(config.split * len(dataset))
    valid_size = len(dataset) - train_size
    train_ds, valid_ds = random_split(dataset, [train_size, valid_size])
    # print examples of train_ds
    return TranslationDataset(
        train_ds, config.src_lang, config.tgt_lang, config.seq_len
    ), TranslationDataset(valid_ds, config.src_lang, config.tgt_lang, config.seq_len)


def get_sentences_iter(ds: Dataset, lang: str):
    for item in ds:
        yield item["translation"][lang]


def get_tokenizer(ds: Dataset, lang: str):
    cached_path = Path("data/tok" + lang)
    if not Path.exists(cached_path):
        tok = Tokenizer(WordLevel(unk_token="<unk>"))
        tok.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["<unk>", "<s>", "</s>", "<pad>"])
        tok.train_from_iterator(get_sentences_iter(ds, lang), trainer=trainer)
        tok.save(str(cached_path))
        return tok
    else:
        return Tokenizer.from_file(str(cached_path))


class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, ds: Dataset, src_lang: str, tgt_lang: str, seq_len: int = 128):
        self.raw_ds = ds
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.src_tok = get_tokenizer(self.raw_ds, src_lang)
        self.tgt_tok = get_tokenizer(self.raw_ds, tgt_lang)

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
        src_seq = both["translation"][self.src_lang]
        tgt_seq = both["translation"][self.tgt_lang]

        src_seq = self.src_tok.encode(src_seq).ids
        tgt_seq = self.tgt_tok.encode(tgt_seq).ids

        pad_len_src = self.seq_len - len(src_seq) - 2  # -2 for sos and eos
        pad_len_tgt = self.seq_len - len(tgt_seq) - 1
        return {
            "src_seq": torch.tensor(
                [self.sos] + src_seq + [self.eos] + [self.pad] * pad_len_src,
                dtype=torch.int64,
            ),
            "tgt_seq_shifted": torch.tensor(
                [self.sos] + tgt_seq + [self.pad] * pad_len_tgt, dtype=torch.int64
            ),  # used to train the model, that is, this is the 'decoder input' tensor
            "tgt_seq_labels": torch.tensor(
                tgt_seq + [self.eos] + [self.pad] * pad_len_tgt, dtype=torch.int64
            ),  # used to calculate the loss, that is, this is the 'label' tensor
        }
