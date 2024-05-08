from datasets import (
    load_dataset,
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
)
from config import DatasetConfig
import torch

DATA_DIR = "data"


def get_processed_dataset_path() -> str:
    return f"{DATA_DIR}/preprocessed"


def retrieve_processed_dataset() -> Dataset | DatasetDict:
    return DatasetDict.load_from_disk(get_processed_dataset_path())


def get_raw_dataset(
    config: DatasetConfig,
) -> Dataset | DatasetDict | IterableDataset | IterableDatasetDict:
    try:
        datasetdict = load_dataset(
            config.ds_name, f"{config.src_lang}-{config.tgt_lang}", cache_dir="data"
        )
    except:  # noqa
        datasetdict = load_dataset(
            config.ds_name, f"{config.tgt_lang}-{config.src_lang}", cache_dir="data"
        )
    return datasetdict


def collate_fn(
    batch: list[dict[str, list[int]]], pad_idx: int
) -> dict[str, torch.Tensor]:
    """
    Collate function used to pad the sequences in the batch.
    """
    with torch.no_grad():
        src = [torch.tensor(item["src"]) for item in batch]
        tgt_shifted = [torch.tensor(item["tgt_shifted"]) for item in batch]
        tgt_labels = [torch.tensor(item["tgt_labels"]) for item in batch]

        src_padded = torch.nn.utils.rnn.pad_sequence(
            src, batch_first=True, padding_value=pad_idx
        )
        tgt_shifted_padded = torch.nn.utils.rnn.pad_sequence(
            tgt_shifted, batch_first=True, padding_value=pad_idx
        )
        tgt_labels_padded = torch.nn.utils.rnn.pad_sequence(
            tgt_labels, batch_first=True, padding_value=pad_idx
        )
    return {
        "src": src_padded,
        "tgt_shifted": tgt_shifted_padded,
        "tgt_labels": tgt_labels_padded,
    }
