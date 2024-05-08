from tokenizer import get_tokenizer
from config import get_config_and_parser
from datasets import Dataset, DatasetDict
from tokenizers import Tokenizer
from tokenizer import SpecialTokens
from dataset import get_raw_dataset, get_processed_dataset_path, DatasetConfig
from torch.utils.data import random_split
import torch


def preprocess(
    ds: Dataset | DatasetDict,
    ds_config: DatasetConfig,
    tokenizer_src: Tokenizer,
    tokenizer_tgt: Tokenizer,
) -> Dataset | DatasetDict:
    """
    Given a dataset, pretokenizes it and stores it at 'save_path'.
    """

    def build(item):
        # Padding and truncation will be done manually in the collate function
        # Here we just pretokenize the text:
        # src = encoder input
        # tgt_shifted = decoder input
        # tgt_labels = expected (decoder) output
        BOS, EOS = SpecialTokens.BOS.value, SpecialTokens.EOS.value
        src = tokenizer_src.encode(
            BOS + item["translation"][ds_config.src_lang] + EOS
        ).ids
        tgt_shifted = tokenizer_tgt.encode(
            BOS + item["translation"][ds_config.tgt_lang]
        ).ids
        tgt_labels = tokenizer_tgt.encode(
            item["translation"][ds_config.tgt_lang] + EOS
        ).ids

        return {"src": src, "tgt_shifted": tgt_shifted, "tgt_labels": tgt_labels}

    # 1. Split the dataset, manually if it doesn't have a test split or using the existing one
    if "train" in ds.keys() and "test" in ds.keys():
        train_ds = ds["train"]
        test_ds = ds["test"]
    else:
        assert (
            "train" in ds.keys()
        ), f"Dataset must have a train split, got dataset: {ds}"
        train_size = int(ds_config.split * len(ds["train"]))
        test_size = len(ds["train"]) - train_size
        train_ds, test_ds = random_split(
            ds["train"],
            [train_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )  # so the dataset remains the same

    # 2. Preprocess the datasets
    train_ds = train_ds.map(build)
    test_ds = test_ds.map(build)

    # 3. Order by length for better padding utilization
    train_ds = train_ds.sort("src", descending=True)

    return DatasetDict(train=train_ds, test=test_ds)


def main():
    ds_config, _, _, _, _ = get_config_and_parser()
    tokenizer_src = get_tokenizer(ds_config.src_lang)
    tokenizer_tgt = get_tokenizer(ds_config.tgt_lang)
    ds = get_raw_dataset(ds_config)
    pretokenized = preprocess(ds, ds_config, tokenizer_src, tokenizer_tgt)
    pretokenized.save_to_disk(get_processed_dataset_path())


if __name__ == "__main__":
    main()
