from datasets import (
    load_dataset,
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
)
from config import DatasetConfig
import torch

from typing import Iterator

import torch.distributed as dist
from torch.utils.data import Sampler

import json
from tokenizer import get_tokenizer
from config import get_config_and_parser
from tokenizers import Tokenizer
from tokenizer import SpecialTokens
from torch.utils.data import random_split
from tqdm import tqdm
import pandas as pd

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


# custom distributed sampler. the objective is to replicate the paper's implementation
# they mention that each batch has X (in the paper 25k) src and X tgt tokens
# what we'll do is keep sampling until we have either X src or X tgt tokens
class CustomDistributedSampler(Sampler):
    def __init__(self, batches_dict: dict) -> None:
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.batches_dict = batches_dict

        # assert that batches_dict n of gpus == world_size
        assert (
            len(batches_dict) == self.world_size
        ), f"Number of gpus in batches_dict ({len(batches_dict)}) is different from world_size ({self.world_size})"

    def __iter__(self) -> Iterator:
        # batches_dict is of the form
        # {
        #     "gpu_0": [[...idxs...], [...idxs...], ...],
        #     "gpu_1": [[...idxs...], [...idxs...], ...],
        #     ...
        # }
        idxs = self.batches_dict[
            f"gpu_{self.rank}"
        ]  # [[...idxs...], [...idxs...], ...]
        randomperm = torch.randperm(len(idxs))
        return iter([idxs[i] for i in randomperm])

    def __len__(self) -> int:
        return len(self.batches_dict[f"gpu_{self.rank}"])


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
    # use pandas since huggingface datasets can't be sorted with a lambda
    train_ds_df = pd.DataFrame(train_ds)
    train_ds_df["combined_length"] = train_ds_df.apply(
        lambda x: len(x["src"]) + len(x["tgt_shifted"]), axis=1
    )
    train_ds_df = train_ds_df.sort_values(by="combined_length")
    # delete the combined_length column
    train_ds_df = train_ds_df.drop(columns=["combined_length"])

    test_ds_df = pd.DataFrame(test_ds)
    test_ds_df["combined_length"] = test_ds_df.apply(
        lambda x: len(x["src"]) + len(x["tgt_shifted"]), axis=1
    )
    test_ds_df = test_ds_df.sort_values(by="combined_length")
    # delete the combined_length column
    test_ds_df = test_ds_df.drop(columns=["combined_length"])

    train_ds = Dataset.from_pandas(train_ds_df)
    test_ds = Dataset.from_pandas(test_ds_df)

    return DatasetDict(train=train_ds, test=test_ds)


def organize_batches(ds: Dataset | DatasetDict, num_gpus: int, n_tokens_per_gpu: int):
    # will create a json object with the following structure:
    # {
    #     "train": {
    #         "gpu_0": [[...idxs for batch 0...], [...idxs for batch 1...], ...],
    #         "gpu_1": [[...idxs for batch 0...], [...idxs for batch 1...], ...],
    #         ...
    #     },
    #     "test": {
    #       ...
    #     }

    # the rules are that each gpu gets the same number of batches, and that each batch has approximately n_tokens_per_gpu tokens in src or tgt (whatever comes first)
    # the idxs for each batch on each gpu ARE NOT NECESSARILY THE SAME, but they are guaranteed to have approximately the same number of tokens
    obj = {"train": {}, "test": {}}

    # add gpus to both train and test
    for split in ["train", "test"]:
        for gpu in range(num_gpus):
            obj[split][f"gpu_{gpu}"] = []

    # iterate over the dataset
    for split in ["train", "test"]:
        curr_batches = {
            "gpu" + str(i): {"total_tokens": 0, "idxs": []} for i in range(num_gpus)
        }
        currgpu = -1
        for idx, item in tqdm(
            enumerate(ds[split]), desc=f"Processing {split} split", total=len(ds[split])
        ):
            currgpu = (currgpu + 1) % num_gpus
            src_len = len(item["src"])
            tgt_len = len(item["tgt_shifted"])
            max_len = max(src_len, tgt_len)

            # 1. Filter out items that are too big (technically wont happen)
            if max_len > n_tokens_per_gpu:
                # if the current item is too big, we skip it
                print(
                    f"Skipping item {idx} in {split} split, with src_len: {src_len} and tgt_len: {tgt_len}"
                )
                continue

            # 2. Decide which gpu to put the item in (the first one that has enough space)
            while curr_batches[f"gpu{currgpu}"]["total_tokens"] + max_len > (
                n_tokens_per_gpu * 1.1
            ):  # we add a 10% margin, so it's not always < n_tokens_per_gpu but approximately n_tokens_per_gpu
                currgpu += 1
                # 2.1 If all gpus are full, we reset the counters
                if currgpu == num_gpus:
                    for gpu in range(num_gpus):
                        curr_batches[f"gpu{gpu}"]["total_tokens"] = 0
                        # we append the idxs to the object
                        obj[split][f"gpu_{gpu}"].append(
                            curr_batches[f"gpu{gpu}"]["idxs"]
                        )
                        curr_batches[f"gpu{gpu}"]["idxs"] = []
                    currgpu = 0
                    break

            # 3. Add the item to the current gpu
            curr_batches[f"gpu{currgpu}"]["total_tokens"] += max_len
            curr_batches[f"gpu{currgpu}"]["idxs"].append(idx)

            # 4. If we finished the loop, we append the leftover idxs to the object
            if idx == len(ds[split]) - 1:
                for gpu in range(num_gpus):
                    obj[split][f"gpu_{gpu}"].append(curr_batches[f"gpu{gpu}"]["idxs"])

    # 4. We finished. Save the object to disk
    with open(get_processed_dataset_path() + "/batches.json", "w") as f:
        json.dump(obj, f)


def print_batch_stats(ds: Dataset | DatasetDict, obj: dict):
    # first, check that the n batches for each split and gpu is the same
    for split in ["train", "test"]:
        gpu0_nbatches = len(obj[split]["gpu_0"])
        for gpu in range(1, len(obj[split])):
            assert (
                len(obj[split][f"gpu_{gpu}"]) == gpu0_nbatches
            ), f"Number of batches for split {split} and gpu_0 is different from gpu_{gpu}"
    for split in ["train", "test"]:
        for gpu in range(len(obj[split])):
            print(f"Stats for {split} split, gpu_{gpu}: ")
            src_tokens = 0
            tgt_tokens = 0
            max_src_tokens = 0
            max_tgt_tokens = 0
            min_src_tokens = 1e9
            min_tgt_tokens = 1e9
            gpubatches = obj[split][f"gpu_{gpu}"]
            idxs = 0
            for batch in tqdm(
                gpubatches, desc="Processing batches", total=len(gpubatches)
            ):
                idxs += len(batch)
                local_src_tokens = 0
                local_tgt_tokens = 0
                for idx in batch:
                    ds_idx = ds[split][idx]
                    src = ds_idx["src"]
                    tgt = ds_idx["tgt_shifted"]
                    srclen = len(src)
                    tgtlen = len(tgt)
                    src_tokens += srclen
                    tgt_tokens += tgtlen
                    local_src_tokens += srclen
                    local_tgt_tokens += tgtlen
                # min and max number of tokens in a batch
                max_src_tokens = max(max_src_tokens, local_src_tokens)
                max_tgt_tokens = max(max_tgt_tokens, local_tgt_tokens)
                min_src_tokens = min(min_src_tokens, local_src_tokens)
                min_tgt_tokens = min(min_tgt_tokens, local_tgt_tokens)
            print("Total batches:", len(obj[split][f"gpu_{gpu}"]))
            print("Average src tokens:", src_tokens / len(obj[split][f"gpu_{gpu}"]))
            print("Average tgt tokens:", tgt_tokens / len(obj[split][f"gpu_{gpu}"]))
            print("Max src tokens:", max_src_tokens)
            print("Max tgt tokens:", max_tgt_tokens)
            print("Min src tokens:", min_src_tokens)
            print("Min tgt tokens:", min_tgt_tokens)
            print("Total tokens:", src_tokens)
            print("Total idxs:", idxs)
            print()


def create_ds(ds_config):
    tokenizer_src = get_tokenizer(ds_config.src_lang)
    tokenizer_tgt = get_tokenizer(ds_config.tgt_lang)
    ds = get_raw_dataset(ds_config)
    pretokenized = preprocess(ds, ds_config, tokenizer_src, tokenizer_tgt)
    pretokenized.save_to_disk(get_processed_dataset_path())


def get_batches_dict() -> dict:
    with open(get_processed_dataset_path() + "/batches.json", "r") as f:
        obj = json.load(f)
    return obj


if __name__ == "__main__":
    # task can be "preprocess" or "organize_batches" or "print_batch_stats"
    (
        ds_config,
        model_config,
        training_config,
        eval_config,
        parser,
    ) = get_config_and_parser(
        extra_args=[{"args": ["--task"], "kwargs": {"type": str, "required": True}}]
    )
    args = parser.parse_args()
    if args.task == "preprocess":
        create_ds(ds_config)
    elif args.task == "organize_batches":
        ds = DatasetDict.load_from_disk(get_processed_dataset_path())
        organize_batches(
            ds, training_config.n_gpus, training_config.tokens_per_step_per_gpu
        )
    elif args.task == "print_batch_stats":
        with open(get_processed_dataset_path() + "/batches.json", "r") as f:
            obj = json.load(f)
        ds = DatasetDict.load_from_disk(get_processed_dataset_path())
        print_batch_stats(ds, obj)
