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
import random

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
        bs, slmax = src_padded.shape
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
    tokenizer: Tokenizer,
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
        src = tokenizer.encode(BOS + item["translation"][ds_config.src_lang] + EOS).ids
        tgt = tokenizer.encode(BOS + item["translation"][ds_config.tgt_lang] + EOS).ids
        tgt_shifted = tgt[:-1]
        tgt_labels = tgt[1:]
        combined_length = len(src) + len(tgt)

        return {
            "src": src,
            "tgt_shifted": tgt_shifted,
            "tgt_labels": tgt_labels,
            "combined_length": combined_length,
        }

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
    train_ds = train_ds.sort("combined_length")
    test_ds = test_ds.sort("combined_length")

    # 4. We don't want the combined_length attribute anymore!
    colnames = ["src", "tgt_labels", "tgt_shifted"]
    train_ds = train_ds.select_columns(colnames)
    test_ds = test_ds.select_columns(colnames)

    print("Finished preprocessing dataset.")
    print("Train length: ", len(train_ds))
    print("Test length: ", len(test_ds))

    return DatasetDict(train=train_ds, test=test_ds)


def organize_batches(ds: Dataset | DatasetDict, num_gpus: int, n_tokens_per_gpu: int):
    obj = {"train": {}, "test": {}}
    # add gpus to both train and test
    for split in ["train", "test"]:
        for gpu in range(num_gpus):
            obj[split][f"gpu_{gpu}"] = []

    # iterate over the dataset
    for split in ["test", "train"]:
        curr_batches = [
            {"total_tokens_src": 0, "total_tokens_tgt": 0, "idxs": []}
            for _ in range(num_gpus)
        ]
        for idx, item in tqdm(
            enumerate(ds[split]), desc=f"Processing {split} split", total=len(ds[split])
        ):
            src_len = len(item["src"])
            tgt_len = len(item["tgt_shifted"])
            max_len = max(src_len, tgt_len)

            if max_len > n_tokens_per_gpu:
                print(
                    f"Skipping item {idx} in {split} split, with src_len: {src_len} and tgt_len: {tgt_len}"
                )
                continue

            # find the first GPU with enough space
            assigned = False
            for gpu in range(num_gpus):
                cur = curr_batches[gpu]
                if (cur["total_tokens_src"] + src_len <= n_tokens_per_gpu * 1.05) and (
                    cur["total_tokens_tgt"] + tgt_len <= n_tokens_per_gpu * 1.05
                ):
                    cur["idxs"].append(idx)
                    cur["total_tokens_src"] += src_len
                    cur["total_tokens_tgt"] += tgt_len
                    assigned = True
                    break

            # if no GPU had enough space, we flush the current batches and start new ones
            if not assigned:
                for gpu in range(num_gpus):
                    obj[split][f"gpu_{gpu}"].append(curr_batches[gpu]["idxs"])
                    curr_batches[gpu] = {
                        "total_tokens_src": 0,
                        "total_tokens_tgt": 0,
                        "idxs": [],
                    }

                # assign the current item to the first GPU (all GPUs are reset)
                curr_batches[0]["idxs"].append(idx)
                curr_batches[0]["total_tokens_src"] = src_len
                curr_batches[0]["total_tokens_tgt"] = tgt_len

        # append any remaining items in the current batches to the output object
        for gpu in range(num_gpus):
            if curr_batches[gpu]["idxs"]:
                obj[split][f"gpu_{gpu}"].append(curr_batches[gpu]["idxs"])

        # ensure all GPUs have the same number of batches
        min_batches = min(len(obj[split][f"gpu_{gpu}"]) for gpu in range(num_gpus))
        for gpu in range(num_gpus):
            batches = obj[split][f"gpu_{gpu}"]
            while True:
                if len(batches) == min_batches:
                    break
                random_batch = random.randint(0, len(batches) - 1)
                batches.pop(random_batch)

    # Save the object to disk
    with open(get_processed_dataset_path() + "/batches.json", "w") as f:
        json.dump(obj, f)


def print_batch_stats(ds: Dataset | DatasetDict, obj: dict):
    # first, check that the n batches for each split and gpu is the same
    for split in ["train", "test"]:
        gpu0_nbatches = len(obj[split]["gpu_0"])
        for gpu in range(1, len(obj[split])):
            assert (
                len(obj[split][f"gpu_{gpu}"]) == gpu0_nbatches
            ), f"Number of batches for split {split} and gpu_0 is different from gpu_{gpu}: {gpu0_nbatches}, {len(obj[split][f'gpu_{gpu}'])}"
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
            # it is expected that average tokens is not close to the maximum because of padding/collating
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
    tokenizer = get_tokenizer()
    ds = get_raw_dataset(ds_config)
    pretokenized = preprocess(ds, ds_config, tokenizer)
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
