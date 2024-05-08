"""
Script used to generate a tokenizer from the config dataset.
"""

from config import get_config_and_parser
from dataset import get_raw_dataset
from tokenizer import train_tokenizer


if __name__ == "__main__":
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
