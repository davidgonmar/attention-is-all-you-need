from dataset import get_dataset
from model import Transformer
from config import get_config_and_parser
import torch
from training import train_transformer


def main():
    ds_config, model_config, training_config, _, _ = get_config_and_parser(update=True)
    train_ds, valid_ds = get_dataset(ds_config, model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("============================INFO============================")
    print("Device:", device)
    print("Dataset config:", ds_config)
    print("Model config:", model_config)
    print("Training config:", training_config)
    print("Src vocab size:", model_config.src_vocab_size)
    print("Tgt vocab size:", model_config.tgt_vocab_size)
    print("===========================================================")
    transformer = (
        Transformer.from_config(model_config)
        .load_from_checkpoint(training_config.checkpoint_path)
        .to_parallel()
        .to(device)
    )
    train_transformer(
        transformer, train_ds, device, train_ds.src_tok, model_config, training_config
    )


if __name__ == "__main__":
    main()
