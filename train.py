from dataset import get_dataset
from torch.utils.data import DataLoader
from model import Transformer
from config import get_config_and_parser
import torch
from training import train_transformer


def main():
    ds_config, model_config, training_config, _ = get_config_and_parser(update=True)
    train_ds, valid_ds = get_dataset(ds_config)
    vocab_size = train_ds.src_tok.get_vocab_size()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("============================INFO============================")
    print("Device:", device)
    print("Dataset config:", ds_config)
    print("Model config:", model_config)
    print("Training config:", training_config)
    print("Vocab size:", vocab_size)
    print("===========================================================")
    train_dl = DataLoader(train_ds, batch_size=training_config.batch_size, shuffle=True)
    transformer = (
        Transformer.from_config(model_config, vocab_size, vocab_size)
        .load_from_checkpoint(training_config.checkpoint_path)
        .to_parallel()
        .to(device)
    )
    train_transformer(
        transformer, train_dl, device, train_ds.src_tok, model_config, training_config
    )


if __name__ == "__main__":
    main()
