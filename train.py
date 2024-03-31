from dataset import get_dataset
from torch.utils.data import DataLoader
from model import Transformer, get_parallel_model
from config import TrainingConfig, DatasetConfig, ModelConfig
import torch
from training import train_transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ds_config = DatasetConfig()
model_config = ModelConfig()
training_config = TrainingConfig()

train_ds, valid_ds = get_dataset(ds_config)

vocab_size = train_ds.src_tok.get_vocab_size()

train_dl = DataLoader(train_ds, batch_size=training_config.batch_size, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=training_config.batch_size, shuffle=False)


transformer = Transformer(
    num_heads=model_config.num_heads,
    d_model=model_config.d_model,
    d_k=model_config.d_k,
    d_v=model_config.d_v,
    d_ff=model_config.d_ff,
    src_vocab_size=vocab_size,
    tgt_vocab_size=vocab_size,
).to(device)

transformer = get_parallel_model(transformer) # will try to parallelize the model if possible

if training_config.model_path:
    try:
        transformer.load_state_dict(torch.load(training_config.model_path)["model"])
    except:
        print("Model not found", training_config.model_path)


train_transformer(transformer, train_dl, device, train_ds.src_tok, model_config, training_config)