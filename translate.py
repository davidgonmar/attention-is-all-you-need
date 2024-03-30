from dataset import Config, get_dataset, get_padding_mask
from torch.utils.data import DataLoader
from model import Transformer

import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = Config(ds_name="opus_books", src_lang="en", tgt_lang="es", split=0.9)

train_ds, valid_ds = get_dataset(config)

vocab_size = train_ds.src_tok.get_vocab_size()

print("Vocab size:", vocab_size)


train_dl = DataLoader(train_ds, batch_size=12, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=16, shuffle=False)


transformer = Transformer(
    num_heads=8,
    d_model=512,
    d_k=64,
    d_v=64,
    d_ff=2048,
    src_vocab_size=vocab_size,
    tgt_vocab_size=vocab_size,
).to(device)

if config.model_path:
    try:
        transformer.load_state_dict(torch.load(config.model_path))
    except:
        raise FileNotFoundError("Model not found" + config.model_path)

import torch

def translate(model, src_sentence, src_tok, tgt_tok, max_len=config.seq_len):
    model.eval()

    # Tokenize source sentence
    src_ids = torch.tensor([src_tok.encode(src_sentence).ids], dtype=torch.long)

    # Initialize target with start token
    tgt_ids = torch.tensor([[tgt_tok.token_to_id("<s>")]], dtype=torch.long)

    # Ensure src and tgt have the same length
    src_len = src_ids.size(1)

    # pad both to max_len
    # shape (1, max_len)
    pad_tok = src_tok.token_to_id("<pad>")

    src_ids = torch.cat([src_ids, torch.tensor([[pad_tok] * (max_len - src_len)], dtype=torch.long)], dim=1).to(device)

    # Generate translation iteratively
    for i in range(max_len - 1):
        with torch.no_grad():
            src_padding_mask = get_padding_mask(src_ids, src_tok.token_to_id("<pad>")).to(device)
            # Generate next token
            output = model(src_ids, tgt_ids, src_padding_mask, None)
            next_token = torch.argmax(output[:, -1], dim=-1).unsqueeze(1)

            # put the next token in the tgt_ids. Dont use cat since it is padded

            tgt_ids = torch.cat([tgt_ids, next_token], dim=1)


            # Check if end token is generated
            if next_token.item() == tgt_tok.token_to_id("</s>"):
                break

    # Decode target sequence
    translation = tgt_tok.decode(tgt_ids[0].tolist())
    print(tgt_ids)

    # Print or return translation
    print(translation)
    return translation



if __name__ == "__main__":
    # ask user for input
    while True:
        src_sentence = input("Enter a sentence to translate: ")
        if src_sentence == "exit":
            break
        translate(transformer, src_sentence, train_ds.src_tok, train_ds.tgt_tok)
