from dataset import DatasetConfig, get_dataset
from torch.utils.data import DataLoader
from model import Transformer, get_parallel_model
from training import get_padding_mask
import torch
from config import ModelConfig
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_config = DatasetConfig()
    model_config = ModelConfig()

    tr_config.add_to_arg_parser(parser)
    model_config.add_to_arg_parser(parser)

    tr_config.update_from_arg_parser(parser.parse_args())
    model_config.update_from_arg_parser(parser.parse_args())
    
    train_ds, valid_ds = get_dataset(tr_config)
    vocab_size = train_ds.src_tok.get_vocab_size()

    train_dl = DataLoader(train_ds, batch_size=12, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=16, shuffle=False)
    

    transformer = Transformer(
        num_heads=model_config.num_heads,
        d_model=model_config.d_model,
        d_k=model_config.d_k,
        d_v=model_config.d_v,
        d_ff=model_config.d_ff,
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
    ).to(device)

    if tr_config.model_path:
        try:
            transformer.load_state_dict(torch.load(tr_config.model_path))
        except:
            raise FileNotFoundError("Model not found" + tr_config.model_path)

    transformer = get_parallel_model(transformer)

    def translate(model, src_sentence, src_tok, tgt_tok, max_len=tr_config.seq_len):
        model.eval()

        # Tokenize source sentence
        src_ids = torch.tensor([src_tok.encode(src_sentence).ids], dtype=torch.long)

        # Initialize target with start token
        tgt_ids = torch.tensor([[tgt_tok.token_to_id("<s>")]], dtype=torch.long).to(device)

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
