from dataset import get_dataset
from model import Transformer
from training import get_padding_mask
import torch
from config import get_config_and_parser


if __name__ == "__main__":
    (
        ds_config,
        model_config,
        tr_config,
        _,
        get_config_and_parser,
    ) = get_config_and_parser(update=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, _ = get_dataset(ds_config, model_config)
    vocab_size = train_ds.src_tok.get_vocab_size()

    transformer = (
        Transformer.from_config(model_config, vocab_size, vocab_size)
        .load_from_checkpoint(tr_config.checkpoint_path)
        .to(device)
    )

    def translate(model, src_sentence, src_tok, tgt_tok, max_len=ds_config.seq_len):
        model.eval()

        src_ids = torch.tensor([src_tok.encode(src_sentence).ids], dtype=torch.long)
        tgt_ids = torch.tensor([[tgt_tok.token_to_id("<s>")]], dtype=torch.long).to(
            device
        )

        src_len = src_ids.size(1)

        # pad both to max_len
        pad_tok = src_tok.token_to_id("<pad>")

        # pad src_ids
        src_ids = torch.cat(
            [
                torch.tensor(src_tok.encode("<s>").ids, dtype=torch.long).unsqueeze(0),
                src_ids,
                torch.tensor(src_tok.encode("</s>").ids, dtype=torch.long).unsqueeze(0),
                torch.tensor([[pad_tok] * (max_len - src_len)], dtype=torch.long),
            ],
            dim=1,
        ).to(device)

        # generate translation iteratively
        for _ in range(max_len - 1):
            with torch.no_grad():
                src_padding_mask = get_padding_mask(
                    src_ids, src_tok.token_to_id("<pad>")
                ).to(device)

                # generate autoregressively
                output = model(src_ids, tgt_ids, src_padding_mask, None)
                next_token = torch.argmax(output[:, -1], dim=-1).unsqueeze(1)
                tgt_ids = torch.cat([tgt_ids, next_token], dim=1)

                # break if end of sentence token is generated
                if next_token.item() == tgt_tok.token_to_id("</s>"):
                    break

        translation = tgt_tok.decode(tgt_ids[0].tolist())
        return translation, tgt_ids

    while True:
        src_sentence = input("Enter a sentence to translate: ")
        if src_sentence == "exit":
            break
        string, tensor = translate(
            transformer, src_sentence, train_ds.src_tok, train_ds.tgt_tok
        )
        print(string)
