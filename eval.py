import torch
from dataset import get_dataset
from model import Transformer
from config import get_config_and_parser
from training import get_padding_mask
from torchtext.data.metrics import bleu_score

if __name__ == "__main__":
    ds_config, model_config, tr_config, eval_config, parser = get_config_and_parser(
        update=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, test_ds = get_dataset(ds_config, model_config)
    vocab_size = test_ds.src_tok.get_vocab_size()
    dl = torch.utils.data.DataLoader(
        test_ds, batch_size=eval_config.batch_size, shuffle=False
    )
    transformer = (
        Transformer.from_config(model_config, vocab_size, vocab_size)
        .load_from_checkpoint(tr_config.checkpoint_path)
        .to_parallel()
        .to(device)
    )

    avg_loss = 0
    avg_bleu = 0
    with torch.no_grad():
        for i, elem in enumerate(dl):
            encoder_input = elem["src"].to(device)
            decoder_input = elem["tgt_shifted"].to(device)
            labels = elem["tgt_labels"].to(device)
            src_mask = get_padding_mask(
                encoder_input, test_ds.src_tok.token_to_id("<pad>")
            ).to(device)
            tgt_mask = get_padding_mask(
                decoder_input, test_ds.tgt_tok.token_to_id("<pad>")
            ).to(device)

            out = transformer(encoder_input, decoder_input, src_mask, tgt_mask)
            loss = torch.nn.functional.cross_entropy(
                out.view(-1, vocab_size),
                labels.view(-1),
                ignore_index=test_ds.tgt_tok.token_to_id("<pad>"),
                reduction="mean",
            )
            avg_loss += loss.item()
            print(" Batch number:", i, " out of:", len(dl), end="\r", flush=True)

            # calculate BLEU
            out = out.argmax(dim=-1)
            for j in range(out.size(0)):  # iterate over batch
                ref = labels[j].tolist()
                cands = out[j].tolist()
                ref = test_ds.tgt_tok.decode(ref).split()
                cands = test_ds.tgt_tok.decode(cands).split()
                sc = bleu_score([cands], [[ref]])
                avg_bleu += sc

        avg_loss /= len(dl)
        avg_bleu /= len(test_ds)  # average BLEU score over the entire dataset

    print("Average loss:", avg_loss)
    print("Average BLEU score:", avg_bleu)
