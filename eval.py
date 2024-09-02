import torch
from dataset import retrieve_processed_dataset
from model import Transformer
from config import get_config_and_parser
from dataset import (
    collate_fn,
    CustomDistributedSampler,
    get_batches_dict,
)
from torch.utils.data import DataLoader
from tokenizer import SpecialTokens, get_tokenizer
import sacrebleu
from torch.distributed.elastic.multiprocessing.errors import record
from tqdm import tqdm


def pad_sequences(sequences, pad_token_id, max_len):
    """Pads a list of sequences to the same length."""
    padded_sequences = [
        (seq + [pad_token_id] * (max_len - len(seq))) for seq in sequences
    ]
    return padded_sequences


def validate_model(model, test_dl, device, ds_config, training_config):
    local_loss = 0
    with torch.no_grad():
        model.eval()
        cands = []
        refs = []
        tokenizer = get_tokenizer()
        for i, elem in enumerate(tqdm(test_dl, desc="Validation", unit="batch")):
            encoder_input = elem["src"].to(device)
            decoder_input = elem["tgt_shifted"].to(device)
            labels = elem["tgt_labels"].to(device)
            src_mask = elem["src_mask"].to(device)
            tgt_mask = elem["tgt_mask"].to(device)
            with torch.autocast("cuda", enabled=True):
                out = model(encoder_input, decoder_input, src_mask, tgt_mask)
                loss = torch.nn.functional.cross_entropy(
                    out.view(-1, out.shape[-1]),
                    labels.view(-1),
                    ignore_index=tokenizer.token_to_id(SpecialTokens.PAD.value),
                    reduction="mean",
                )
                local_loss += loss.item()

                for j in range(encoder_input.size(0)):  # iterate over batch
                    ref = labels[j].tolist()
                    cand = model.generate(
                        tokenizer,
                        encoder_input[j],
                        src_mask[j],
                    ).tolist()
                    cands.append(cand)
                    refs.append(ref)

        decoded_refs = tokenizer.decode_batch(refs)
        decoded_cands = tokenizer.decode_batch(cands)
        bleu = sacrebleu.corpus_bleu(decoded_cands, [decoded_refs]).score

        return bleu, local_loss / len(test_dl)


if __name__ == "__main__":

    @record
    def main():
        ds_config, model_config, tr_config, parser = get_config_and_parser()
        test_ds = retrieve_processed_dataset()["test"]
        tokenizer = get_tokenizer()
        pad_id = tokenizer.token_to_id(SpecialTokens.PAD.value)
        device = torch.device("cuda")
        test_dl = DataLoader(
            test_ds,
            batch_sampler=CustomDistributedSampler(
                get_batches_dict()["test"], ranks=list(range(tr_config.n_gpus))
            ),  # make aa single gpu use the dataset split for all gpus
            collate_fn=lambda x: collate_fn(x, pad_id),
        )
        transformer = (
            Transformer.from_config(model_config)
            .load_from_checkpoint(tr_config.checkpoint_path)
            .to(device)
        )
        avg_bleu, avg_loss = validate_model(
            transformer, test_dl, device, ds_config, tr_config
        )

        print("avg bleu: ", avg_bleu)
        print("avg loss: ", avg_loss)

    main()
