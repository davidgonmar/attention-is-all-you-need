import torch
from dataset import retrieve_processed_dataset
from model import Transformer
from config import get_config_and_parser

import torch.distributed as dist
from dataset import (
    collate_fn,
    CustomDistributedSampler,
    get_batches_dict,
)
from torch.utils.data import DataLoader
from tokenizer import SpecialTokens, get_tokenizer
import os
import sacrebleu


def validate_model(model, test_dl, device, ds_config, training_config, bleu="estimate"):
    from train import get_padding_mask

    local_loss = 0
    local_bleu = 0
    with torch.no_grad():
        model.eval()
        for i, elem in enumerate(test_dl):
            encoder_input = elem["src"].to(device)
            decoder_input = elem["tgt_shifted"].to(device)
            labels = elem["tgt_labels"].to(device)
            tokenizer = get_tokenizer()
            src_mask = get_padding_mask(
                encoder_input, tokenizer.token_to_id(SpecialTokens.PAD.value)
            ).to(device)
            tgt_mask = get_padding_mask(
                decoder_input, tokenizer.token_to_id(SpecialTokens.PAD.value)
            ).to(device)

            # Forward pass
            out = model(encoder_input, decoder_input, src_mask, tgt_mask)
            loss = torch.nn.functional.cross_entropy(
                out.view(-1, out.shape[-1]),
                labels.view(-1),
                ignore_index=tokenizer.token_to_id(SpecialTokens.PAD.value),
                reduction="mean",
                label_smoothing=training_config.label_smoothing
            )
            local_loss += loss.item()

            # Calculate BLEU
            if bleu == "estimate":
                out = out.argmax(dim=-1)
                for j in range(out.size(0)):  # iterate over batch
                    ref = labels[j].tolist()
                    cands = out[j].tolist()
                    ref = tokenizer.decode(ref).replace(" ##", "")
                    cands = tokenizer.decode(cands).replace(" ##", "")
                    sc = sacrebleu.corpus_bleu([cands], [[ref]]).score
                    local_bleu += sc / out.size(0)
            else:
                for j in range(encoder_input.size(0)):  # iterate over batch
                    ref = labels[j].tolist()
                    cands = model.module.generate(
                        tokenizer,
                        encoder_input[i].unsqueeze(0),
                        src_mask[i].unsqueeze(0),
                    )[0].tolist()
                    ref = tokenizer.decode(ref).replace(" ##", "")
                    cands = tokenizer.decode(cands).replace(" ##", "")
                    sc = sacrebleu.corpus_bleu([cands], [[ref]]).score
                    local_bleu += sc / out.size(0)
    local_loss /= len(test_dl)
    local_bleu /= len(test_dl)

    # Aggregate losses and BLEU scores across all processes
    total_loss = torch.tensor(local_loss, device=device)
    total_bleu = torch.tensor(local_bleu, device=device)

    # Sum all values across processes
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_bleu, op=dist.ReduceOp.SUM)

    # Calculate average by dividing by world size
    avg_loss = total_loss.item() / dist.get_world_size()
    avg_bleu = total_bleu.item() / dist.get_world_size()

    return avg_loss, avg_bleu


if __name__ == "__main__":
    ds_config, model_config, tr_config, eval_config, parser = get_config_and_parser()
    test_ds = retrieve_processed_dataset()["test"]
    tokenizer_tgt = get_tokenizer(ds_config.tgt_lang)
    pad_id = tokenizer_tgt.token_to_id(SpecialTokens.PAD.value)
    args = parser.parse_args()
    dist.init_process_group(backend="nccl")
    local_rank, global_rank, world_size = (
        int(os.environ["LOCAL_RANK"]),
        int(os.environ["RANK"]),
        int(os.environ["WORLD_SIZE"]),
    )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda")
    test_dl = DataLoader(
        test_ds,
        batch_sampler=CustomDistributedSampler(get_batches_dict()["test"]),
        collate_fn=lambda x: collate_fn(x, pad_id),
    )
    transformer = (
        Transformer.from_config(model_config)
        .load_from_checkpoint(tr_config.checkpoint_path)
        .to(device)
    )
    transformer = torch.nn.parallel.DistributedDataParallel(transformer)
    avg_loss, avg_bleu = validate_model(
        transformer, test_dl, device, ds_config, tr_config, bleu="real"
    )

    if global_rank == 0:
        print("Average loss: ", avg_loss)
        print("Average BLEU: ", avg_bleu)
