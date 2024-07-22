import torch
from dataset import retrieve_processed_dataset
from model import Transformer
from config import get_config_and_parser
from train import get_padding_mask
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

if __name__ == "__main__":
    ds_config, model_config, tr_config, eval_config, parser = get_config_and_parser(
        update=True,
        extra_args=[{"args": ["--ddp"], "kwargs": {"action": "store_true"}}],
    )
    test_ds = retrieve_processed_dataset()["test"]
    tokenizer_tgt = get_tokenizer(ds_config.tgt_lang)
    pad_id = tokenizer_tgt.token_to_id(SpecialTokens.PAD.value)
    args = parser.parse_args()
    if args.ddp:
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
    else:
        device = torch.device("cuda")
        transformer = (
            Transformer.from_config(model_config)
            .load_from_checkpoint(tr_config.checkpoint_path)
            .to(device)
        )
        test_dl = DataLoader(
            test_ds,
            batch_size=128,
            collate_fn=lambda x: collate_fn(x, pad_id),
        )
    avg_loss = 0
    avg_bleu = 0
    with torch.no_grad():
        transformer.eval()
        for i, elem in enumerate(test_dl):
            encoder_input = elem["src"].to(device)
            decoder_input = elem["tgt_shifted"].to(device)
            labels = elem["tgt_labels"].to(device)
            src_tok, tgt_tok = get_tokenizer(ds_config.src_lang), get_tokenizer(
                ds_config.tgt_lang
            )
            src_mask = get_padding_mask(
                encoder_input, src_tok.token_to_id(SpecialTokens.PAD.value)
            ).to(device)
            tgt_mask = get_padding_mask(
                decoder_input, tgt_tok.token_to_id(SpecialTokens.PAD.value)
            ).to(device)

            out = transformer(encoder_input, decoder_input, src_mask, tgt_mask)
            loss = torch.nn.functional.cross_entropy(
                out.view(-1, out.shape[-1]),
                labels.view(-1),
                ignore_index=tgt_tok.token_to_id(SpecialTokens.PAD.value),
                reduction="mean",
            )
            avg_loss += loss.item()
            print(" Batch number:", i, " out of:", len(test_dl), end="\r", flush=True)

            # calculate BLEU
            s = "Schöne Tiere und leckere Torten locken"
            tok = tgt_tok.encode(s).ids
            print(tok)
            print(tgt_tok.decode(tok))
            out = out.argmax(dim=-1)
            for j in range(out.size(0)):  # iterate over batch
                ref = labels[j].tolist()
                cands = out[j].tolist()
                ref = tgt_tok.decode(ref).replace(" ##", "")
                cands = tgt_tok.decode(cands).replace(" ##", "")
                if j == 0:
                    print("ref: ", ref, "\ncands: ", cands)
                sc = sacrebleu.corpus_bleu(cands, [ref]).score
                avg_bleu += sc

        avg_loss /= len(test_dl)
        avg_bleu /= len(test_ds)  # average BLEU score over the entire dataset

    print("Average loss:", avg_loss)
    print("Average BLEU score:", avg_bleu)
