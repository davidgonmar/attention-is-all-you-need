from dataset import (
    retrieve_processed_dataset,
    collate_fn,
    CustomDistributedSampler,
    get_batches_dict,
)
from model import Transformer
from config import get_config_and_parser
import torch
import torch.nn as nn
from config import ModelConfig, TrainingConfig, DatasetConfig
from typing import Tuple
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from datasets import DatasetDict
import os
from torch.distributed import init_process_group
from tokenizer import get_tokenizer, SpecialTokens
import time
import wandb
from torch.distributed.elastic.multiprocessing.errors import record

torch.backends.cuda.matmul.allow_tf32 = True


def get_optim_and_scheduler(
    model: nn.Module, model_config: ModelConfig, training_config: TrainingConfig
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.lr,
        betas=(training_config.b1, training_config.b2),
        eps=training_config.eps,
    )

    def sq_lambda(i):
        d_model_inv_sqrt = model_config.d_model**-0.5
        step_num = max(i, 1)  # Ensure step_num is at least 1 to avoid division by zero
        warmup_steps = training_config.warmup_steps
        factor = min(step_num**-0.5, step_num * warmup_steps**-1.5)
        return d_model_inv_sqrt * factor

    def cst_lambda(i):
        return 1.0

    lambda_func = sq_lambda if training_config.use_scheduler else cst_lambda

    scheduler = LambdaLR(optimizer, lambda i: lambda_func(i))

    return optimizer, scheduler


def get_padding_mask(seq: torch.Tensor, pad_token: int) -> torch.Tensor:
    """
    Returns a mask tensor representing which elements are padding tokens

    Args:
        seq: tensor of shape (batch_size, seq_len)
        pad_token: token representing padding

    Returns:
        mask tensor of shape (batch_size, 1, 1, seq_len)
    """
    return (seq != pad_token).unsqueeze(1).unsqueeze(2)

def validate_model(
    model: nn.Module,
    test_dl: DataLoader,
    device: torch.device,
    ds_config: DatasetConfig,
    training_config: TrainingConfig,
    pad_id: int,
) -> float:
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_id, reduction="mean")
    total_loss = 0
    with torch.no_grad():
        for _, elem in enumerate(test_dl):
            encoder_input = elem["src"].to(device)
            decoder_input = elem["tgt_shifted"].to(device)
            labels = elem["tgt_labels"].to(device)
            src_mask = get_padding_mask(encoder_input, pad_id)
            tgt_mask = get_padding_mask(decoder_input, pad_id)
            out = model(encoder_input, decoder_input, src_mask, tgt_mask)
            loss = criterion(out.view(-1, out.size(-1)), labels.view(-1))
            total_loss += loss.item()
    return total_loss / len(test_dl)

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    model_config: ModelConfig,
    global_step: int,
    checkpoint_dir: str,
    checkpoint_save_filename: str,
) -> None:
    torch.save(
        {
            "model": (
                model.module.state_dict()
                if hasattr(model, "module")
                else model.state_dict()
            ),  # save the model depending on whether it is wrapped in a DistributedDataParallel or something
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "model_config": model_config,  # does not occupy much space, but useful so we do not accidentally load a different model config
            "global_step": global_step,
        },
        (checkpoint_dir / checkpoint_save_filename.format(global_step=global_step)),
    )

def train_transformer(
    model: Transformer,
    dsdict: DatasetDict,
    device: torch.device,
    pad_id: int,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    ds_config: DatasetConfig,
    global_rank: int = 0,
) -> None:
    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=pad_id,
        reduction="mean",
        label_smoothing=training_config.label_smoothing,
    )

    optimizer, scheduler = get_optim_and_scheduler(model, model_config, training_config)
    train_dl = DataLoader(
        dsdict["train"],
        batch_sampler=CustomDistributedSampler(get_batches_dict()["train"]),
        collate_fn=lambda x: collate_fn(x, pad_id),
        pin_memory=True,
    )
    test_dl = DataLoader(
        dsdict["test"],
        batch_sampler=CustomDistributedSampler(get_batches_dict()["test"]),
        collate_fn=lambda x: collate_fn(x, pad_id),
        pin_memory=True,
    )
    global_step = 0
    if training_config.checkpoint_path is not None:
        # we dont load data here because it is already loaded in the model
        checkpoint = torch.load(training_config.checkpoint_path)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        global_step = checkpoint["global_step"] + 1
        print("Loaded checkpoint from", training_config.checkpoint_path)
        print(
            "info: saved global_step=",
            global_step - 1,
            "starting from global_step=",
            global_step,
        )
        print("info: lr=", optimizer.param_groups[0]["lr"])
        print("info: warmup_steps=", training_config.warmup_steps)
    scaler = torch.amp.GradScaler("cuda")
    model.train()
    accum_loss = 0
    while True:
        for _, elem in enumerate(train_dl):
            start = time.time()
            encoder_input = elem["src"].to(device)
            decoder_input = elem["tgt_shifted"].to(device)
            labels = elem["tgt_labels"].to(device)
            src_mask = get_padding_mask(encoder_input, pad_id)
            tgt_mask = get_padding_mask(decoder_input, pad_id)
            lossitem = None
            with torch.autocast("cuda", enabled=False):
                out = model(encoder_input, decoder_input, src_mask, tgt_mask)
                loss = criterion(
                    out.view(-1, out.size(-1)), labels.view(-1)
                )  # flatten the output and target tensors
                lossitem = loss.item()
                accum_loss += lossitem
                loss = loss / training_config.grad_accum_steps  # so grads are averaged
            scaler.scale(loss).backward()

            # grad accum
            if (global_step + 1) % training_config.grad_accum_steps == 0:
                if global_rank == 0:
                    print("Performing grad update")
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            if global_rank == 0:
                print(
                    "global_step:",
                    global_step,
                    "loss:",
                    lossitem,  # we want to see the actual loss! not the scaled one
                    "time:",
                    str(time.time() - start) + "s",
                    "batch_size:",
                    encoder_input.size(0),
                    "src len:",
                    encoder_input.size(1),
                    "tgt len: ",
                    decoder_input.size(1),
                    "lr: ",
                    scheduler.get_lr()[0],
                )

            if (global_step + 1) % training_config.eval_freq == 0:
                start = time.time()
                valid_loss = validate_model(
                    model, test_dl, device, ds_config, training_config, pad_id
                )
                model.train()
                avg_train_loss = accum_loss / training_config.eval_freq
                if global_rank == 0:
                    print(
                        "[VALIDATION FINISHED] global_step: ",
                        global_step,
                        "valid loss: ",
                        valid_loss,
                        "time taken :",
                        str(time.time() - start) + "s",
                    )
                    wandb.log(
                        {
                            "global_step": global_step,
                            "train/loss": avg_train_loss,
                            "eval/loss": valid_loss,
                            "lr": scheduler.get_lr()[0],
                        }
                    )
                accum_loss = 0
                # save each `training_config.save_freq` steps
            if (
                (global_step + 1) % training_config.save_freq
            ) == 0 and global_rank == 0:
                print("Saving checkpoint... global_step:", global_step)
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    model_config,
                    global_step,
                    training_config.checkpoint_dir,
                    training_config.checkpoint_save_filename,
                )
            global_step += 1
            if global_step >= training_config.max_global_steps:
                print("Training complete, global_step:", global_step)
                return
    
    # final save
    if global_rank == 0:
        print("Saving final checkpoint... global_step:", global_step)
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            model_config,
            global_step,
            training_config.checkpoint_dir,
            training_config.checkpoint_save_filename,
        )

@record
def main():
    #### Multi-GPU training ####
    init_process_group(backend="nccl")
    local_rank, global_rank, world_size = (
        int(os.environ["LOCAL_RANK"]),
        int(os.environ["RANK"]),
        int(os.environ["WORLD_SIZE"]),
    )
    torch.cuda.set_device(local_rank)
    ###############################################

    ds_config, model_config, training_config, parser = get_config_and_parser(
        update=True,
    )
    dsdict = retrieve_processed_dataset()
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    device = torch.device("cuda")
    print("============================INFO============================")
    print("Device:", device)
    print("Dataset config:", ds_config)
    print("Model config:", model_config)
    print("Training config:", training_config)
    print("Shared vocab size:", model_config.vocab_size)
    print("===========================DISTRIBUTED INFO============================")
    print("Local rank:", local_rank)
    print("Global rank:", global_rank)
    print("World size:", world_size)
    print("===========================================================")

    transformer = (
        Transformer.from_config(model_config).load_from_checkpoint(
            training_config.checkpoint_path
        )
    ).to(device)
    transformer = torch.nn.parallel.DistributedDataParallel(
        transformer,
        device_ids=[local_rank],
        output_device=local_rank,
    )
    tokenizer = get_tokenizer()
    pad_id = tokenizer.token_to_id(SpecialTokens.PAD.value)

    if global_rank == 0:
        tcfgdict = training_config.to_dict()
        t_cfg_log = {
            key: tcfgdict[key] for key in tcfgdict if not key.startswith("checkpoint")
        }
        wandb.init(
            name="run-0",
            project="attention-is-all-you-need",
            config={
                "ds": ds_config.to_dict(),
                "model": model_config.to_dict(),
                "training": t_cfg_log,
            },
        )
    train_transformer(
        transformer,
        dsdict,
        device,
        pad_id,
        model_config,
        training_config,
        ds_config,
        global_rank,
    )


if __name__ == "__main__":
    main()
