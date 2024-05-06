from dataset import get_dataset
from model import Transformer
from config import get_config_and_parser
import torch
from tokenizers import Tokenizer
import torch.nn as nn
from config import ModelConfig, TrainingConfig
from typing import Tuple
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
import os
from torch.distributed import init_process_group
from torch.utils.data.distributed import DistributedSampler
import time


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


def train_transformer(
    model: Transformer,
    train_dataset: Dataset,
    device: torch.device,
    tokenizer: Tokenizer,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    global_rank: int = 0,
) -> None:
    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=tokenizer.token_to_id("<pad>"),
        reduction="mean",
        label_smoothing=training_config.label_smoothing,
    )

    optimizer, scheduler = get_optim_and_scheduler(model, model_config, training_config)
    train_dl = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,  # per GPU
        shuffle=False,
        sampler=DistributedSampler(train_dataset, shuffle=True),
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
    while True:
        model.train()
        for _, elem in enumerate(train_dl):
            start = time.time()
            encoder_input = elem["src"].to(device)
            decoder_input = elem["tgt_shifted"].to(device)
            labels = elem["tgt_labels"].to(device)
            src_mask = get_padding_mask(
                encoder_input, tokenizer.token_to_id("<pad>")
            ).to(device)
            tgt_mask = get_padding_mask(
                decoder_input, tokenizer.token_to_id("<pad>")
            ).to(device)

            out = model(encoder_input, decoder_input, src_mask, tgt_mask)
            loss = criterion(
                out.view(-1, out.size(-1)), labels.view(-1)
            )  # flatten the output and target tensors

            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if global_rank == 0:
                print(
                    "global_step:",
                    global_step,
                    "loss:",
                    loss.item(),
                )
                # save each `training_config.save_freq` steps
                if (global_step % (training_config.save_freq)) == 0:
                    print("Saving checkpoint... global_step:", global_step)
                    torch.save(
                        {
                            "model": (
                                model.module.state_dict()
                                if hasattr(model, "module")
                                else model.state_dict()
                            ),  # save the model depending on whether it is wrapped in a DataParallel or something
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "model_config": model_config,  # does not occupy much space, but useful so we do not accidentally load a different model config
                            "global_step": global_step,
                        },
                        (
                            training_config.checkpoint_dir
                            / training_config.checkpoint_save_filename.format(
                                global_step=global_step
                            )
                        ),
                    )
            global_step += 1
            if global_step >= training_config.max_global_steps:
                print("Training complete, global_step:", global_step)
                return


def main():
    #### Multi-GPU training ####
    # nccl only supported on linux
    init_process_group(backend="nccl" if os.name == "posix" else "gloo")
    local_rank, global_rank, world_size = (
        int(os.environ["LOCAL_RANK"]),
        int(os.environ["RANK"]),
        int(os.environ["WORLD_SIZE"]),
    )
    torch.cuda.set_device(local_rank)
    ###############################################

    ds_config, model_config, training_config, _, parser = get_config_and_parser(
        update=True,
        extra_args=[{"args": ["--nocompile"], "kwargs": {"action": "store_true"}}],
    )
    train_ds, valid_ds = get_dataset(ds_config, model_config)
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    device = torch.device("cuda")
    print("============================INFO============================")
    print("Device:", device)
    print("Dataset config:", ds_config)
    print("Model config:", model_config)
    print("Training config:", training_config)
    print("Src vocab size:", model_config.src_vocab_size)
    print("Tgt vocab size:", model_config.tgt_vocab_size)
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
    if not parser.parse_args().nocompile:
        transformer = torch.compile(transformer)

    train_transformer(
        transformer,
        train_ds,
        device,
        train_ds.src_tok,
        model_config,
        training_config,
        global_rank,
    )


if __name__ == "__main__":
    main()
