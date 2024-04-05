from tokenizers import Tokenizer
import torch
import torch.nn as nn
from config import ModelConfig, TrainingConfig
from typing import Tuple
from model import Transformer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader


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
) -> None:
    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=tokenizer.token_to_id("<pad>"),
        reduction="mean",
        label_smoothing=training_config.label_smoothing,
    )

    optimizer, scheduler = get_optim_and_scheduler(model, model_config, training_config)
    train_dl = DataLoader(
        train_dataset, batch_size=training_config.batch_size, shuffle=True
    )
    start_epoch = 0
    start_iteration = 0
    if training_config.checkpoint_path is not None:
        checkpoint = torch.load(training_config.checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"]
        start_iteration = checkpoint["iteration"] + 1
        dl_len = len(train_dl)
        if start_iteration >= dl_len - 1:
            start_iteration = 0
            start_epoch += 1

        print("Loaded checkpoint from", training_config.checkpoint_path)
        print(
            "info: saved epoch=",
            checkpoint["epoch"],
            "starting from epoch=",
            start_epoch,
            "iteration=",
            checkpoint["iteration"],
            "starting from iteration=",
            start_iteration,
        )
        print("info: lr=", optimizer.param_groups[0]["lr"])
        print("info: warmup_steps=", training_config.warmup_steps)

    is_first_epoch_in_session = True
    for epoch in range(start_epoch, training_config.max_epochs):
        model.train()
        for i, elem in enumerate(train_dl):
            if i < start_iteration and is_first_epoch_in_session:
                print("Skipping iteration", i, end="\r", flush=True)
                continue
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

            print(
                "iter:",
                i,
                " out of ",
                len(train_dl),
                " epoch:",
                epoch,
                " loss:",
                loss.item(),
            )
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # remember save_freq can be a decimal. For example, 10 means save 10 times per epoch
            if (i % int(len(train_dl) / training_config.save_freq) == 0) or (
                i == len(train_dl) - 1
            ):
                print("Saving checkpoint... Epoch:", epoch, " Iteration:", i)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch": epoch,
                        "model_config": model_config,  # does not occupy much space, but useful so we do not accidentally load a different model config
                        "iteration": i,
                    },
                    (
                        training_config.checkpoint_dir
                        / training_config.checkpoint_save_filename.format(epoch=epoch)
                    ),
                )

        is_first_epoch_in_session = False
