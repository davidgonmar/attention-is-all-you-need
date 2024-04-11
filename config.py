from argparse import ArgumentParser
from typing import List, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import yaml


def configs_from_yaml(
    yaml_path: Path,
) -> Tuple[dict, dict, dict]:
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    checkpoint_dir = config["training"].get("checkpoint_dir")
    assert checkpoint_dir is not None, "checkpoint_dir must be provided in the yaml"
    # If no checkpoint path is provided in the yaml, get the latest checkpoint
    if (
        "checkpoint_filename" not in config["training"]
        or config["training"]["checkpoint_filename"] is None
        or config["training"]["checkpoint_filename"] == "latest"
    ):
        config["training"]["checkpoint_filename"] = _get_latest_checkpoint_path(
            checkpoint_dir
        )
    if config["training"]["checkpoint_filename"] == "dont_use":
        config["training"]["checkpoint_filename"] = None

    return config["dataset"], config["model"], config["training"], config["eval"]


def _get_latest_checkpoint_path(base_path: str) -> Optional[Path]:
    dir = Path(base_path)
    if not dir.exists():
        raise FileNotFoundError("No checkpoints directory found")

    checkpoints = list(dir.glob("*.pth"))

    if not checkpoints:
        return None

    complete = max(checkpoints, key=lambda x: x.stat().st_ctime)

    # only return the filename
    return complete.name


class BaseConfig:
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            + ", ".join([f"{k}={v}" for k, v in self.__dict__.items()])
            + ")"
        )

    def add_to_arg_parser(self, parser: ArgumentParser) -> None:
        for k, v in self.__dict__.items():
            if not isinstance(v, bool):
                parser.add_argument(f"--{k}", type=type(v))
            else:

                def parse_bool(x):
                    return x.lower() in ["true", "1", "yes"]

                parser.add_argument(f"--{k}", type=parse_bool)

    def update_from_arg_parser(self, args: List[Any]) -> None:
        for k, v in self.__dict__.items():
            if hasattr(args, k) and getattr(args, k) is not None:
                setattr(self, k, getattr(args, k))

    def load_from_dict(self, d: dict) -> None:
        for k, v in d.items():
            setattr(self, k, v)


@dataclass
class DatasetConfig(BaseConfig):
    ds_name: str = "opus_books"
    src_lang: str = "en"
    tgt_lang: str = "es"
    split: int = 0.9  # ignored if dataset has train and test splits already
    seq_len: int = 400


@dataclass
class ModelConfig(BaseConfig):
    num_heads: int = 8
    d_model: int = 512
    d_k: int = 64
    d_v: int = 64
    d_ff: int = 2048
    dropout: float = 0.1
    n_encoder_layers: int = 6
    n_decoder_layers: int = 6
    src_vocab_size: int = 10000
    tgt_vocab_size: int = 10000


@dataclass
class TrainingConfig(BaseConfig):
    max_epochs: int = 100
    batch_size: int = 12
    lr: float = 1.0  # Learning rate base, will be scaled by scheduler
    use_scheduler: bool = True
    b1: float = 0.9
    b2: float = 0.98
    eps: float = 1e-9
    warmup_steps: int = 4000
    checkpoint_dir: Path = Path("checkpoints")
    checkpoint_filename: Optional[str] = None
    checkpoint_save_filename: str = "checkpoint_{epoch}.pth"
    save_freq: float = 1.0  # once per epoch by default
    label_smoothing: float = 0.1

    @property
    def checkpoint_path(self) -> Path:
        self.checkpoint_dir = Path(self.checkpoint_dir)
        return (
            self.checkpoint_dir / self.checkpoint_filename
            if self.checkpoint_filename is not None
            else None
        )


@dataclass
class EvalConfig(BaseConfig):
    checkpoint_dir: Path = Path("checkpoints")
    checkpoint_filename: Optional[str] = None
    batch_size: int = 12

    @property
    def checkpoint_path(self) -> Path:
        self.checkpoint_dir = Path(self.checkpoint_dir)
        return (
            self.checkpoint_dir / self.checkpoint_filename
            if self.checkpoint_filename is not None
            else None
        )


def get_config_no_parser(config_path: Optional[Path] = None):
    ds_config = DatasetConfig()
    model_config = ModelConfig()
    training_config = TrainingConfig()
    eval_config = EvalConfig()

    if config_path is not None:
        (
            ds_config_dict,
            model_config_dict,
            training_config_dict,
            eval_config_dict,
        ) = configs_from_yaml(config_path)
        ds_config.load_from_dict(ds_config_dict)
        model_config.load_from_dict(model_config_dict)
        training_config.load_from_dict(training_config_dict)
        eval_config.load_from_dict(eval_config_dict)

    return ds_config, model_config, training_config, eval_config


def get_config_and_parser(
    existing_parser: Optional[ArgumentParser] = None, update: bool = True
) -> Tuple[DatasetConfig, ModelConfig, TrainingConfig, ArgumentParser]:
    parser = existing_parser or ArgumentParser(conflict_handler="resolve")
    parser.add_argument(
        "--config", type=Path, default=None
    )  # allows default config file to be passed (will be overridden)
    ds_config = DatasetConfig()
    model_config = ModelConfig()
    training_config = TrainingConfig()
    eval_config = EvalConfig()
    ds_config.add_to_arg_parser(parser)
    model_config.add_to_arg_parser(parser)
    training_config.add_to_arg_parser(parser)
    eval_config.add_to_arg_parser(parser)

    args = parser.parse_args()

    # load configs from yaml if provided
    if args.config is not None:
        print(f"Loading configs from {args.config}")
        (
            ds_config_dict,
            model_config_dict,
            training_config_dict,
            eval_config_dict,
        ) = configs_from_yaml(args.config)
        ds_config.load_from_dict(ds_config_dict)
        model_config.load_from_dict(model_config_dict)
        training_config.load_from_dict(training_config_dict)
        eval_config.load_from_dict(eval_config_dict)

    # now update configs to get overrides from command line
    ds_config.update_from_arg_parser(args)
    model_config.update_from_arg_parser(args)
    training_config.update_from_arg_parser(args)
    eval_config.update_from_arg_parser(args)

    return ds_config, model_config, training_config, eval_config, parser
