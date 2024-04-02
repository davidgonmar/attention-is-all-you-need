from argparse import ArgumentParser
from typing import List, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

CHECKPINTS_DIR = Path("checkpoints")


def _get_latest_checkpoint_path() -> Optional[Path]:
    dir = CHECKPINTS_DIR
    if not dir.exists():
        raise FileNotFoundError("No checkpoints directory found")

    checkpoints = list(dir.glob("*.pth"))

    if not checkpoints:
        return None

    return max(checkpoints, key=lambda x: x.stat().st_ctime)

class BaseConfig:
    def __repr__(self):
        return f"{self.__class__.__name__}(" + ", ".join(
            [f"{k}={v}" for k, v in self.__dict__.items()]) + ")"

    def add_to_arg_parser(self, parser: ArgumentParser) -> None:
        for k, v in self.__dict__.items():
            if not isinstance(v, bool):
                parser.add_argument(f"--{k}", type=type(v), default=v)
            else:
                def parse_bool(x):
                    return x.lower() in ["true", "1", "yes"]
                parser.add_argument(f"--{k}", type=parse_bool, default=v)

    def update_from_arg_parser(self, args: List[Any]) -> None  :
        for k, v in self.__dict__.items():
            setattr(self, k, getattr(args, k) if hasattr(args, k) else v)
        
@dataclass
class DatasetConfig(BaseConfig):
    ds_name: str = "opus_books"
    src_lang: str = "en"
    tgt_lang: str = "es"
    split: int = 0.9
    seq_len: int = 400
    
@dataclass
class ModelConfig(BaseConfig):
    num_heads: int = 8
    d_model: int = 512
    d_k: int = 64
    d_v: int = 64
    d_ff: int = 2048

@dataclass
class TrainingConfig(BaseConfig):
    max_epochs: int = 100
    batch_size: int = 12
    lr: float = 1.0 # Learning rate base, will be scaled by scheduler
    use_scheduler: bool = True
    b1: float = 0.9
    b2: float = 0.98
    eps: float = 1e-9
    warmup_steps: int = 4000
    checkpoint_path : Optional[Path] = _get_latest_checkpoint_path()
    save_freq: int = 1 # once per epoch
    save_info: bool = False # Save the epoch and iteration in the checkpoint
    label_smoothing: float = 0.1

def get_config_and_parser(existing_parser: Optional[ArgumentParser] = None, update: bool = True) -> Tuple[DatasetConfig, ModelConfig, TrainingConfig, ArgumentParser]:
    parser = existing_parser or ArgumentParser()
    ds_config = DatasetConfig()
    model_config = ModelConfig()
    training_config = TrainingConfig()

    ds_config.add_to_arg_parser(parser)
    model_config.add_to_arg_parser(parser)
    training_config.add_to_arg_parser(parser)

    if update:
        ds_config.update_from_arg_parser(parser.parse_args())
        model_config.update_from_arg_parser(parser.parse_args())
        training_config.update_from_arg_parser(parser.parse_args())

    return ds_config, model_config, training_config, parser