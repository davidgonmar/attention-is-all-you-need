# Attention is all you need implementation in PyTorch

This is a PyTorch implementation of the the paper [Attention is All You Need](https://arxiv.org/abs/1706.03762).

## Requirements

The requirements are listed in the `requirements.txt` file. Other versions will probably work, but these are the ones I used.

## Configuration

All configurable parameters are in the `config.py` file. To override a default value, pass in an argument to the script, for example:

```bash
python train.py --max_epochs 10
```

```python
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
    save_info_in_filename: bool = False # Save the epoch and iteration in the checkpoint
    label_smoothing: float = 0.1
```

## Usage

### Training

To train the model, run the `train.py` script. The script will download the dataset, preprocess it, and train the model. The model will be saved in the `checkpoints` directory.

```
torchrun --nproc_per_node 1 train.py --config configs/laptop_wmt14.yaml
torchrun --nproc_per_node 4 train.py --config configs/distrib_wmt14.yaml
```

### Inference

To generate translations, run the `translate.py` script. The script will load the model from the `checkpoints` directory and generate translations for the test set. It is a simple script where the user can input a sentence in the source language and get the translation in the target language.

## Citations

```bibtex
@misc{vaswani2023attention,
      title={Attention Is All You Need},
      author={Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin},
      year={2023},
      eprint={1706.03762},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

### Usage

1. Train the tokenizer

```bash
    python train_tokenizer.py --config configs/distrib_wmt14.yaml
```

2. Preprocess the dataset (pretokenize it)

```bash
    python preprocess_data.py --config configs/distrib_wmt14.yaml
```

3. Run the training script

```bash
    torchrun --nproc_per_node 4 train.py --config configs/distrib_wmt14.yaml
```
