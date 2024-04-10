import pytest
from config import DatasetConfig
from dataset import get_dataset


class TestDataset:
    opus_books_config = DatasetConfig(
        ds_name="opus_books",
        src_lang="en",
        tgt_lang="es",
        split=0.9,
        seq_len=50,
    )

    wmt14_config = DatasetConfig(
        ds_name="wmt14",
        src_lang="en",
        tgt_lang="de",
        split=0.9,
        seq_len=50,
    )

    @pytest.mark.parametrize(
        "config",
        [
            opus_books_config,
            wmt14_config,
        ],
    )
    def test_get_dataset(self, config):
        train_ds, valid_ds = get_dataset(
            config, {"src_vocab_size": 100, "tgt_vocab_size": 100}
        )
        assert len(train_ds) + len(valid_ds) == len(train_ds.raw_ds)
        assert len(train_ds) == int(config.split * len(train_ds.raw_ds))
        assert len(valid_ds) == len(train_ds.raw_ds) - len(train_ds)
        assert train_ds.src_lang == config.src_lang
        assert train_ds.tgt_lang == config.tgt_lang
        assert train_ds.seq_len == config.seq_len
        assert valid_ds.src_lang == config.src_lang
        assert valid_ds.tgt_lang == config.tgt_lang
        assert valid_ds.seq_len == config.seq_len
        assert train_ds.src_tok.get_vocab_size() > 0
        assert train_ds.tgt_tok.get_vocab_size() > 0
        assert valid_ds.src_tok.get_vocab_size() > 0
        assert valid_ds.tgt_tok.get_vocab_size() > 0
        assert train_ds.src_tok.get_vocab_size() == valid_ds.src_tok.get_vocab_size()
        assert train_ds.tgt_tok.get_vocab_size() == valid_ds.tgt_tok.get_vocab_size()
