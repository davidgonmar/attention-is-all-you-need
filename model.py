from typing import Optional
import torch
from torch import Tensor
from torch.nn import functional as F
from torch import nn

from config import ModelConfig

USE_TORCH_SDPA = True


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, inner_dim: int):
        """
        Args:
            d_model: dimensionality of the input and output
            inner_dim: dimensionality of the inner layer, also called d_ff in the paper
        """
        super().__init__()
        self.fc1 = nn.Linear(d_model, inner_dim)
        self.fc2 = nn.Linear(inner_dim, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class InputEmbedder(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x) * self.d_model**0.5


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        pos_encoding = torch.zeros(seq_len, d_model)  # (seq_len, d_model)
        pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.pow(10000, torch.arange(0, d_model, 2) / d_model)

        pos_encoding[:, 0::2] = torch.sin(pos / div_term)
        pos_encoding[:, 1::2] = torch.cos(pos / div_term)
        self.register_buffer("pos_encoding", pos_encoding)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x is of shape (batch_size, seq_len, d_model)
        # so we want only to add the positional encoding to the seq_len dimension
        cropped_pos_encoding = self.pos_encoding[: x.size(1), :]  # (seq_len, d_model)
        return self.dropout(x + cropped_pos_encoding)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        d_k: int,
        d_v: int,
        causal: bool,
        save_attn_scores_to_visualize=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.causal = causal
        self.d_k = d_k
        self.d_v = d_v
        self.save_attn_scores_to_visualize = save_attn_scores_to_visualize
        self.attn_scores = None

        assert d_model % num_heads == 0, "d_model should be divisible by num_heads"

        self.W_q = nn.Linear(d_model, num_heads * d_k, bias=False)
        self.W_k = nn.Linear(d_model, num_heads * d_k, bias=False)
        self.W_v = nn.Linear(d_model, num_heads * d_v, bias=False)
        self.W_o = nn.Linear(num_heads * d_v, d_model, bias=False)

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, pad_attn_mask: Tensor = None):
        """
        len_q, len_k, len_v are the lengths of the sequences for Q, K, V
        Args:
            Q: Query matrix with shape (batch_size, len_q, d_model)
            K: Key matrix with shape (batch_size, len_k, d_model)
            V: Value matrix with shape (batch_size, len_v, d_model)
        """

        len_q, len_k, len_v = Q.size(1), K.size(1), V.size(1)
        assert len_k == len_v, "len_k and len_v must be equal, got {} and {}".format(
            len_k, len_v
        )
        batch_size = Q.size(0)  # should be equal to K.size(0) and V.size(0)

        # Project query, key and value into d_k * num_heads and d_v * num_heads
        # We transpose them so the 'inner (right-most) matrices' are of shape
        # (len_x, d_x), so shape is (batch_size, num_heads, len_x, d_x)
        Q = (
            self.W_q(Q)
            .view(batch_size, len_q, self.num_heads, self.d_k)
            .transpose(1, 2)
        )  # shape (batch_size, num_heads, len_q, d_k)
        K = (
            self.W_k(K)
            .view(batch_size, len_k, self.num_heads, self.d_k)
            .transpose(1, 2)
        )  # shape (batch_size, num_heads, len_k, d_k)
        V = (
            self.W_v(V)
            .view(batch_size, len_v, self.num_heads, self.d_v)
            .transpose(1, 2)
        )  # shape (batch_size, num_heads, len_v, d_v)

        out = self._scaled_dot_product_attention(
            Q, K, V, pad_attn_mask
        )  # (n, num_heads, seq_len, d_v)

        # now, we need to multiply by the linear with input size num_heads * d_v
        out = out.transpose(1, 2)  # shape (batch_size, len_q, num_heads, d_v)

        assert out.size(2) == self.num_heads
        assert out.size(3) == self.d_v
        assert out.size(1) == len_q
        assert out.size(0) == batch_size

        # We then merge the heads together to get (batch_size, len_q, num_heads * d_v)
        # In the paper, num_heads * d_v = d_model
        # Dont use view because memory layout is not compatible
        out = out.reshape(batch_size, len_q, self.num_heads * self.d_v)
        return self.W_o(out)

    def _scaled_dot_product_attention(
        self, Q: Tensor, K: Tensor, V: Tensor, pad_attn_mask: Optional[Tensor]
    ) -> Tensor:
        """
        This is equivalent to separately computing the attention for each head.
        Args:
            Q: Query matrix with shape (batch_size, num_heads, len_q, d_k)
            K: Key matrix with shape (batch_size, num_heads, len_k, d_k)
            V: Value matrix with shape (batch_size, num_heads, len_v = len_k, d_v)
        """

        mask = pad_attn_mask
        if self.causal:
            causal_mask = (
                torch.tril(torch.ones(Q.size(-2), K.size(-2)), diagonal=0)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(Q.device)  # (1, 1, len_q, len_k)
            )
            mask = (
                mask.bool() & causal_mask.bool()
                if mask is not None
                else causal_mask.bool()
            )
        else:
            if mask is not None:
                mask = mask.bool()

        if USE_TORCH_SDPA:
            # manually pass the 'fused' mask
            return torch.nn.functional.scaled_dot_product_attention(
                Q,
                K,
                V,
                mask,
                dropout_p=0.0,
                is_causal=False,
                scale=1.0 / self.d_k**0.5,
            )

        x = (Q @ K.transpose(-2, -1)) / (
            self.d_k**0.5
        )  # (batch_size, num_heads, len_q, len_k)
        # len_v = len_k !!!
        if mask is not None:
            x = x.masked_fill(mask == 0, float("-inf"))
        x = F.softmax(x, dim=-1)  # (batch_size, num_heads, len_q, len_k)
        if self.save_attn_scores_to_visualize:
            self.attn_scores = x
        return x @ V


class EncoderLayer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        d_k: int,
        d_v: int,
        d_ff: int,
        dropout: float,
    ):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(
            num_heads=num_heads, d_model=d_model, d_k=d_k, d_v=d_v, causal=False
        )
        self.position_wise_feed_forward = PositionWiseFeedForward(
            d_model=d_model, inner_dim=d_ff
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input: Tensor, pad_attn_mask: Tensor) -> Tensor:
        out1 = self.multi_head_attention(input, input, input, pad_attn_mask)
        out1 = self.layer_norm1(self.dropout(out1) + input)  # residual connection

        out2 = self.position_wise_feed_forward(out1)
        out2 = self.layer_norm2(self.dropout(out2) + out1)  # residual connection

        return out2


class Encoder(nn.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        d_k: int,
        d_v: int,
        d_ff: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    num_heads=num_heads,
                    d_model=d_model,
                    d_k=d_k,
                    d_v=d_v,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, input: Tensor, pad_attn_mask: Tensor) -> Tensor:
        for layer in self.layers:
            input = layer(input, pad_attn_mask)
        return input


class DecoderLayer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        d_k: int,
        d_v: int,
        d_ff: int,
        dropout: float,
    ):
        super().__init__()
        self.masked_multi_head_attention = MultiHeadAttention(
            num_heads=num_heads, d_model=d_model, d_k=d_k, d_v=d_v, causal=True
        )
        self.multi_head_attention = MultiHeadAttention(
            num_heads=num_heads, d_model=d_model, d_k=d_k, d_v=d_v, causal=False
        )
        self.position_wise_feed_forward = PositionWiseFeedForward(
            d_model=d_model, inner_dim=d_ff
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        encoder_output: Tensor,
        input: Tensor,
        pad_attn_mask_src: Tensor,
        pad_attn_mask_tgt: Tensor,
    ) -> Tensor:
        out1 = self.masked_multi_head_attention(input, input, input, pad_attn_mask_tgt)
        out1 = self.layer_norm1(self.dropout(out1) + input)  # residual connection

        out2 = self.multi_head_attention(
            input, encoder_output, encoder_output, pad_attn_mask_src
        )  # we use pad_attn_mask_src because we want to mask the padding in the encoder output
        out2 = self.layer_norm2(self.dropout(out2) + out1)  # residual connection

        out3 = self.position_wise_feed_forward(out2)
        out3 = self.layer_norm3(self.dropout(out3) + out2)  # residual connection

        return out3


class Decoder(nn.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        d_k: int,
        d_v: int,
        d_ff: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    num_heads=num_heads,
                    d_model=d_model,
                    d_k=d_k,
                    d_v=d_v,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        encoder_output: Tensor,
        input: Tensor,
        pad_attn_mask_src: Tensor,
        pad_attn_mask_tgt: Tensor,
    ) -> Tensor:
        for layer in self.layers:
            input = layer(encoder_output, input, pad_attn_mask_src, pad_attn_mask_tgt)
        return input


class Transformer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        d_k: int,
        d_v: int,
        d_ff: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        dropout: float,
        n_encoder_layers: int,
        n_decoder_layers: int,
    ):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.config = ModelConfig(
            num_heads=num_heads,
            d_model=d_model,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            n_encoder_layers=n_encoder_layers,
            n_decoder_layers=n_decoder_layers,
            dropout=dropout,
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
        )
        self.encoder = Encoder(
            num_heads=num_heads,
            d_model=d_model,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            dropout=dropout,
            num_layers=n_encoder_layers,
        )
        self.decoder = Decoder(
            num_heads=num_heads,
            d_model=d_model,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            dropout=dropout,
            num_layers=n_decoder_layers,
        )
        self.input_embedder = InputEmbedder(d_model=d_model, vocab_size=src_vocab_size)
        self.positional_encoder = PositionalEncoding(
            d_model=d_model, seq_len=3000, dropout=dropout
        )
        self.output_embedder = InputEmbedder(d_model=d_model, vocab_size=tgt_vocab_size)
        self.positional_decoder = PositionalEncoding(
            d_model=d_model, seq_len=3000, dropout=dropout
        )
        self.linear = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_pad_attn_mask: Tensor,
        tgt_pad_attn_mask: Tensor,
    ) -> Tensor:
        src = self.input_embedder(src)
        src = self.positional_encoder(src)
        src = self.dropout(src)

        encoder_output = self.encoder(src, src_pad_attn_mask)

        tgt = self.output_embedder(tgt)
        tgt = self.positional_decoder(tgt)
        tgt = self.dropout(tgt)

        decoder_output = self.decoder(
            encoder_output, tgt, src_pad_attn_mask, tgt_pad_attn_mask
        )

        decoder_output = self.linear(decoder_output)

        return decoder_output

    @staticmethod
    def from_config(
        config: ModelConfig,
    ) -> "Transformer":
        return Transformer(
            num_heads=config.num_heads,
            d_model=config.d_model,
            d_k=config.d_k,
            d_v=config.d_v,
            d_ff=config.d_ff,
            src_vocab_size=config.src_vocab_size,
            tgt_vocab_size=config.tgt_vocab_size,
            dropout=config.dropout,
            n_encoder_layers=config.n_encoder_layers,
            n_decoder_layers=config.n_decoder_layers,
        )

    def load_from_checkpoint(self, checkpoint_path: str) -> "Transformer":
        if not checkpoint_path:
            print("No checkpoint path provided, starting from scratch")
            return self
        try:
            checkpoint = torch.load(checkpoint_path)
            if checkpoint["model_config"] != self.config:
                raise ValueError(
                    "Model config in checkpoint does not match the current model config. Checkpoint: {}, Model: {}".format(
                        checkpoint["model_config"], self.config
                    )
                )
            self.load_state_dict(checkpoint["model"])
            print("Loaded model from", checkpoint_path)
        except FileNotFoundError:
            print("Model not found at", checkpoint_path, "Starting from scratch")
        return self
