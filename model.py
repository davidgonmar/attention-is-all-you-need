import torch
from torch import Tensor
from torch.nn import functional as F
from torch import nn

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
    def __init__(self, d_model: int, seq_len: int, dropout: float = 0) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pos_encoding = torch.zeros(seq_len, d_model)  # (seq_len, d_model)
        pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.pow(10000, torch.arange(0, d_model, 2) / d_model)

        pos_encoding[:, 0::2] = torch.sin(pos / div_term)
        pos_encoding[:, 1::2] = torch.cos(pos / div_term)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, x: Tensor) -> Tensor:
        # x is of shape (batch_size, seq_len, d_model)
        # so we want only to add the positional encoding to the seq_len dimension
        cropped_pos_encoding = self.pos_encoding[: x.size(1), :]  # (seq_len, d_model)
        x = x + cropped_pos_encoding
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(
        self, num_heads: int, d_model: int, d_k: int, d_v: int, mask: bool = False
    ):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.causal = mask
        self.d_k = d_k
        self.d_v = d_v

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

    def _scaled_dot_product_attention(self, Q: Tensor, K: Tensor, V: Tensor, pad_attn_mask) -> Tensor:
        """
        This is equivalent to separately computing the attention for each head.
        Args:
            Q: Query matrix with shape (batch_size, num_heads, len_q, d_k)
            K: Key matrix with shape (batch_size, num_heads, len_k, d_k)
            V: Value matrix with shape (batch_size, num_heads, len_v = len_k, d_v)
        """
        x = (
            Q @ K.transpose(-2, -1)
        ) / self.d_k**0.5  # (batch_size, num_heads, len_q, len_k)
        # len_v = len_k !!!
        mask = pad_attn_mask
        if self.causal:
            # print("before masking: ", x)
            # Apply masking, will be broadcasted to shape (batch_size, num_heads, len_q, len_k)
            # Basically, create a matrix with 1s below and in the diagonal, 0s above
            # Then, mask where mask == 0 with -inf
            # So basically we set the values above the diagonal to -inf
            # When softmax is applied, these values will become 0
            causal_mask = (
                torch.tril(torch.ones(x.size(-2), x.size(-1)), diagonal=0)
                .view(1, 1, x.size(-2), x.size(-1))
                .to(x.device) # (1, 1, len_q, len_k)
            )
            mask = mask.bool() & causal_mask.bool() if mask is not None else causal_mask.bool()
            x = x.masked_fill(mask == 0, -1e9) # (batch_size, num_heads, len_q, len_k)
            # show the attention matrix, black and white (black is 0, white is 1)
            """softmaxed = F.softmax(x, dim=-1)
            plt.figure(figsize=(20, 20))
            plt.imshow(softmaxed[0, 0].detach().cpu().numpy(), cmap="gray")
            plt.show()
            """ 

        else:
            if mask is not None:
                """mask = mask.broadcast_to(x.shape)
                import matplotlib.pyplot as plt
                # show the mask
                plt.figure(figsize=(20, 20))
                plt.imshow(mask[0, 0].detach().cpu().numpy(), cmap="gray")
                plt.show()
                x = x.masked_fill(mask == 0, -1e9)"""
                x = x.masked_fill(mask == 0, -1e9) # (batch_size, num_heads, len_q, len_k)
        # print("after masking: ", x)
        return F.softmax(x, dim=-1) @ V  # (batch_size, num_heads, len_q, d_v)


class EncoderLayer(nn.Module):
    def __init__(self, num_heads: int, d_model: int, d_k: int, d_v: int, d_ff: int):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(
            num_heads=num_heads, d_model=d_model, d_k=d_k, d_v=d_v
        )
        self.position_wise_feed_forward = PositionWiseFeedForward(
            d_model=d_model, inner_dim=d_ff
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, input: Tensor, pad_attn_mask: Tensor) -> Tensor:
        out1 = self.multi_head_attention(input, input, input, pad_attn_mask)
        out1 += input  # residual connection
        out1 = self.layer_norm1(out1)

        out2 = self.position_wise_feed_forward(out1)
        out2 += out1  # residual connection
        out2 = self.layer_norm2(out2)

        return out2


class Encoder(nn.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        d_k: int,
        d_v: int,
        d_ff: int,
        num_layers: int = 6,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    num_heads=num_heads, d_model=d_model, d_k=d_k, d_v=d_v, d_ff=d_ff
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, input: Tensor, pad_attn_mask: Tensor) -> Tensor:
        for layer in self.layers:
            input = layer(input, pad_attn_mask)
        return input


class DecoderLayer(nn.Module):
    def __init__(self, num_heads: int, d_model: int, d_k: int, d_v: int, d_ff: int):
        super().__init__()
        self.masked_multi_head_attention = MultiHeadAttention(
            num_heads=num_heads, d_model=d_model, d_k=d_k, d_v=d_v, mask=True
        )
        self.multi_head_attention = MultiHeadAttention(
            num_heads=num_heads, d_model=d_model, d_k=d_k, d_v=d_v, mask=False
        )
        self.position_wise_feed_forward = PositionWiseFeedForward(
            d_model=d_model, inner_dim=d_ff
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)

    def forward(self, encoder_output: Tensor, input: Tensor, pad_attn_mask_src: Tensor, pad_attn_mask_tgt: Tensor) -> Tensor:
        out1 = self.masked_multi_head_attention(input, input, input, pad_attn_mask_tgt)
        out1 += input  # residual connection
        out1 = self.layer_norm1(out1)

        out2 = self.multi_head_attention(input, encoder_output, encoder_output, pad_attn_mask_src) # we use pad_attn_mask_src because we want to mask the padding in the encoder output
        out2 += out1  # residual connection
        out2 = self.layer_norm1(out2)

        out3 = self.position_wise_feed_forward(out2)
        out3 += out2  # residual connection
        out3 = self.layer_norm2(out3)

        return out3


class Decoder(nn.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        d_k: int,
        d_v: int,
        d_ff: int,
        num_layers: int = 6,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    num_heads=num_heads, d_model=d_model, d_k=d_k, d_v=d_v, d_ff=d_ff
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, encoder_output: Tensor, input: Tensor, pad_attn_mask_src: Tensor, pad_attn_mask_tgt: Tensor) -> Tensor:
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
    ):
        super().__init__()
        self.encoder = Encoder(
            num_heads=num_heads, d_model=d_model, d_k=d_k, d_v=d_v, d_ff=d_ff
        )
        self.decoder = Decoder(
            num_heads=num_heads, d_model=d_model, d_k=d_k, d_v=d_v, d_ff=d_ff
        )
        self.input_embedder = InputEmbedder(d_model=d_model, vocab_size=src_vocab_size)
        self.positional_encoder = PositionalEncoding(d_model=d_model, seq_len=1000)
        self.output_embedder = InputEmbedder(d_model=d_model, vocab_size=tgt_vocab_size)
        self.positional_decoder = PositionalEncoding(d_model=d_model, seq_len=1000)
        self.linear = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src: Tensor, tgt: Tensor, src_pad_attn_mask: Tensor, tgt_pad_attn_mask: Tensor) -> Tensor:
        src = self.input_embedder(src)
        src = self.positional_encoder(src)

        encoder_output = self.encoder(src, src_pad_attn_mask)

        tgt = self.output_embedder(tgt)
        tgt = self.positional_decoder(tgt)

        decoder_output = self.decoder(encoder_output, tgt, src_pad_attn_mask, tgt_pad_attn_mask)

        decoder_output = self.linear(decoder_output)
        
        return decoder_output


"""multi_head_attention = MultiHeadAttention(num_heads=8, d_model=512, d_k=64, d_v=64)


q = torch.randn(64, 10, 512)
k = torch.randn(64, 10, 512)
v = torch.randn(64, 10, 512)

multi_head_attention(q, k, v)


# now with different sequence length
q = torch.randn(64, 20, 512)
k = torch.randn(64, 10, 512)
v = torch.randn(64, 10, 512)

multi_head_attention(q, k, v)


encoder = Encoder(num_heads=8, d_model=512, d_k=64, d_v=64, d_ff=2048)


x = torch.randn(64, 10, 512)

encoder_output = encoder(x)

print(encoder_output.shape)

decoder = Decoder(num_heads=8, d_model=512, d_k=64, d_v=64, d_ff=2048)

decoder(encoder_output, x)

print(encoder_output.shape)


transformer = Transformer(
    num_heads=8,
    d_model=512,
    d_k=64,
    d_v=64,
    d_ff=2048,
    src_vocab_size=1000,
    tgt_vocab_size=1000,
)

src = torch.randint(0, 1000, (64, 10))
tgt = torch.randint(0, 1000, (64, 10))

x = transformer(src, tgt)

print(x.shape)

x.sum().backward()


# loop to generate a sequence
input = torch.randint(0, 1000, (1, 1))

for i in range(10):
    output = transformer(input, input)
    output = output.argmax(dim=-1)  # (1, 1)
    input = torch.cat([input, output[:, -1:]], dim=-1)
"""
