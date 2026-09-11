import torch
from torch import nn

from .modules import (
    AddAndNorm,
    Embedding,
    MultiHeadAttention,
    PositionalEncoding,
    PositionwiseFeedForward,
)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, d_kq: int, d_v: int, h: int, d_ff: int, p_drop: float = 0.1) -> None:
        super().__init__()
        self.mha = MultiHeadAttention(d_model=d_model, d_kq=d_kq, d_v=d_v, h=h)
        self.norm1 = AddAndNorm(d_model=d_model, p_drop=p_drop)

        self.ffnn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, p_drop=p_drop)
        self.norm2 = AddAndNorm(d_model=d_model, p_drop=p_drop)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # input (x) -> (batch, sequence_length, d_model)
        # input (mask): padding_mask -> (batch, 1, 1, sequence_length)
        # output -> (batch, sequence_length, d_model)

        sublayer_x = self.mha(x_q=x, x_kv=x, mask=mask)
        out = self.norm1(residual_x=x, sublayer_x=sublayer_x)

        sublayer_x = self.ffnn(x=out)
        out = self.norm2(residual_x=out, sublayer_x=sublayer_x)

        return out


class Encoder(nn.Module):
    def __init__(
        self,
        n_enc: int,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        d_kq: int,
        d_v: int,
        h: int,
        d_ff: int,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()
        self.emb = Embedding(vocab_size=vocab_size, d_model=d_model)
        self.pe = PositionalEncoding(max_seq_len=max_seq_len, d_model=d_model)
        self.dropout = nn.Dropout(p=p_drop)

        self.enc = nn.ModuleList(
            [EncoderLayer(d_model=d_model, d_kq=d_kq, d_v=d_v, h=h, d_ff=d_ff, p_drop=p_drop) for _ in range(n_enc)]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # input (x) -> (batch, sequence_length)
        # input (mask): padding_mask -> (batch, 1, 1, sequence_length)
        # output -> (batch, sequence_length, d_model)

        emb = self.emb(x=x)
        pe = self.pe(seq_len=x.size(1))
        out = self.dropout(input=emb + pe)

        for enc_layer in self.enc:
            out = enc_layer(x=out, mask=mask)

        return out
