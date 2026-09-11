import torch
from torch import nn

from .modules import (
    AddAndNorm,
    Embedding,
    MultiHeadAttention,
    PositionalEncoding,
    PositionwiseFeedForward,
)


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, d_kq: int, d_v: int, h: int, d_ff: int, p_drop: float = 0.1) -> None:
        super().__init__()

        self.masked_mha = MultiHeadAttention(d_model=d_model, d_kq=d_kq, d_v=d_v, h=h)
        self.norm1 = AddAndNorm(d_model=d_model, p_drop=p_drop)

        self.cross_mha = MultiHeadAttention(d_model=d_model, d_kq=d_kq, d_v=d_v, h=h)
        self.norm2 = AddAndNorm(d_model=d_model, p_drop=p_drop)

        self.ffnn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, p_drop=p_drop)
        self.norm3 = AddAndNorm(d_model=d_model, p_drop=p_drop)

    def forward(
        self,
        x: torch.Tensor,
        x_enc: torch.Tensor,
        self_mask: torch.Tensor | None = None,
        cross_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sublayer_x = self.masked_mha(x_q=x, x_kv=x, mask=self_mask)
        out = self.norm1(residual_x=x, sublayer_x=sublayer_x)

        sublayer_x = self.cross_mha(x_q=out, x_kv=x_enc, mask=cross_mask)
        out = self.norm2(residual_x=out, sublayer_x=sublayer_x)

        sublayer_x = self.ffnn(x=out)
        out = self.norm3(residual_x=out, sublayer_x=sublayer_x)

        return out


class Decoder(nn.Module):
    def __init__(
        self,
        n_dec: int,
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

        self.dec = nn.ModuleList(
            [DecoderLayer(d_model=d_model, d_kq=d_kq, d_v=d_v, h=h, d_ff=d_ff, p_drop=p_drop) for _ in range(n_dec)]
        )

    def forward(
        self,
        x: torch.Tensor,
        x_enc: torch.Tensor,
        self_mask: torch.Tensor | None = None,
        cross_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        emb = self.emb(x)
        pe = self.pe(x.size(1))
        out = self.dropout(emb + pe)

        for dec_layer in self.dec:
            out = dec_layer(out, x_enc, self_mask=self_mask, cross_mask=cross_mask)

        return out
