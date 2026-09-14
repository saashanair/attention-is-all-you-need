import torch
from torch import nn

from .config import TransformerConfig
from .decoder import Decoder
from .encoder import Encoder


class Transformer(nn.Module):
    causal_mask: torch.Tensor  # declared for type checkers -- register_buffer() alone types this as a generic Module

    def __init__(
        self, src_vocab_size: int, tgt_vocab_size: int, shared_embeddings: bool, cfg: TransformerConfig
    ) -> None:
        super().__init__()

        if shared_embeddings and src_vocab_size != tgt_vocab_size:
            raise ValueError('src_vocab_size and tgt_vocab_size must match when shared_embeddings=True')

        self.encoder = Encoder(
            n_enc=cfg.n_enc,
            vocab_size=src_vocab_size,
            max_seq_len=cfg.max_seq_len,
            d_model=cfg.d_model,
            d_kq=cfg.d_kq,
            d_v=cfg.d_v,
            h=cfg.h,
            d_ff=cfg.d_ff,
            p_drop=cfg.p_drop,
        )
        self.decoder = Decoder(
            n_dec=cfg.n_dec,
            vocab_size=tgt_vocab_size,
            max_seq_len=cfg.max_seq_len,
            d_model=cfg.d_model,
            d_kq=cfg.d_kq,
            d_v=cfg.d_v,
            h=cfg.h,
            d_ff=cfg.d_ff,
            p_drop=cfg.p_drop,
        )

        self.linear = nn.Linear(in_features=cfg.d_model, out_features=tgt_vocab_size)

        if shared_embeddings:
            self.decoder.emb.emb.weight = self.encoder.emb.emb.weight
        self.linear.weight = self.decoder.emb.emb.weight

        causal_mask = torch.triu(torch.ones((cfg.max_seq_len, cfg.max_seq_len), dtype=torch.bool), diagonal=1)
        self.register_buffer('causal_mask', causal_mask, persistent=False)

    def _compute_causal_mask(self, mask_size: int):
        return self.causal_mask[:mask_size, :mask_size]

    def encode(self, x, mask=None):
        return self.encoder(x=x, mask=mask)

    def decode(
        self,
        x: torch.Tensor,
        x_enc: torch.Tensor,
        self_mask: torch.Tensor | None = None,
        cross_mask: torch.Tensor | None = None,
    ):
        return self.decoder(x=x, x_enc=x_enc, self_mask=self_mask, cross_mask=cross_mask)

    def forward(
        self, src: torch.Tensor, tgt: torch.Tensor, src_pad_mask: torch.Tensor, tgt_pad_mask: torch.Tensor
    ) -> torch.Tensor:
        # input (src) -> (batch, sl_src)
        # input (tgt) -> (batch, sl_tgt)
        # input (src_pad_mask) -> (batch, sl_src)
        # input (tgt_pad_mask) -> (batch, sl_tgt)
        # output -> (batch, sl_tgt, vocab_size_tgt)

        # need to make sure masks are broadcastable to (batch, head, sl_q, sl_kv) to pass on to attention
        src_pad_mask = src_pad_mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, sl_src)
        tgt_pad_mask = tgt_pad_mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, sl_tgt)
        tgt_causal_mask = self._compute_causal_mask(tgt.shape[-1])  # (sl_tgt, sl_tgt)

        self_mask = tgt_causal_mask | tgt_pad_mask  # (batch, 1, sl_tgt, sl_tgt)

        out = self.encode(x=src, mask=src_pad_mask)
        out = self.decode(
            x=tgt, x_enc=out, self_mask=self_mask, cross_mask=src_pad_mask
        )  # cross_mask depends on src as the input data for cross-attention is the encoder's output
        logits = self.linear(out)
        return logits

    # TODO: write code to generate output at inference time
    # don't forget to apply softmax - as per the paper
    # def generate(self):
