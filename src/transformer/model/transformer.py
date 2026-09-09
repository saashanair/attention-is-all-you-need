import torch
import torch.nn as nn
from pydantic import BaseModel, Field, PositiveInt, field_validator
from .encoder import Encoder
from .decoder import Decoder

class TransformerConfig(BaseModel):
    n_enc: PositiveInt
    n_dec: PositiveInt
    src_vocab_size: PositiveInt
    tgt_vocab_size: PositiveInt
    max_seq_len: PositiveInt
    d_model: PositiveInt
    d_kq: PositiveInt
    d_v: PositiveInt
    h: PositiveInt
    d_ff: PositiveInt
    p_drop: float = Field(default=0.1, ge=0.0, lt=1.0)
    share_embeddings: bool = True

    @field_validator('d_model')
    @classmethod
    def _ensure_dmodel_is_even(cls, v: int) -> int:
        # the formula for positional encoding, as implemented from the paper,
        # relies on d_model being even
        if v % 2:
            raise ValueError(f'd_model must be even, got {v}')
        return v

class Transformer(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.encoder = Encoder(n_enc=cfg.n_enc, vocab_size=cfg.src_vocab_size, max_seq_len=cfg.max_seq_len, d_model=cfg.d_model, d_kq=cfg.d_kq, d_v=cfg.d_v, h=cfg.h, d_ff=cfg.d_ff, p_drop=cfg.p_drop)
        self.decoder = Decoder(n_dec=cfg.n_dec, vocab_size=cfg.tgt_vocab_size, max_seq_len=cfg.max_seq_len, d_model=cfg.d_model, d_kq=cfg.d_kq, d_v=cfg.d_v, h=cfg.h, d_ff=cfg.d_ff, p_drop=cfg.p_drop)

        self.linear = nn.Linear(in_features=cfg.d_model, out_features=cfg.tgt_vocab_size)

        if cfg.share_embeddings:
            self.decoder.emb.emb.weight = self.encoder.emb.emb.weight
        self.linear.weight = self.decoder.emb.emb.weight

        causal_mask = torch.triu(torch.ones((cfg.max_seq_len, cfg.max_seq_len), dtype=torch.bool), diagonal=1)
        self.register_buffer('causal_mask', causal_mask, persistent=False)

    def _compute_causal_mask(self, mask_size:int):
        return self.causal_mask[:mask_size, :mask_size]

    def encode(self, source, source_mask=None):
        return self.encoder(x=source, mask=source_mask)

    def decode(self, encoder_output: torch.Tensor, decoder_input: torch.Tensor, self_mask: torch.Tensor | None = None, cross_mask: torch.Tensor | None = None):
        return self.decoder(x=decoder_input, x_enc=encoder_output, self_mask=self_mask, cross_mask=cross_mask)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_pad_mask: torch.Tensor, tgt_pad_mask: torch.Tensor) -> torch.Tensor:
        # src -> (batch, sl_src)
        # tgt -> (batch, sl_tgt)
        # src_pad_mask -> (batch, sl_src)
        # tgt_pad_mask -> (batch, sl_tgt)

        # need to make sure masks are broadcastable to (batch, head, sl_q, sl_kv) to pass on to attention
        src_pad_mask = src_pad_mask.unsqueeze(1).unsqueeze(1) # (batch, 1, 1, sl_src)
        tgt_pad_mask = tgt_pad_mask.unsqueeze(1).unsqueeze(1) # (batch, 1, 1, sl_tgt)
        tgt_causal_mask = self._compute_causal_mask(tgt.shape[-1]) # (sl_tgt, sl_tgt)

        self_mask = tgt_causal_mask | tgt_pad_mask # (batch, 1, sl_tgt, sl_tgt)

        out = self.encode(source=src, source_mask=src_pad_mask)
        out = self.decode(encoder_output=out, decoder_input=tgt, self_mask=self_mask, cross_mask=src_pad_mask) # cross_mask depends on src as the input data there is the encoder's outptu
        logits = self.linear(out)
        return logits

    #TODO: write code to generate output at inference time
    # don't forget to apply softmax - as per the paper
    #def generate(self):
