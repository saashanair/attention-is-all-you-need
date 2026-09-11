import torch
import torch.nn as nn
from pydantic import BaseModel, Field, PositiveInt, field_validator, model_validator
from .encoder import Encoder
from .decoder import Decoder

class TransformerArchConfig(BaseModel):
    n_enc: PositiveInt = 2
    n_dec: PositiveInt = 2
    max_seq_len: PositiveInt = 1000
    d_model: PositiveInt = 256
    d_kq: PositiveInt = 64
    d_v: PositiveInt = 64
    h: PositiveInt = 4
    d_ff: PositiveInt = 1024
    p_drop: float = Field(default=0.1, ge=0.0, lt=1.0)

    @field_validator('d_model')
    @classmethod
    def _ensure_dmodel_is_even(cls, v: int) -> int:
        # the formula for positional encoding, as implemented from the paper,
        # relies on d_model being even
        if v % 2:
            raise ValueError(f'd_model must be even, got {v}')
        return v

class TransformerVocabConfig(BaseModel):
    src_vocab_size: PositiveInt
    tgt_vocab_size: PositiveInt
    shared_embeddings: bool = True

    @model_validator(mode='after')
    def _check_embedding_sharing(self):
        # not hard-line defense, but stops obvious error of mismatched vocab_sizes with shared embeddings
        if self.shared_embeddings and self.src_vocab_size != self.tgt_vocab_size:
            raise ValueError('shared_embeddings=True requires src_vocab_size and tgt_vocab_size to match')
        return self

class Transformer(nn.Module):
    def __init__(self, arch_cfg: TransformerArchConfig, vocab_cfg: TransformerVocabConfig) -> None:
        super().__init__()
        self.encoder = Encoder(n_enc=arch_cfg.n_enc, vocab_size=vocab_cfg.src_vocab_size, max_seq_len=arch_cfg.max_seq_len, d_model=arch_cfg.d_model, d_kq=arch_cfg.d_kq, d_v=arch_cfg.d_v, h=arch_cfg.h, d_ff=arch_cfg.d_ff, p_drop=arch_cfg.p_drop)
        self.decoder = Decoder(n_dec=arch_cfg.n_dec, vocab_size=vocab_cfg.tgt_vocab_size, max_seq_len=arch_cfg.max_seq_len, d_model=arch_cfg.d_model, d_kq=arch_cfg.d_kq, d_v=arch_cfg.d_v, h=arch_cfg.h, d_ff=arch_cfg.d_ff, p_drop=arch_cfg.p_drop)

        self.linear = nn.Linear(in_features=arch_cfg.d_model, out_features=vocab_cfg.tgt_vocab_size)

        if vocab_cfg.shared_embeddings:
            self.decoder.emb.emb.weight = self.encoder.emb.emb.weight
        self.linear.weight = self.decoder.emb.emb.weight

        causal_mask = torch.triu(torch.ones((arch_cfg.max_seq_len, arch_cfg.max_seq_len), dtype=torch.bool), diagonal=1)
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
