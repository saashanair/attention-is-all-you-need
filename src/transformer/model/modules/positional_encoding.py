import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        pe = torch.zeros((max_seq_len, d_model))  # pe -> (max_seq_len, d_model)
        pos = torch.arange(max_seq_len).unsqueeze(1)  # pos -> (max_seq_len, 1)

        # paper's calculation for positional encoding assumes that d_model is always even (specifically d_model = 512)
        if d_model % 2:
            raise ValueError(f'd_model must be even, currently d_model={d_model}')

        # a^b = e^(b ln a) -> 1/10000^(2i/d_model) = 10000^(-(2i/d_model)) = e^(-(2i/d_model) * ln(10000))
        _2i = torch.arange(0, d_model, 2)  # _2i -> (d_model/2)
        _2i_term = -(_2i / d_model)
        log_term = math.log(10000)
        exp_term = _2i_term * log_term  # exp_term -> (d_model/2)
        denom = torch.exp(exp_term)  # denom -> (d_model/2)

        # pe -> (max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(pos * denom)
        pe[:, 1::2] = torch.cos(pos * denom)

        self.register_buffer('pe', pe, persistent=False)

    def forward(self, seq_len: int) -> torch.Tensor:
        if not 1 <= seq_len <= self.max_seq_len:
            raise ValueError(f'seq_len={seq_len} should be an integer in [1, max_seq_len={self.max_seq_len}]')
        # input -> int describing len of text sequence
        # output -> (seq_len, d_model)
        return self.pe[:seq_len, :]
