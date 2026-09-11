import math

import torch
import torch.nn.functional as F
from torch import nn


def scaled_dot_product_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, scaling_factor: float, mask: torch.Tensor | None = None
) -> torch.Tensor:
    # Q -> (batch, h, sequence_length_q, d_kq)
    # K -> (batch, h, sequence_length_kv, d_kq)
    # V -> (batch, h, sequence_length_kv, d_v)
    # mask -> broadcastable to (batch, h, sequence_length_q, sequence_length_kv),
    #         dtype bool, True where attention is disallowed

    K_t = torch.transpose(K, -1, -2)  # K_t -> (batch, h, d_kq, sequence_length_kv)
    out = torch.matmul(Q, K_t)  # out -> (batch, h, sequence_length_q, sequence_length_kv)
    out = out / scaling_factor  # out -> (batch, h, sequence_length_q, sequence_length_kv)

    if mask is not None:
        # mask is True, where the attention is disallowed
        # which mean, wherever the mask has 1, set it to -inf,
        # so that softmax gives these exactly 0 weight (exp(-inf)=0)
        out = out.masked_fill(mask, float('-inf'))

    out = F.softmax(out, dim=-1)  # out -> (batch, h, sequence_length_q, sequence_length_kv)
    out = torch.matmul(out, V)  # out -> (batch, h, sequence_length_q, d_v)
    return out


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, d_kq: int, d_v: int, h: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_kq = d_kq  # for Q.K_t to work out, d_k and d_q must be equal
        self.d_v = d_v
        self.h = h

        self.scaling_factor = math.sqrt(d_kq)

        self.w_q = nn.Linear(d_model, d_kq * h)
        self.w_k = nn.Linear(d_model, d_kq * h)
        self.w_v = nn.Linear(d_model, d_v * h)
        self.w_o = nn.Linear(d_v * h, d_model)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # input (x_q) -> (batch, sequence_length_q, d_model)
        # input (x_kv) -> (batch, sequence_length_kv, d_model)
        # input (mask) -> built by the caller (encoder/decoder layer) for padding and/or causal,
        #       broadcastable to (batch, h, sequence_length_q, sequence_length_kv)

        b_q, sl_q = x_q.shape[0], x_q.shape[1]
        b_kv, sl_kv = x_kv.shape[0], x_kv.shape[1]

        assert b_q == b_kv, 'batch size between q and kv should be the same'

        Q = self.w_q(x_q)  # Q -> (batch, sequence_length_q, d_kq * h)
        Q = torch.reshape(Q, (b_q, sl_q, self.h, self.d_kq))  # Q -> (batch, sequence_length_q, h, d_kq)
        Q = torch.transpose(Q, 1, 2)  # Q -> (batch, h, sequence_length_q, d_kq)

        K = self.w_k(x_kv)
        K = torch.reshape(K, (b_kv, sl_kv, self.h, self.d_kq))  # K -> (batch, sequence_length_kv, h, d_kq)
        K = torch.transpose(K, 1, 2)  # K -> (batch, h, sequence_length_kv, d_kq)

        V = self.w_v(x_kv)
        V = torch.reshape(V, (b_kv, sl_kv, self.h, self.d_v))  # V -> (batch, sequence_length_kv, h, d_v)
        V = torch.transpose(V, 1, 2)  # V -> (batch, h, sequence_length_kv, d_v)

        out = scaled_dot_product_attention(
            Q=Q, K=K, V=V, scaling_factor=self.scaling_factor, mask=mask
        )  # out -> (batch, h, sequence_length_q, d_v)

        out = torch.transpose(out, 1, 2)  # out -> (batch, sequence_length_q, h, d_v)
        out = torch.reshape(out, (b_q, sl_q, self.d_v * self.h))  # out -> (batch, sequence_length_q, d_v * h)

        out = self.w_o(out)  # out -> (batch, sequence_length_q, d_model)

        return out
