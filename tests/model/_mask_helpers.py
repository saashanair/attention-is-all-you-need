import torch


def get_padding_mask(sl: int, n_pad: int = 2):
    # n_pad : apply padding mask to the last n elements
    m = torch.zeros(sl, dtype=torch.bool)
    m[-n_pad:] = True
    return m


def get_causal_mask(sl: int):
    return torch.triu(torch.ones(sl, sl, dtype=torch.bool), diagonal=1)
