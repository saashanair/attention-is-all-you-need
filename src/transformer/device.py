import torch


def get_device() -> torch.device:
    return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}
