from functools import partial

from torch.utils.data import DataLoader

from .dataset import collate_translation_batch


def get_dataloader(data, batch_size: int, pad_id: int, shuffle: bool = True):
    return DataLoader(
        dataset=data,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(collate_translation_batch, pad_id=pad_id),
    )


def build_dataloaders(data_dict: dict, pad_id: int, batch_size: int) -> dict[str, DataLoader]:
    return {
        'train': get_dataloader(data_dict['train'], batch_size=batch_size, pad_id=pad_id, shuffle=True),
        'validation': get_dataloader(data_dict['validation'], batch_size=batch_size, pad_id=pad_id, shuffle=False),
        'test': get_dataloader(data_dict['test'], batch_size=batch_size, pad_id=pad_id, shuffle=False),
    }
