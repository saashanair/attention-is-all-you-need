import torch
from datasets import DatasetDict
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from ..tokenizers import Tokenizer


def get_pad_mask(x: torch.Tensor, pad_id: int) -> torch.Tensor:
    return x == pad_id


def collate_translation_batch(batch: list[tuple[torch.Tensor, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
    src = [b[0] for b in batch]
    tgt = [b[1] for b in batch]

    src_batch = pad_sequence(src, batch_first=True, padding_value=pad_id)  # (batch, seq_len_src)
    tgt_batch = pad_sequence(tgt, batch_first=True, padding_value=pad_id)  # (batch, seq_len_tgt)

    tgt_batch_shifted = tgt_batch[
        :, :-1
    ]  # (batch, seq_len_tgt-1); decoder input is right shifted => has <BOS> but not <EOS>
    label = tgt_batch[:, 1:]  # (batch, seq_len_tgt-1); label input is shifted left => no <BOS> but has <EOS>

    return {
        'encoder_input': src_batch,  # (batch, seq_len_src)
        'decoder_input': tgt_batch_shifted,  # (batch, seq_len_tgt-1)
        'label': label,  # (batch, seq_len_tgt-1)
        'src_pad_mask': get_pad_mask(
            src_batch, pad_id
        ),  # (batch, seq_len_src); whereever the tensor has pad_id, should be True
        'tgt_pad_mask': get_pad_mask(tgt_batch_shifted, pad_id),  # (batch, seq_len_tgt-1)
    }


class TranslationDataset(Dataset):
    def __init__(self, src_tokens: list[list[int]], tgt_tokens: list[list[int]]) -> None:

        assert len(src_tokens) == len(tgt_tokens), 'Source and Target data should have the same number of rows.'

        self.src_tokens = src_tokens
        self.tgt_tokens = tgt_tokens

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.src_tokens[index], dtype=torch.long),
            torch.tensor(self.tgt_tokens[index], dtype=torch.long),
        )

    def __len__(self) -> int:
        return len(self.src_tokens)


def drop_long_rows(
    src_tokens: list[list[int]], tgt_tokens: list[list[int]], max_seq_len: int
) -> tuple[list[list[int]], list[list[int]]]:
    src_filtered = []
    tgt_filtered = []
    for st, tt in zip(src_tokens, tgt_tokens):
        if len(st) <= max_seq_len and len(tt) <= max_seq_len:
            src_filtered.append(st)
            tgt_filtered.append(tt)
    return src_filtered, tgt_filtered


def preprocess_data(
    src_data: list[str], tgt_data: list[str], src_tokenizer: Tokenizer, tgt_tokenizer: Tokenizer, max_seq_len: int
) -> tuple[list[list[int]], list[list[int]]]:
    src_tokens = [src_tokenizer.encode(text, add_special_tokens=True) for text in src_data]
    tgt_tokens = [tgt_tokenizer.encode(text, add_special_tokens=True) for text in tgt_data]

    return drop_long_rows(src_tokens, tgt_tokens, max_seq_len)


def build_translation_dataset(
    raw: DatasetDict, src_lang: str, tgt_lang: str, max_seq_len: int, src_tokenizer: Tokenizer, tgt_tokenizer: Tokenizer
) -> dict[str, TranslationDataset]:
    train_data = preprocess_data(
        raw['train'][src_lang], raw['train'][tgt_lang], src_tokenizer, tgt_tokenizer, max_seq_len
    )
    train_data = TranslationDataset(*train_data)

    val_data = preprocess_data(
        raw['validation'][src_lang], raw['validation'][tgt_lang], src_tokenizer, tgt_tokenizer, max_seq_len
    )
    val_data = TranslationDataset(*val_data)

    test_data = preprocess_data(raw['test'][src_lang], raw['test'][tgt_lang], src_tokenizer, tgt_tokenizer, max_seq_len)
    test_data = TranslationDataset(*test_data)

    return {
        'train': train_data,
        'validation': val_data,
        'test': test_data,
    }
