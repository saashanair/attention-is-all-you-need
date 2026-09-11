from abc import ABC, abstractmethod
from typing import ClassVar


class Tokenizer(ABC):
    PAD, BOS, EOS, UNK = '<pad>', '<start>', '<end>', '<unk>'
    SPECIALS: ClassVar[list[str]] = [PAD, BOS, EOS, UNK]

    # === properties for id look up ===
    @property
    def bos_id(self) -> int:
        return self.token_to_id(self.BOS)

    @property
    def eos_id(self) -> int:
        return self.token_to_id(self.EOS)

    @property
    def pad_id(self) -> int:
        return self.token_to_id(self.PAD)

    @property
    def unk_id(self) -> int:
        return self.token_to_id(self.UNK)

    @property
    def special_ids(self) -> set[int]:
        return {self.pad_id, self.bos_id, self.eos_id, self.unk_id}

    # === abstract methods ===
    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @abstractmethod
    def token_to_id(self, token: str) -> int: ...

    @abstractmethod
    def id_to_token(self, idx: int) -> str: ...

    @abstractmethod
    def _has_token(self, token: str) -> bool: ...

    @abstractmethod
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...

    @abstractmethod
    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str: ...

    @abstractmethod
    def save(self, path: str) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> Tokenizer: ...

    # === methods ===

    def _validate_specials_in_vocab(self) -> None:
        missing = [t for t in self.SPECIALS if not self._has_token(t)]
        if missing:
            raise ValueError(f'vocab missing special tokens: {missing}')
