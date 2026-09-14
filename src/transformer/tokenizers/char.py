from .base import Tokenizer


class CharTokenizer(Tokenizer):
    # vocab_size added to make the signature uniform
    def __init__(self, corpus: list[str]):
        chars = sorted(
            {ch for line in corpus for ch in line}
        )  # using set comprehension over join for memory efficiency, as join would create a new string that is the total length of the corpus
        if len(chars) == 0:
            raise ValueError('corpus is empty')
        self.vocab = self.SPECIALS + chars
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self._validate_specials_in_vocab()
        self._unk_id = self.stoi[self.UNK]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def token_to_id(self, token: str) -> int:
        return self.stoi.get(token, self._unk_id)

    def id_to_token(self, idx: int) -> str:
        if not 0 <= idx < self.vocab_size:
            raise IndexError(f'id {idx} out of range [0, {self.vocab_size})')
        return self.vocab[idx]

    def _has_token(self, token: str) -> bool:
        return token in self.stoi

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = [self.token_to_id(ch) for ch in text]
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        special_ids = self.special_ids
        return ''.join(self.id_to_token(i) for i in ids if not skip_special_tokens or i not in special_ids)

    def save(self, path: str) -> None:
        raise NotImplementedError('save has not yet been implemented on CharTokenizer')

    @classmethod
    def load(cls, path: str) -> CharTokenizer:
        raise NotImplementedError('load has not yet been implemented on CharTokenizer')
