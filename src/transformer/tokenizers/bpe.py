from __future__ import annotations

from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

from .base import Tokenizer

# NOTE: implemented by Claude, wiring in the BPE tokenizer from the `tokenizers` (HF) library
# directly, as a shortcut to get BPE training working now. Plan is to replace this with a
# from-scratch BPE implementation later, consistent with the rest of this repo.


class BPETokenizer(Tokenizer):
    def __init__(
        self,
        corpus: list[str] | None = None,
        vocab_size: int | None = None,
        hf_tokenizer: HFTokenizer | None = None,
    ) -> None:
        if hf_tokenizer is not None:
            # reloading an already-trained tokenizer (see `load`) -- skip training entirely
            self._hf = hf_tokenizer
        else:
            if corpus is None or vocab_size is None:
                raise ValueError('corpus and vocab_size are required to train a new BPETokenizer')

            self._hf = HFTokenizer(BPE(unk_token=self.UNK))
            self._hf.pre_tokenizer = (
                Whitespace()
            )  # split on whitespace/punctuation before BPE learns merges within each piece

            trainer = BpeTrainer(
                vocab_size=vocab_size,
                special_tokens=[self.PAD, self.BOS, self.EOS, self.UNK],  # reserved as atomic entries, this order
            )
            self._hf.train_from_iterator(corpus, trainer=trainer)

        self._validate_specials_in_vocab()

    @property
    def vocab_size(self) -> int:
        return self._hf.get_vocab_size()

    def token_to_id(self, token: str) -> int:
        id_ = self._hf.token_to_id(token)
        return id_ if id_ is not None else self._hf.token_to_id(self.UNK)

    def id_to_token(self, idx: int) -> str:
        token = self._hf.id_to_token(idx)
        if token is None:
            raise IndexError(f'id {idx} out of range [0, {self.vocab_size})')
        return token

    def _has_token(self, token: str) -> bool:
        # raw HF lookup, not self.token_to_id -- that one falls back to unk_id, which would make
        # every token look like it "exists" (as unk)
        return self._hf.token_to_id(token) is not None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = self._hf.encode(text).ids
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self._hf.decode(ids, skip_special_tokens=skip_special_tokens)

    def save(self, path: str) -> None:
        self._hf.save(path)

    @classmethod
    def load(cls, path: str) -> BPETokenizer:
        return cls(hf_tokenizer=HFTokenizer.from_file(path))
