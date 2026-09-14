from typing import Literal

from pydantic import BaseModel, PositiveInt, model_validator


class TokenizerConfig(BaseModel):
    tokenizer_type: Literal['char', 'bpe'] = 'char'
    shared_tokenizer: bool = True
    vocab_size: PositiveInt | None = None  # used when tokenizer shared between src and tgt
    src_vocab_size: PositiveInt | None = None
    tgt_vocab_size: PositiveInt | None = None

    @model_validator(mode='after')
    def _validate_vocab_size(self):
        if self.tokenizer_type != 'bpe':
            return self  # char tokenizer derives its vocab from the corpus, nothing to require here

        # runs only for bpe tokenizer; not needed for char
        if self.shared_tokenizer:
            if self.vocab_size is None:
                raise ValueError('shared bpe tokenizer requires vocab_size to be set')
        else:
            if self.src_vocab_size is None or self.tgt_vocab_size is None:
                raise ValueError('non-shared bpe tokenizer requires src_vocab_size and tgt_vocab_size to be set')

        return self
