from pydantic import BaseModel, Field, PositiveInt, field_validator


class TransformerConfig(BaseModel):
    n_enc: PositiveInt = 2
    n_dec: PositiveInt = 2
    max_seq_len: PositiveInt = 1000
    d_model: PositiveInt = 256
    d_kq: PositiveInt = 64
    d_v: PositiveInt = 64
    h: PositiveInt = 4
    d_ff: PositiveInt = 1024
    p_drop: float = Field(default=0.1, ge=0.0, lt=1.0)

    @field_validator('d_model')
    @classmethod
    def _ensure_dmodel_is_even(cls, v: int) -> int:
        # the formula for positional encoding, as implemented from the paper,
        # relies on d_model being even
        if v % 2:
            raise ValueError(f'd_model must be even, got {v}')
        return v
