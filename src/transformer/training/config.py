from pydantic import BaseModel, Field, PositiveFloat, PositiveInt, model_validator

from ..data.config import DataConfig
from ..model.config import TransformerConfig
from ..tokenizers.config import TokenizerConfig


class OptimizerConfig(BaseModel):
    beta1: PositiveFloat = Field(default=0.9, lt=1.0)
    beta2: PositiveFloat = Field(default=0.98, lt=1.0)
    eps: PositiveFloat = 1e-9
    warmup_steps: PositiveInt = 4000


class TrainConfig(BaseModel):
    model_cfg: TransformerConfig
    tok_cfg: TokenizerConfig
    data_cfg: DataConfig
    optim_cfg: OptimizerConfig
    num_epochs: PositiveInt = 2
    batch_size: PositiveInt = 32
    label_smoothing: float = Field(default=0.1, ge=0.0, lt=1.0)
    shared_embeddings: bool = True
    resume_path: str | None = None
    chkpt_n_epochs: int = 10

    @model_validator(mode='after')
    def _check_embedding_sharing(self):
        if self.shared_embeddings and not self.tok_cfg.shared_tokenizer:
            raise ValueError('shared_embeddings=True requires shared_tokenizer=True')
        return self
