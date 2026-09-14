from .data.config import DataConfig
from .device import get_device
from .model.config import TransformerConfig
from .tokenizers import TokenizerConfig
from .training import runner
from .training.config import OptimizerConfig, TrainConfig


def main():
    device = get_device()
    cfg = TrainConfig(
        model_cfg=TransformerConfig(),  # all fields have defaults now
        optim_cfg=OptimizerConfig(),  # all defaults
        data_cfg=DataConfig(),  # all default
        # tok_cfg=TokenizerConfig(tokenizer_type='char', vocab_size=None),
        tok_cfg=TokenizerConfig(tokenizer_type='bpe', vocab_size=1000),
        num_epochs=1,  # keep tiny for a first smoke run
        batch_size=128,
    )
    runner.run(cfg=cfg, device=device)


if __name__ == '__main__':
    main()
