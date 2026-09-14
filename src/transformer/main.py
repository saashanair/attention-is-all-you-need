from datetime import datetime
from pathlib import Path

from .data.config import DataConfig
from .device import get_device
from .model.config import TransformerConfig
from .paths import get_experiment_path
from .tokenizers import TokenizerConfig
from .training import runner
from .training.config import OptimizerConfig, TrainConfig


def validate_checkpoint_compatibility(saved_cfg, cfg, resume_path: Path):
    if saved_cfg.model_cfg != cfg.model_cfg:
        return f'model settings do not match the one found at resume_path={resume_path}'
    if saved_cfg.tok_cfg != cfg.tok_cfg:
        return f'tokenizer settings do not match the one found at resume_path={resume_path}'
    return None


def sync_train_cfg(run_path, train_cfg):
    if train_cfg.resume_path is not None:
        saved_cfg = TrainConfig.model_validate_json((run_path / 'config.json').read_text())
        msg = validate_checkpoint_compatibility(saved_cfg, train_cfg, run_path)
        if msg:
            raise ValueError(msg)
        return

    with open(f'{run_path}/config.json', 'w') as f:
        f.write(train_cfg.model_dump_json(indent=2))


def get_experiment_dir_name(train_cfg) -> str:
    # add astimezone to satisfy the linter
    return f'{datetime.now().astimezone().strftime("%Y%m%d-%H%M")}-{train_cfg.tok_cfg.tokenizer_type}-ne{train_cfg.model_cfg.n_enc}-nd{train_cfg.model_cfg.n_dec}-ep{train_cfg.num_epochs}-b{train_cfg.batch_size}'


def main():
    device = get_device()
    cfg = TrainConfig(
        model_cfg=TransformerConfig(),  # all fields have defaults now
        optim_cfg=OptimizerConfig(warmup_steps=2000),  # ~29k examples / 128 batch * 100 epochs =~ 22.7k steps total
        data_cfg=DataConfig(),  # all default
        tok_cfg=TokenizerConfig(tokenizer_type='bpe', vocab_size=8000),
        num_epochs=100,
        batch_size=128,
        resume_path=None,
        # resume_path='runs/20260914-1733-bpe-ne2-nd2-ep100-b128',
        early_stopping=True,
        early_stopping_patience=5,
        early_stopping_min_delta=0.1,
    )
    store_path = get_experiment_path(
        experiment_dir_name=get_experiment_dir_name(train_cfg=cfg), resume_path=cfg.resume_path
    )
    sync_train_cfg(store_path, cfg)

    runner.run(cfg=cfg, device=device, store_path=store_path)


if __name__ == '__main__':
    main()
