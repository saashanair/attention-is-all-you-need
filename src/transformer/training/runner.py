import torch
from torch import nn
from tqdm import tqdm

from ..data import build_dataloaders, build_translation_pipeline
from ..model import Transformer
from .config import TrainConfig
from .epoch import evaluate, train_one_epoch


def get_lr(step: int, d_model: int, warmup_steps: int) -> float:
    step = max(step, 1)
    return d_model**-0.5 * min(step**-0.5, step * warmup_steps**-1.5)


def run(cfg: TrainConfig, device: torch.device):

    translation_pipeline_metadata = build_translation_pipeline(
        data_cfg=cfg.data_cfg, tok_cfg=cfg.tok_cfg, max_seq_len=cfg.model_cfg.max_seq_len
    )
    data_loaders = build_dataloaders(
        data_dict=translation_pipeline_metadata.data_dict,
        pad_id=translation_pipeline_metadata.pad_id,
        batch_size=cfg.batch_size,
    )

    transformer = Transformer(
        src_vocab_size=translation_pipeline_metadata.src_vocab_size,
        tgt_vocab_size=translation_pipeline_metadata.tgt_vocab_size,
        shared_embeddings=cfg.shared_embeddings,
        cfg=cfg.model_cfg,
    ).to(device)

    # we are using LambdaLR to implement the lr_rate function defined in the paper
    # LambdaLR works as: lr = base_lr * lr_lambda(step_num), where base_lr is the frozen inital lr; thus setting lr to 1.0 ensure we replicate the function as defined in the paper
    optimizer = torch.optim.Adam(
        params=transformer.parameters(), lr=1.0, betas=(cfg.optim_cfg.beta1, cfg.optim_cfg.beta2), eps=cfg.optim_cfg.eps
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step_num: get_lr(step_num, cfg.model_cfg.d_model, cfg.optim_cfg.warmup_steps),
    )

    loss_fn = nn.CrossEntropyLoss(
        label_smoothing=cfg.label_smoothing, ignore_index=translation_pipeline_metadata.pad_id
    )

    for epoch in range(cfg.num_epochs):
        train_loss = train_one_epoch(
            data_loaders['train'],
            transformer,
            optimizer,
            lr_scheduler,
            loss_fn,
            device,
            epoch_label=f'Epoch {epoch + 1}/{cfg.num_epochs}',
        )
        val_loss = evaluate(data_loaders['validation'], transformer, loss_fn, device, desc='validation')

        ### LOGGING
        tqdm.write(
            f'Epoch {epoch + 1:03d}/{cfg.num_epochs:03d}'
            f' | train loss {train_loss / len(data_loaders["train"]):.3f}'
            f' | val loss {val_loss / len(data_loaders["validation"]):.3f}'
        )
