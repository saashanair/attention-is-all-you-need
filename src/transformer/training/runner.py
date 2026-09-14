from pathlib import Path

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..data import build_dataloaders, build_translation_pipeline
from ..model import Transformer
from ..paths import CHECKPOINTS_DIR, LOGGING_DIR, TOKENIZER_DIR
from .checkpointing import load_best_val_loss, load_checkpoint, save_checkpoint
from .config import TrainConfig
from .epoch import evaluate, train_one_epoch

LAST_CHECKPOINT = 'last.pt'
BEST_CHECKPOINT = 'best.pt'


def get_lr(step: int, d_model: int, warmup_steps: int) -> float:
    step = max(step, 1)
    return d_model**-0.5 * min(step**-0.5, step * warmup_steps**-1.5)


def run(cfg: TrainConfig, device: torch.device, store_path: Path):

    translation_pipeline = build_translation_pipeline(
        data_cfg=cfg.data_cfg,
        tok_cfg=cfg.tok_cfg,
        max_seq_len=cfg.model_cfg.max_seq_len,
        tok_store_path=store_path / TOKENIZER_DIR,
        resume=bool(cfg.resume_path),
    )
    data_loaders = build_dataloaders(
        data_dict=translation_pipeline.data_dict,
        pad_id=translation_pipeline.pad_id,
        batch_size=cfg.batch_size,
    )

    transformer = Transformer(
        src_vocab_size=translation_pipeline.src_vocab_size,
        tgt_vocab_size=translation_pipeline.tgt_vocab_size,
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

    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing, ignore_index=translation_pipeline.pad_id)

    tensorboard_writer = SummaryWriter(log_dir=f'{store_path}/{LOGGING_DIR}')

    start_epoch = 0
    best_loss = float('inf')

    if cfg.resume_path:
        start_epoch = load_checkpoint(
            checkpoint_path=f'{store_path}/{CHECKPOINTS_DIR}/{LAST_CHECKPOINT}',
            model=transformer,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )
        best_loss = load_best_val_loss(checkpoint_path=f'{store_path}/{CHECKPOINTS_DIR}/{BEST_CHECKPOINT}')

    end_epoch = start_epoch + cfg.num_epochs

    for epoch in range(start_epoch, end_epoch):
        train_loss = train_one_epoch(
            data_loaders['train'],
            transformer,
            optimizer,
            lr_scheduler,
            loss_fn,
            device,
            epoch_label=f'Epoch {epoch + 1}/{end_epoch}',
            tensorboard_writer=tensorboard_writer,
        )
        val_loss = evaluate(data_loaders['validation'], transformer, loss_fn, device, desc='validation')

        if epoch == start_epoch or epoch == end_epoch - 1 or epoch % cfg.chkpt_n_epochs == 0:
            save_checkpoint(
                checkpoint_path=f'{store_path}/{CHECKPOINTS_DIR}/{LAST_CHECKPOINT}',
                epoch=epoch,
                model=transformer,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                train_loss=train_loss,
                val_loss=val_loss,
            )

        if val_loss <= best_loss:
            save_checkpoint(
                checkpoint_path=f'{store_path}/{CHECKPOINTS_DIR}/{BEST_CHECKPOINT}',
                epoch=epoch,
                model=transformer,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                train_loss=train_loss,
                val_loss=val_loss,
            )
            best_loss = val_loss

        ### LOGGING
        tqdm.write(f'Epoch {epoch + 1:03d}/{end_epoch:03d} | train loss {train_loss:.3f} | val loss {val_loss:.3f}')
        tensorboard_writer.add_scalar('Loss/train', train_loss, epoch)
        tensorboard_writer.add_scalar('Loss/val', val_loss, epoch)
        tensorboard_writer.add_scalar('Best Loss/val', best_loss, epoch)
        tensorboard_writer.flush()

    tensorboard_writer.close()
