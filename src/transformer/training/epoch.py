import torch
from tqdm import tqdm

from ..device import move_batch_to_device


def train_one_epoch(data_loader, transformer, optimizer, lr_scheduler, loss_fn, device, epoch_label):
    transformer.train()
    running_loss = 0.0

    pbar = tqdm(data_loader, desc=epoch_label, unit='batch')

    for batch_idx, batch in enumerate(pbar):
        batch = move_batch_to_device(batch, device=device)
        logits = transformer(
            src=batch['encoder_input'],
            tgt=batch['decoder_input'],
            src_pad_mask=batch['src_pad_mask'],
            tgt_pad_mask=batch['tgt_pad_mask'],
        )

        logits = torch.reshape(logits, (-1, logits.size(-1)))  # (batch * sequence_length, vocab_size)
        labels = torch.reshape(batch['label'], (-1,))  # (batch * sequence_length,)

        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        running_loss += loss.item()
        pbar.set_postfix(loss=f'{loss.item():.3f}', lr=f'{lr_scheduler.get_last_lr()[0]:.2e}')

    return running_loss


def evaluate(data_loader, transformer, loss_fn, device, desc='validation'):
    transformer.eval()
    running_loss = 0.0

    pbar = tqdm(data_loader, desc=f'  {desc}', unit='batch', leave=False)

    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            batch = move_batch_to_device(batch, device=device)
            logits = transformer(
                src=batch['encoder_input'],
                tgt=batch['decoder_input'],
                src_pad_mask=batch['src_pad_mask'],
                tgt_pad_mask=batch['tgt_pad_mask'],
            )

            logits = torch.reshape(logits, (-1, logits.size(-1)))
            labels = torch.reshape(batch['label'], (-1,))

            loss = loss_fn(logits, labels)
            running_loss += loss.item()

    return running_loss
