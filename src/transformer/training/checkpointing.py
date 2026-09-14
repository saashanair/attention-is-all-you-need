import torch


def save_checkpoint(checkpoint_path, epoch, model, optimizer, lr_scheduler, train_loss, val_loss):
    cpt = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }

    torch.save(cpt, checkpoint_path)
    print(f'checkpoint saved at epoch {epoch} to {checkpoint_path}')


def load_checkpoint(checkpoint_path, model, optimizer, lr_scheduler):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    best_loss = checkpoint['val_loss']

    return start_epoch, best_loss
