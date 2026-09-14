import torch


def save_checkpoint(
    checkpoint_path, epoch, model, optimizer, lr_scheduler, train_loss, val_loss, epochs_without_improvement
):
    cpt = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'epochs_without_improvement': epochs_without_improvement,
    }

    torch.save(cpt, checkpoint_path)
    print(f'checkpoint saved at epoch {epoch} to {checkpoint_path}')


def load_checkpoint(checkpoint_path, model, optimizer, lr_scheduler):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    epochs_without_improvement = checkpoint['epochs_without_improvement']

    return start_epoch, epochs_without_improvement


def load_best_val_loss(checkpoint_path):
    return torch.load(checkpoint_path)['val_loss']
