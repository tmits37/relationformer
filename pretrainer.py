import os
from tqdm import tqdm

import torch

def train_epoch(model,
                data_loader, 
                loss_fn, 
                optimizer, 
                device, 
                epoch, 
                writer, 
                is_master
                ):

    model.train()


    total_loss = 0
    with tqdm(data_loader, unit="batch") as tepoch:
        max_iter_in_epoch = len(tepoch)
        for idx, batch in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch}")
            images, seg, heatmap = batch

            images = images.to(device)
            seg = seg.to(device)
            heatmap = heatmap.to(device)

            optimizer.zero_grad() # 배치별 그래디언트 초기화
            out = model(images)
            losses = loss_fn(out, heatmap)
            loss = losses

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if is_master:
                iters = int(max_iter_in_epoch * (epoch - 1) + idx)
                writer.add_scalar('Train/Loss', loss.item() / len(images), iters)
            
            tepoch.set_postfix(loss=loss.item() / len(images))


    return total_loss / len(data_loader)


def validate_epoch(
    model,
    config,
    data_loader, 
    loss_fn, 
    device, 
    epoch, 
    val_interval,
    writer, 
    is_master):

    model.eval()

    total_loss = 0
    with tqdm(data_loader, unit="batch") as tepoch:
        max_iter_in_epoch = len(tepoch)
        for idx, batch in enumerate(tepoch):
            tepoch.set_description(f"Val: {epoch}")
            images, seg, heatmap = batch

            images = images.to(device)
            heatmap = heatmap.to(device)

            out = model(images)
            losses = loss_fn(out, heatmap)
            loss = losses
            total_loss += loss.item()

            if is_master:
                iters = int(max_iter_in_epoch * (epoch - 1) / val_interval + idx)
                writer.add_scalar('Val/Loss', loss.item() / len(images), iters)

            tepoch.set_postfix(loss=loss.item() / len(images))

    return total_loss / len(data_loader)


def save_checkpoint(model, optimizer, scheduler, epoch, config):
    """
    Save a checkpoint of the training process.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epoch (int): The current epoch number.
        checkpoint_path (str): The file path to save the checkpoint.
    """
    # Prepare the checkpoint dictionary
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'schedulaer_state_dict': scheduler.state_dict()
    }

    # If using DDP, save the original model wrapped inside DDP
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        checkpoint['model_state_dict'] = model.module.state_dict()

    savedir = os.path.join(config.TRAIN.SAVE_PATH, "runs", '%s_%d' % (config.log.exp_name, config.DATA.SEED), 'models')
    os.makedirs(savedir, exist_ok=True)
    checkpoint_path = os.path.join(savedir, f'epochs_{epoch}.pth')

    # Save the checkpoint
    torch.save(checkpoint, checkpoint_path)