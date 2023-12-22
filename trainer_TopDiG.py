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
    scaler = torch.cuda.amp.GradScaler()
    with tqdm(data_loader, unit="batch") as tepoch:
        max_iter_in_epoch = len(tepoch)
        for idx, batch in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch}")
            # TODO 코코스타일이 적용되는지 확인하기
            images, seg, nodes, edges = batch

            images = images.to(device)
            seg = seg.to(device)
            nodes = [node.to(device) for node in nodes]
            edges = [edge.to(device) for edge in edges]

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                out = model(images)
                losses = loss_fn(out, {'nodes': nodes, 'edges': edges})
                loss = losses['total']

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            if is_master:
                iters = max_iter_in_epoch * epoch + idx
                writer.add_scalar('Train/Loss', loss.item(), iters)
                writer.add_scalar('Train/Loss/class', losses['class'].item(), iters)
                writer.add_scalar('Train/Loss/nodes', losses['nodes'].item(), iters)
                writer.add_scalar('Train/Loss/boxes', losses['boxes'].item(), iters)
                writer.add_scalar('Train/Loss/edges', losses['edges'].item(), iters)
            
            tepoch.set_postfix(loss=loss.item())


    return total_loss / len(data_loader)


def validate_epoch(
    model, 
    config,
    data_loader, 
    loss_fn, 
    device, 
    epoch, 
    writer, 
    is_master):

    model.eval()

    total_loss = 0
    with tqdm(data_loader, unit="batch") as tepoch:
        max_iter_in_epoch = len(tepoch)
        for idx, batch in enumerate(tepoch):
            tepoch.set_description(f"Val: {epoch}")
            images, _, nodes, edges = batch

            images = images.to(device)
            nodes = [node.to(device) for node in nodes]
            edges = [edge.to(device) for edge in edges]

            out = model(images)
            losses = loss_fn(out, {'nodes': nodes, 'edges': edges})
            loss = losses['total']
            total_loss += loss.item()

            if is_master:
                iters = max_iter_in_epoch * epoch + idx
                writer.add_scalar('Val/Loss', loss.item(), iters)
                writer.add_scalar('Val/Loss/class', losses['class'].item(), iters)
                writer.add_scalar('Val/Loss/nodes', losses['nodes'].item(), iters)
                writer.add_scalar('Val/Loss/boxes', losses['boxes'].item(), iters)
                writer.add_scalar('Val/Loss/edges', losses['edges'].item(), iters)

            tepoch.set_postfix(loss=loss.item())

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
    if isinstance(model, torch.nn.parallel.DistributedDataParallel): # 해당없음
        checkpoint['model_state_dict'] = model.module.state_dict()

    savedir = os.path.join(config.TRAIN.SAVE_PATH, "runs", '%s_%d' % (config.log.exp_name, config.DATA.SEED), 'models')
    os.makedirs(savedir, exist_ok=True)
    checkpoint_path = os.path.join(savedir, f'epochs_{epoch}.pth')

    # Save the checkpoint
    torch.save(checkpoint, checkpoint_path)