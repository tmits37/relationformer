import os

import torch
from tqdm import tqdm


def train_epoch(model, data_loader, loss_fn, optimizer, device, epoch, writer,
                is_master):
    model.train()

    total_loss = 0
    scaler = torch.cuda.amp.GradScaler()
    with tqdm(data_loader, unit='batch') as tepoch:
        max_iter_in_epoch = len(tepoch)
        for idx, batch in enumerate(tepoch):
            tepoch.set_description(f'Epoch {epoch}')
            images, heatmaps, nodes, edges = batch

            images = images.to(device)  # 8, 3, 300, 300
            heatmaps = heatmaps.to(device)  # 8, 300, 300, 1
            nodes = [node.to(device) for node in nodes]  # 8, num_gt_nodes
            edges = [edge.to(device) for edge in edges]  # 8, num_gt_edges

            optimizer.zero_grad()
            # [0]: 256개 중에 gt 노드 개수만큼 뽑기
            # [1]: pred와 매칭되게 순서맞게 arrange

            with torch.cuda.amp.autocast():
                pred = model(images)
                gt = {'nodes': nodes, 'edges': edges, 'heatmaps': heatmaps}
                losses = loss_fn(pred, gt)
                loss = losses['total']

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            if is_master:
                iters = int(max_iter_in_epoch * (epoch - 1) + idx)
                writer.add_scalar('Train/Loss',
                                  loss.item() / len(images), iters)
                writer.add_scalar('Train/Loss/node',
                                  losses['node'].item() / len(images), iters)
                writer.add_scalar('Train/Loss/graph1',
                                  losses['graph1'].item() / len(images), iters)
                writer.add_scalar('Train/Loss/graph2',
                                  losses['graph2'].item() / len(images), iters)

            tepoch.set_postfix(loss=loss.item() / len(images))

    return total_loss / len(data_loader)


def validate_epoch(model, config, data_loader, loss_fn, device, epoch,
                   val_interval, writer, is_master):
    model.train()

    total_loss = 0
    with tqdm(data_loader, unit='batch') as tepoch:
        max_iter_in_epoch = len(tepoch)
        for idx, batch in enumerate(tepoch):
            tepoch.set_description(f'Val: {epoch}')
            images, heatmaps, nodes, edges = batch

            images = images.to(device)
            heatmaps = heatmaps.to(device)
            nodes = [node.to(device) for node in nodes]
            edges = [edge.to(device) for edge in edges]

            with torch.no_grad():
                pred = model(images)
            gt = {'nodes': nodes, 'edges': edges, 'heatmaps': heatmaps}
            losses = loss_fn(pred, gt)

            loss = losses['total']
            total_loss += loss.item()
            if is_master:
                iters = int(max_iter_in_epoch * (epoch - 1) / val_interval +
                            idx)
                writer.add_scalar('Val/Loss', loss.item() / len(images), iters)
                writer.add_scalar('Val/Loss/node',
                                  losses['node'].item() / len(images), iters)
                writer.add_scalar('Train/Loss/graph1',
                                  losses['graph1'].item() / len(images), iters)
                writer.add_scalar('Train/Loss/graph2',
                                  losses['graph2'].item() / len(images), iters)

            tepoch.set_postfix(loss=loss.item() / len(images))

    return total_loss / len(data_loader)


def save_checkpoint(model, optimizer, scheduler, epoch, config):
    """Save a checkpoint of the training process.

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
        'schedulaer_state_dict': scheduler.state_dict(),
    }

    # If using DDP, save the original model wrapped inside DDP
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):  # 해당없음
        checkpoint['model_state_dict'] = model.module.state_dict()

    savedir = os.path.join(
        config.TRAIN.SAVE_PATH,
        'runs',
        '%s_%d' % (config.log.exp_name, config.DATA.SEED),
        'models',
    )
    os.makedirs(savedir, exist_ok=True)
    checkpoint_path = os.path.join(savedir, f'epochs_{epoch}.pth')

    # Save the checkpoint
    torch.save(checkpoint, checkpoint_path)
