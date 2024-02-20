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
            images, seg, nodes, edges, pts_labels = batch

            images = images.to(device)
            seg = seg.to(device)
            nodes = [node.to(device) for node in nodes]
            edges = [edge.to(device) for edge in edges]
            pts_labels = [pts_label.to(device) for pts_label in pts_labels]

            optimizer.zero_grad()

            targets = {'nodes': nodes, 'edges': edges, 'segs':seg, 'pts_labels':pts_labels}
            wh = torch.ones(1,2).to(device) * 0.05
            targets['labels'] = [torch.zeros(len(x)).long().to(device) for x in targets['nodes']]
            targets['boxes'] = [torch.cat([x, wh.repeat(len(x), 1)], dim=-1) for x in targets['nodes']]
            targets_converted = [{key: value for key, value in zip(targets.keys(), values)} for values in zip(*targets.values())]

            h, out, srcs = model(images, targets=targets_converted)

            losses = loss_fn(h, out, targets)
            loss = losses['total']

            loss.backward()
            optimizer.step()

            loss_keys = list(losses.keys())
            loss_keys.remove('total')

            total_loss += loss.item()
            if is_master:
                iters = int(max_iter_in_epoch * (epoch - 1) + idx)
                writer.add_scalar('Train/Loss', loss.item() / len(images), iters)
                for loss_id in loss_keys:
                    writer.add_scalar(f'Train/Loss/{loss_id}', losses[f'{loss_id}'].item() / len(images), iters)

            # for name, param in model.mlp_edge.named_parameters():
            #     if param.requires_grad:
            #         print(param.grad)
            #         # writer.add_histogram(name + '_grad', param.grad, iters)
            
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

    model.train()

    total_loss = 0
    with tqdm(data_loader, unit="batch") as tepoch:
        max_iter_in_epoch = len(tepoch)
        for idx, batch in enumerate(tepoch):
            tepoch.set_description(f"Val: {epoch}")
            images, seg, nodes, edges, pts_labels = batch

            images = images.to(device)
            seg = seg.to(device)
            nodes = [node.to(device) for node in nodes]
            edges = [edge.to(device) for edge in edges]
            pts_labels = [pts_label.to(device) for pts_label in pts_labels]

            targets = {'nodes': nodes, 'edges': edges, 'segs':seg, 'pts_labels':pts_labels}
            wh = torch.ones(1,2).to(device) * 0.05
            targets['labels'] = [torch.zeros(len(x)).long().to(device) for x in targets['nodes']]
            targets['boxes'] = [torch.cat([x, wh.repeat(len(x), 1)], dim=-1) for x in targets['nodes']]
            targets_converted = [{key: value for key, value in zip(targets.keys(), values)} for values in zip(*targets.values())]

            with torch.no_grad():
                h, out, srcs = model(images, targets=targets_converted)

            losses = loss_fn(h, out, targets)
            loss = losses['total']
            total_loss += loss.item()

            loss_keys = list(losses.keys())
            loss_keys.remove('total')
            if is_master:
                iters = int(max_iter_in_epoch * (epoch - 1) / val_interval + idx)
                writer.add_scalar('Val/Loss', loss.item() / len(images), iters)
                for loss_id in loss_keys:
                    writer.add_scalar(f'Val/Loss/{loss_id}', losses[f'{loss_id}'].item() / len(images), iters)

            tepoch.set_postfix(loss=loss.item() / len(images))

    return total_loss / len(data_loader)


def save_checkpoint(model, optimizer, epoch, config):
    """
    Save a checkpoint of the training process.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epoch (int): The current epoch number.
        checkpoint_path (str): The file path to save the checkpoint.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }

    # If using DDP, save the original model wrapped inside DDP
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        checkpoint['model_state_dict'] = model.module.state_dict()

    savedir = os.path.join(config.TRAIN.SAVE_PATH, "runs", '%s_%d' % (config.log.exp_name, config.DATA.SEED), 'models')
    os.makedirs(savedir, exist_ok=True)
    checkpoint_path = os.path.join(savedir, f'epochs_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)