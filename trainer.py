import os
from tqdm import tqdm

import torch

from inference import relation_infer


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
            images, seg, nodes, edges = batch

            images = images.to(device)
            seg = seg.to(device)
            nodes = [node.to(device) for node in nodes]
            edges = [edge.to(device) for edge in edges]

            optimizer.zero_grad()

            targets = {'nodes': nodes, 'edges': edges, 'segs':seg}
            wh = torch.ones(1,2).to(device) * 0.05
            targets['labels'] = [torch.zeros(len(x)).long().to(device) for x in targets['nodes']]
            targets['boxes'] = [torch.cat([x, wh.repeat(len(x), 1)], dim=-1) for x in targets['nodes']]
            targets_converted = [{key: value for key, value in zip(targets.keys(), values)} for values in zip(*targets.values())]

            h, out, srcs = model(images, targets=targets_converted)

            losses = loss_fn(h, out, {'nodes': nodes, 'edges': edges, 'segs':seg})
            loss = losses['total']

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if is_master:
                iters = int(max_iter_in_epoch * (epoch - 1) + idx)
                writer.add_scalar('Train/Loss', loss.item() / len(images), iters)
                writer.add_scalar('Train/Loss/class', losses['class'].item() / len(images), iters)
                writer.add_scalar('Train/Loss/nodes', losses['nodes'].item() / len(images), iters)
                writer.add_scalar('Train/Loss/boxes', losses['boxes'].item() / len(images), iters)

                # writer.add_scalar('Train/Loss/edges', losses['edges'].item(), iters)

                if loss_fn.seg:
                    writer.add_scalar('Train/Loss/segs', losses['segs'].item() / len(images), iters)

                if loss_fn.two_stage:
                    for i, l in enumerate(['class', 'boxes', 'cards', 'nodes']):
                        writer.add_scalar(f'Train/Loss/enc/{l}', losses[f'{l}_enc'].item() / len(images), iters)
                        for j in range(3):
                            writer.add_scalar(f'Train/Loss/aux/{j}/{l}', losses[f'{l}_aux_{j}'].item() / len(images), iters)

            # for name, param in model.aux_fpn_head.named_parameters():
            #     if param.requires_grad:
            #         # print(param.grad)
            #         writer.add_histogram(name + '_grad', param.grad, iters)
            
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
            images, seg, nodes, edges = batch

            images = images.to(device)
            seg = seg.to(device)
            nodes = [node.to(device) for node in nodes]
            edges = [edge.to(device) for edge in edges]

            targets = {'nodes': nodes, 'edges': edges, 'segs':seg}
            wh = torch.ones(1,2).to(device) * 0.05
            targets['labels'] = [torch.zeros(len(x)).long().to(device) for x in targets['nodes']]
            targets['boxes'] = [torch.cat([x, wh.repeat(len(x), 1)], dim=-1) for x in targets['nodes']]
            targets_converted = [{key: value for key, value in zip(targets.keys(), values)} for values in zip(*targets.values())]

            h, out, srcs = model(images, targets=targets_converted)

            losses = loss_fn(h, out, {'nodes': nodes, 'edges': edges, 'segs':seg})
            loss = losses['total']
            total_loss += loss.item()

            if is_master:
                iters = int(max_iter_in_epoch * (epoch - 1) / val_interval + idx)
                writer.add_scalar('Val/Loss', loss.item() / len(images), iters)
                writer.add_scalar('Val/Loss/class', losses['class'].item() / len(images), iters)
                writer.add_scalar('Val/Loss/nodes', losses['nodes'].item() / len(images), iters)
                writer.add_scalar('Val/Loss/boxes', losses['boxes'].item() / len(images), iters)

                if config.MODEL.DECODER.RLN_TOKEN > 0:
                    writer.add_scalar('Val/Loss/edges', losses['edges'].item() / len(images), iters)

                if loss_fn.seg:
                    writer.add_scalar('Val/Loss/segs', losses['segs'].item() / len(images), iters)

                if loss_fn.two_stage:
                    for i, l in enumerate(['class', 'boxes', 'cards', 'nodes']):
                        writer.add_scalar(f'Val/Loss/enc/{l}', losses[f'{l}_enc'].item() / len(images), iters)
                        for j in range(3):
                            writer.add_scalar(f'Val/Loss/aux/{j}/{l}', losses[f'{l}_aux_{j}'].item() / len(images), iters)

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
    # Prepare the checkpoint dictionary
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }

    # relation_embed_checkpoint = {
    #     'epoch': epoch,
    #     'model_state_dict': relation_embed.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict()
    # }

    # If using DDP, save the original model wrapped inside DDP
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        checkpoint['model_state_dict'] = model.module.state_dict()

    savedir = os.path.join(config.TRAIN.SAVE_PATH, "runs", '%s_%d' % (config.log.exp_name, config.DATA.SEED), 'models')
    os.makedirs(savedir, exist_ok=True)
    checkpoint_path = os.path.join(savedir, f'epochs_{epoch}.pth')
    # relation_embed_checkpoint_path = os.path.join(savedir, f'relation_embed_epochs_{epoch}.pth')

    # Save the checkpoint
    torch.save(checkpoint, checkpoint_path)
    # torch.save(relation_embed_checkpoint, relation_embed_checkpoint_path)