import os
from tqdm import tqdm

import torch

from inference import relation_infer


def train_epoch(model,
                relation_embed,
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
            h, out, srcs = model(images)

            losses = loss_fn(h, out, {'nodes': nodes, 'edges': edges, 'segs':seg})
            loss = losses['total']

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if is_master:
                iters = max_iter_in_epoch * epoch + idx
                writer.add_scalar('Train/Loss', loss.item(), iters)
                writer.add_scalar('Train/Loss/class', losses['class'].item(), iters)
                writer.add_scalar('Train/Loss/nodes', losses['nodes'].item(), iters)
                writer.add_scalar('Train/Loss/boxes', losses['boxes'].item(), iters)

                # writer.add_scalar('Train/Loss/edges', losses['edges'].item(), iters)

                if loss_fn.seg:
                    writer.add_scalar('Train/Loss/segs', losses['segs'].item(), iters)

            # for name, param in model.aux_fpn_head.named_parameters():
            #     if param.requires_grad:
            #         # print(param.grad)
            #         writer.add_histogram(name + '_grad', param.grad, iters)
            
            tepoch.set_postfix(loss=loss.item())

    return total_loss / len(data_loader)


def validate_epoch(
    model, 
    relation_embed,
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
            images, seg, nodes, edges = batch

            images = images.to(device)
            seg = seg.to(device)
            nodes = [node.to(device) for node in nodes]
            edges = [edge.to(device) for edge in edges]

            h, out, srcs = model(images)
            # pred_nodes, pred_edges = relation_infer(
            #     h.detach(), 
            #     out, 
            #     relation_embed, 
            #     config.MODEL.DECODER.OBJ_TOKEN, 
            #     config.MODEL.DECODER.RLN_TOKEN
            # )
            losses = loss_fn(h, out, {'nodes': nodes, 'edges': edges, 'segs':seg})
            loss = losses['total']
            total_loss += loss.item()

            if is_master:
                iters = max_iter_in_epoch * epoch + idx
                writer.add_scalar('Val/Loss', loss.item(), iters)
                writer.add_scalar('Val/Loss/class', losses['class'].item(), iters)
                writer.add_scalar('Val/Loss/nodes', losses['nodes'].item(), iters)
                writer.add_scalar('Val/Loss/boxes', losses['boxes'].item(), iters)

                if config.MODEL.DECODER.RLN_TOKEN > 0:
                    writer.add_scalar('Val/Loss/edges', losses['edges'].item(), iters)

                if loss_fn.seg:
                    writer.add_scalar('Val/Loss/segs', losses['segs'].item(), iters)
            tepoch.set_postfix(loss=loss.item())

    return total_loss / len(data_loader)


def save_checkpoint(model, relation_embed, optimizer, epoch, config):
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