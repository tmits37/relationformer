import os
from shutil import copyfile
import yaml
import json
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader_cocostyle import CrowdAI, image_graph_collate_road_network_coco
from models.TopDiG import build_TopDiG
from models.DGS import HungarianMatcher
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from losses_TopDiG import SetCriterion
import torch.multiprocessing
from dataloader_cocostyle import build_inria_coco_data
from trainer_TopDiG import train_epoch, validate_epoch, save_checkpoint
from dataloader_cocostyle_road import build_road_coco_data

os.environ['TORCH_DISTRIBUTED_DEBUG']='DETAIL'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        default=None,
                        help='config file (.yml) containing the hyper-parameters for training. '
                            'If None, use the nnU-Net config. See /config for examples.')
    parser.add_argument('--resume', default=None, help='checkpoint of the last epoch of the model')
    # parser.add_argument('--seg_net', default=None, help='checkpoint of the segmentation model')
    parser.add_argument('--device', default='cuda',
                            help='device to use for training')
    parser.add_argument('--dataset', default='building', 
                        help='building_dataset')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        print("Setting the Local Rank")
        os.environ['LOCAL_RANK'] = str(args.local_rank)
        
    return args


def init_for_distributed(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        print(os.environ['RANK'], os.environ['LOCAL_RANK'], os.environ['WORLD_SIZE'])
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.distributed = True
        os.environ['TORCH_DISTRIBUTED_DEBUG']='DETAIL'
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.local_rank = args.rank % torch.cuda.device_count()
    else: # --rank 디폴트 값(0)이면 else문 진행
        print('Not using distributed mode')
        args.distributed = False
        args.local_rank = 0
        return None

    # 1. setting for distributed training
    # opts.rank = rank
    # local_gpu_id = int(opts.gpu_ids[opts.rank])
    # torch.cuda.set_device(local_gpu_id)
    # if opts.rank is not None:
    #     print("Use GPU: {} for training".format(local_gpu_id))

    torch.cuda.set_device(args.local_rank)
    args.dist_backend = 'nccl' # nvcc
    print('| distributed init (rank {}): {}'.format(
        args.rank, 'env://'), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, 
                                         init_method="env://127.0.0.1:29500",
                                         world_size=args.world_size, 
                                         rank=args.rank)
    torch.distributed.barrier()
    # setup_for_distributed(args.rank == 0)
    return None


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)
        

def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)


def match_name_keywords(name, name_keywords):
    for keyword in name_keywords:
        if keyword == name.split('.')[0]:
            return True
    else: return False


def main(args):
    # Load the config files
    with open(args.config) as f:
        print('\n*** Config file')
        print(args.config)
        config = yaml.load(f, Loader=yaml.FullLoader)
        print(config['log']['message'])
    config = dict2obj(config)
    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.cuda_visible_device))

    exp_path = os.path.join(config.TRAIN.SAVE_PATH, "runs", '%s_%d' % (config.log.exp_name, config.DATA.SEED))
    if os.path.exists(exp_path) and args.resume == None:
        print('ERROR: Experiment folder exist, please change exp name in config file')
    else:
        try:
            os.makedirs(exp_path)
            copyfile(args.config, os.path.join(exp_path, "config.yaml"))
        except:
            pass

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.multiprocessing.set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_descriptor')

    init_for_distributed(args)
    device = torch.device(args.device)

### Setting the dataset
    if args.dataset == 'road':
        print("Loading the road dataset")
        train_ds = build_road_coco_data(config, mode='train')
        # val_ds = build_road_coco_data(config, mode='test')
    else:
        train_ds = build_inria_coco_data(config, mode='train')
        # val_ds = build_inria_coco_data(config, mode='test')

    if args.distributed:
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        # val_sampler = DistributedSampler(val_ds, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_ds)
        # val_sampler = torch.utils.data.SequentialSampler(val_ds)


    train_loader = DataLoader(
        train_ds,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        sampler=train_sampler,
        collate_fn=image_graph_collate_road_network_coco,
        pin_memory=True,
        drop_last=True
        )

    # val_loader = DataLoader(
    #     val_ds,
    #     batch_size=int(config.DATA.BATCH_SIZE / args.world_size),
    #     num_workers=int(config.DATA.NUM_WORKERS / args.world_size),
    #     sampler=val_sampler,
    #     collate_fn=image_graph_collate_road_network_coco,
    #     pin_memory=True,
    #     drop_last=True # False면 마지막 에폭에서 아래 에러 발생
    #     # RuntimeError: Sizes of tensors must match except in dimension 2.
    #     # Expected size 4 but got size ot size 3 for tensor number 1 in the list.
    #     )

    ### Setting the model
    net = build_TopDiG(config)
    matcher = HungarianMatcher()

    if args.distributed:
        device = torch.device(f"cuda:{args.rank}")

        net = DistributedDataParallel(net.cuda(args.local_rank), 
                                      device_ids=[args.local_rank],
                                      broadcast_buffers=False,
                                      find_unused_parameters=True
                                      )
    else:
        net = net.to(device)
        matcher = matcher.to(device)

    # Loss = L_node + L_graph
    loss = SetCriterion(config, matcher, net, distributed=args.distributed).cuda(args.local_rank)

    ### Setting optimizer
    param_dicts = [
        { # pretrained R2U-Net의 러닝 레이트, 112가지 파라미터
            "params": 
                [p for n, p in net.named_parameters()
                 if match_name_keywords(n, ['backbone']) and p.requires_grad],
            "lr": float(config.TRAIN.LR_BACKBONE)
        },
        { # DiG_generator의 러닝 레이트, 58가지 파라미터
            "params":
                [p for n, p in net.named_parameters()
                 if match_name_keywords(n, ['decoder']) and p.requires_grad],
            "lr": float(config.TRAIN.LR)
        },
    ]

    # 5.3 섹션에 Adam 사용한다고 적혀있음 TopDiG
    # 근데 요즘 트렌드는 AdamW
    optimizer = torch.optim.AdamW(
        param_dicts, lr=float(config.TRAIN.LR), weight_decay=float(config.TRAIN.WEIGHT_DECAY)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.TRAIN.LR_DROP)
    
    n_epochs = config.TRAIN.EPOCHS
    last_epoch = 1
    if args.resume: # 학습 재개 코드
        checkpoint = torch.load(args.resume)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['schedulaer_state_dict'])
        scheduler.step_size = config.TRAIN.LR_DROP
        last_epoch = scheduler.last_epoch + 1

        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(args.local_rank)

        checkpoint = None

    print("Check local rank or not")
    print("=======================")
    print("Local rank:", args.local_rank) # 0

    if args.local_rank == 0:
        is_master = True
    else:
        is_master = False


    writer = SummaryWriter(
        log_dir=os.path.join(config.TRAIN.SAVE_PATH, "runs", '%s_%d' % (config.log.exp_name, config.DATA.SEED)),
    ) if is_master else None

    for epoch in range(last_epoch, n_epochs+1):
        train_loss = train_epoch(
            net,
            data_loader=train_loader,
            loss_fn=loss, 
            optimizer=optimizer, 
            device=device, 
            epoch=epoch, 
            writer=writer, 
            is_master=is_master)
        if is_master:
            scheduler.step()
            current_lr = scheduler.get_last_lr()
            print(f"Epoch {epoch}, Training Loss: {train_loss}")
            print(f"Epoch {epoch}: Current learning rate = {current_lr}")

        if is_master and (epoch % config.TRAIN.VAL_INTERVAL == 0):
            save_checkpoint(
                net,
                optimizer,
                scheduler,
                epoch, 
                config)
            # validate_epoch(
            #     net,
            #     config=config,
            #     data_loader=val_loader,
            #     loss_fn=loss, 
            #     device=device, 
            #     epoch=epoch, 
            #     val_interval=config.TRAIN.VAL_INTERVAL,
            #     writer=writer, 
            #     is_master=is_master)

    if is_master and writer:
        writer.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)