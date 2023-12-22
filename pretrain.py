import os
from shutil import copyfile
import yaml
import json
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_inria import build_inria_data
from dataloader_cocostyle import CrowdAI, image_graph_collate_road_network_coco
from models.backbone_R2U_Net import R2U_Net
from utils import image_graph_collate_road_network
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

import torch.multiprocessing

from pretrainer import train_epoch, validate_epoch, save_checkpoint
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
    parser.add_argument('--cuda_visible_device', nargs='*', type=int, default=[0,1],
                            help='list of index where skip conn will be made')
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
    else:
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

class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)

def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)

def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out
# 배치 사이즈로 나누다 보면 마지막 이터레이션에 빈 공간이 생기는 걸 채워줌
def image_graph_collate_road_network(batch):
    images = torch.stack([item[0] for item in batch], 0).contiguous()
    seg = torch.stack([item[1] for item in batch], 0).contiguous()
    points = [item[2] for item in batch]
    edges = [item[3] for item in batch]
    return [images, seg, points, edges]

def main(args):
    # Load the config files
    with open(args.config) as f:
        print('\n*** Config file')
        print(args.config)
        config = yaml.load(f, Loader=yaml.FullLoader)
        print(config['log']['message'])
    config = dict2obj(config)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.cuda_visible_device))

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
    torch.multiprocessing.set_sharing_strategy('file_system')
    # device = torch.device("cuda") if args.device=='cuda' else torch.device("cpu")

    # gpu 2개 사용할 경우, world_size = 2, rank = 0 and 1 으로 두번 실행되는것을 관찰할 수 있다.
    init_for_distributed(args)

    # 근데 이 라인은 1번만 실행된다.
    print(args.rank, args.local_rank)

    device = torch.device(args.device)
    # if args.distributed:
    #     args.rank = torch.distributed.get_rank()
    #     device = torch.device(f"cuda:{args.rank}")
    #     print(args.rank, device, args.local_rank) # 0, cuda:0, 0


    ### Setting the dataset
    # 데이터 셋을 데이터 폴더에서 불러온다
    # train_ds, val_ds = build_inria_data(config, mode='split')
    train_ds = build_inria_data(config)
    val_ds = build_inria_data(config, mode='test')
    # train_ds = CrowdAI(images_directory='/nas/tsgil/dataset/Inria_building/cocostyle/images',
    #                   annotations_path='/nas/tsgil/dataset/Inria_building/cocostyle/annotation.json')
    # val_ds = train_ds

    if args.distributed: # 랜덤하게 섞어서 샘플 뽑기
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False)
    else: # 순서대로 샘플 뽑기
        train_sampler = torch.utils.data.RandomSampler(train_ds)
        val_sampler = torch.utils.data.SequentialSampler(val_ds)

    # 데이터 셋을 지정한 배치로 나눠서 로드하기
    train_loader = DataLoader(
        train_ds,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        sampler=train_sampler,
        # collate_fn=image_graph_collate_road_network,
        pin_memory=True
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=int(config.DATA.BATCH_SIZE / args.world_size),
        num_workers=int(config.DATA.NUM_WORKERS / args.world_size),
        sampler=val_sampler,
        # collate_fn=image_graph_collate_road_network,
        pin_memory=True
        )
    # train_loader = torch.utils.data.DataLoader(train_ds, batch_size=6, collate_fn=image_graph_collate_road_network_coco)
    # val_loader = train_loader


    ### Setting the model
    net = R2U_Net() # .to(device), R2U Net 빌드하여 데이터 로더 투입
    # matcher = build_matcher(config) # R2U Net은 매쳐 필요 없음
    # relation_embed = build_relation_embed(config)

    if args.distributed:
        device = torch.device(f"cuda:{args.rank}")

        net = DistributedDataParallel(net.cuda(args.local_rank), 
                                      device_ids=[args.local_rank],
                                      broadcast_buffers=False,
                                      find_unused_parameters=True
                                      )
    else:
        net = net.to(device)
        # matcher = matcher.to(device)

    # if args.distributed:
    #     relation_embed = net.module.relation_embed
    # else:
    #     relation_embed = net.relation_embed

    # 이부분도 일단 프리트레인 할거기 때문에 MSE loss로 변경 토치에 존재
    # loss = SetCriterion(config, matcher, net, distributed=args.distributed).cuda(args.local_rank)
    # loss = nn.MSELoss()
    loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10]).cuda())

    ### Setting optimizer
    # param_dicts = [
    #     { # 기본 러닝 레이트 매개변수 집합, 모델의 대부분의 파라미터
    #         "params":
    #             [p for n, p in net.named_parameters()
    #              if not match_name_keywords(n, ["encoder.0"]) and not match_name_keywords(n, ['reference_points', 'sampling_offsets']) and p.requires_grad],
    #         "lr": float(config.TRAIN.LR)
    #     },
    #     { # 백본 러닝 레이트 매개변수 집합, encoder.0 이름을 갖는 모듈의 파라미터
    #         "params": [p for n, p in net.named_parameters() if match_name_keywords(n, ["encoder.0"]) and p.requires_grad],
    #         "lr": float(config.TRAIN.LR_BACKBONE)
    #     },
    #     { # 특정 부분 세밀하게 조정하기 위한 러닝 레이트 매개변수 집합, 레퍼런스 포인트, 샘플링 오프셋 모듈의 파라미터
    #         "params": [p for n, p in net.named_parameters() if match_name_keywords(n, ['reference_points', 'sampling_offsets']) and p.requires_grad],
    #         "lr": float(config.TRAIN.LR)*0.1
    #     }
    # ]
    param_dicts = [
        {
            "params": [p for name, p in net.named_parameters() if p.requires_grad],
            "lr": float(config.TRAIN.LR)
        }
    ]

    optimizer = torch.optim.AdamW(
        param_dicts, lr=float(config.TRAIN.LR), weight_decay=float(config.TRAIN.WEIGHT_DECAY)
    )
    # optimizer = torch.optim.Adam(
    #     param_dicts, weight_decay=float(config.TRAIN.WEIGHT_DECAY)
    # )

    # 논문에서 lr 스케줄러 안 쓴다고 함. 한번 사용해보기
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.TRAIN.LR_DROP, verbose=True)
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        last_epoch = scheduler.last_epoch
        scheduler.step_size = config.TRAIN.LR_DROP


    print("Check local rank or not")
    print("=======================")
    print(args.local_rank)

    if args.local_rank == 0:
        is_master = True
    else:
        is_master = False


    writer = SummaryWriter(
        log_dir=os.path.join(config.TRAIN.SAVE_PATH, "runs", '%s_%d' % (config.log.exp_name, config.DATA.SEED)),
    ) if is_master else None

    n_epochs = config.TRAIN.EPOCHS
    for epoch in range(1, n_epochs+1):
        train_loss = train_epoch(
            net,
            # relation_embed,
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
            validate_epoch(
                net, 
                # relation_embed,
                config=config,
                data_loader=val_loader, 
                loss_fn=loss, 
                device=device, 
                epoch=epoch, 
                writer=writer, 
                is_master=is_master)
        save_checkpoint(
            net, 
            # relation_embed,
            optimizer,
            scheduler,
            epoch, 
            config)

    if is_master and writer:
        writer.close()

if __name__ == '__main__':
    args = parse_args()
    torch.multiprocessing.set_sharing_strategy('file_system')
    main(args)
