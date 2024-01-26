import argparse
import json
import os

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_inria import build_inria_data
from models.backbone_R2U_Net import build_backbone


class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)


def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)


def image_graph_collate_road_network(batch):
    images = torch.stack([item[0] for item in batch], 0).contiguous()
    seg = torch.stack([item[1] for item in batch], 0).contiguous()
    points = [item[2] for item in batch]
    edges = [item[3] for item in batch]
    return [images, seg, points, edges]


def parse_args():
    parser = argparse.ArgumentParser(
        description='relationformer inference model')
    parser.add_argument('config', type=str, help='test config file path')
    parser.add_argument('checkpoint', type=str, help='checkpoint file')
    parser.add_argument('--show-dir', type=str, required=True, help='savedir')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    dir_name_ = 'epochs_'
    for i in range(1, 14):
        dir_name = dir_name_ + str(i * 10)
        config_file = 'configs/inria_pretrain.yaml'
        ckpt_path = f'/nas/tsgil/relationformer/work_dirs/R2U_Net_pretrain/runs/baseline_R2U_Net_pretrain_epoch200_5e-6_10/models/{dir_name}.pth'  # noqa
        show_dir = f'/nas/tsgil/gil/infer_R2U/exp11/{dir_name}'

        # args = parse_args()
        # config_file = args.config
        # ckpt_path = args.checkpoint
        # show_dir = args.show_dir

        with open(config_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        config = dict2obj(config)

        val_ds = build_inria_data(  # 이 함수는 폴더에서 셔플해서 ds 생성함
            config, mode='test')

        val_loader = DataLoader(
            val_ds,
            batch_size=config.DATA.BATCH_SIZE,
            shuffle=False,
            num_workers=config.DATA.NUM_WORKERS,
            # collate_fn=image_graph_collate_road_network,
            pin_memory=True,
        )
        print(len(val_ds),
              len(val_loader))  # 테스트셋 전체 지역 패치: 25036개, 훈련셋 지역 21: 631개

        model = build_backbone(config)
        device = torch.device('cuda')
        model = model.to(device)

        checkpoint = torch.load(ckpt_path, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(
            checkpoint['model_state_dict'], strict=False)
        unexpected_keys = [
            k for k in unexpected_keys
            if not (k.endswith('total_params') or k.endswith('total_ops'))
        ]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))

        # model.eval()
        model.train()

        if not os.path.isdir(show_dir):
            os.makedirs(show_dir)
            os.makedirs(show_dir + '/img')
            os.makedirs(show_dir + '/gt_heatmap')
            os.makedirs(show_dir + '/pred_heatmap')

        iteration = 0
        for idx, (images, seg,
                  heatmap) in enumerate(tqdm(val_loader,
                                             total=len(val_loader))):
            iteration = iteration + 1
            if iteration == 4:  # 1~3 이터레이션만 돌리기 총 3400 iters
                break
            images = images.cuda()

            seg = seg.cuda()
            with torch.no_grad():
                out = model(images)[1].detach()
            out = torch.sigmoid(out)  # BCEwithLogit을 사용했기 때문

            start_idx = idx * config.DATA.BATCH_SIZE
            for i in range(len(images)):  # 0~31, 배치 사이즈만큼 이미지 처리
                # img of patch
                # NumPy 배열을 PIL 이미지로 변환
                image = images[i].cpu().numpy()
                min_value = np.min(image)
                max_value = np.max(image)
                image = (image - min_value) / (max_value - min_value)
                image = (image * 255).astype('uint8')
                image = image.transpose(1, 2, 0)
                image = Image.fromarray(image, 'RGB')
                # 이미지 파일로 저장할 경로 지정
                save_path = f'{show_dir}/img/{start_idx+i}.png'
                # 이미지 파일로 저장
                image.save(save_path)
                # gt_heatmap
                image = heatmap[i].cpu().numpy()
                # image = image+0.5
                image = (image * 255).astype('uint8')
                image = np.squeeze(image, axis=0)
                image = Image.fromarray(image, 'L')
                save_path = f'{show_dir}/gt_heatmap/{start_idx+i}.png'
                image.save(save_path)
                # pred_heatmap
                image = out[i].cpu().numpy()
                image = (image * 255).astype('uint8')
                image = np.squeeze(image, axis=0)
                image = Image.fromarray(image, 'L')
                save_path = f'{show_dir}/pred_heatmap/{start_idx+i}.png'
                image.save(save_path)
