import os
import yaml
import sys
sys.path.append("..")
import json
import numpy as np
import torch
from dataloader_cocostyle import CrowdAI, image_graph_collate_road_network_coco
from models.TopDiG import build_TopDiG
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)
        
def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)

def min_max_normalize(image, percentile, nodata=-1.):
    image = image.astype('float32')
    mask = np.mean(image, axis=2) != nodata * image.shape[2]

    percent_min = np.percentile(image, percentile, axis=(0, 1))
    percent_max = np.percentile(image, 100-percentile, axis=(0, 1))

    if image.shape[1] * image.shape[0] - np.sum(mask) > 0:
        mdata = np.ma.masked_equal(image, nodata, copy=False)
        mdata = np.ma.filled(mdata, np.nan)
        percent_min = np.nanpercentile(mdata, percentile, axis=(0, 1))

    norm = (image-percent_min) / (percent_max - percent_min)
    norm[norm < 0] = 0
    norm[norm > 1] = 1
    norm = (norm * 255).astype('uint8') * mask[:, :, np.newaxis]

    return norm

if __name__ == "__main__":
    dir_name_ = 'epochs_'
    for i in range(1, 11):
        dir_name = dir_name_ + str(i*2)
        config_file = "/nas/tsgil/relationformer/configs/TopDiG_train.yaml"
        ckpt_path = f"/nas/tsgil/relationformer/work_dirs/TopDiG_train/runs/baseline_TopDiG_train_epoch20_ptm_10/models/{dir_name}.pth"
        show_dir = f'/nas/tsgil/gil/infer_TopDiG/exp1/{dir_name}'

        with open(config_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        config = dict2obj(config)

        dataset = CrowdAI(
            images_directory='/nas/tsgil/dataset/Inria_building/cocostyle/images',
            annotations_path='/nas/tsgil/dataset/Inria_building/cocostyle/annotation.json'
        )
        val_sampler = torch.utils.data.SequentialSampler(dataset)

        val_loader = DataLoader(
            dataset,
            batch_size=config.DATA.BATCH_SIZE,
            num_workers=config.DATA.NUM_WORKERS,
            sampler=val_sampler,
            collate_fn=image_graph_collate_road_network_coco,
            pin_memory=True,
            drop_last=True
            )
        print(len(dataset), len(val_loader))

        model = build_TopDiG(config)
        # print(model.backbone.detectionBranch.conv[1].running_mean) # 배치 정규화에 대한 learnable 파라미터
        
        device = torch.device("cuda")
        model = model.to(device)
        model.train() # eval() 시에는 모델 아웃풋이 안좋음
        # model.eval() # eval() 시에는 모델 아웃풋이 안좋음

        if not os.path.isdir(show_dir):
            os.makedirs(show_dir)

        iteration = 0
        for idx, (images, heatmaps, nodes, edges) in enumerate(tqdm(val_loader, total=len(val_loader))):
            iteration = iteration+1
            if iteration == 3:
                break
            images = images.cuda()

            with torch.no_grad():
                out = model(images)
                # print(model.backbone.detectionBranch.conv[1].running_mean) # 배치 정규화에 대한 learnable 파라미터
                out_nodes = model.v[1].detach().cpu().numpy()
                out_heatmaps = model.h

            start_idx = idx*config.DATA.BATCH_SIZE
            for i in range(len(images)):
                fig, axes = plt.subplots(2, 2, figsize=(12, 12))

                # 첫 번째 서브플롯 - gt_heatmap
                gt_heatmap = heatmaps[i].detach().cpu().numpy().transpose(1,2,0)
                axes[0, 0].imshow(min_max_normalize(gt_heatmap, 0.5))
                # axes[0, 0].imshow(gt_heatmap)
                axes[0, 0].set_title('gt_heatmap')

                # 두 번째 서브플롯 - gt_image
                image = images[i].detach().cpu().numpy().transpose(1,2,0)
                axes[0, 1].imshow(min_max_normalize(image, 0.5))
                nodes_i = nodes[i].cpu().numpy() * image.shape[0]
                nodes_i = nodes_i.astype('int64')
                axes[0, 1].scatter(nodes_i[:, 1], nodes_i[:, 0], color='r')
                edges_i = edges[i].cpu().numpy()
                for e in edges_i:
                    connect = np.stack([nodes_i[e[0]], nodes_i[e[1]]], axis=0)
                    axes[0, 1].plot(connect[:,1], connect[:,0])
                axes[0, 1].set_title('gt_image')

                # 세 번째 서브플롯 - pred_heatmap
                pred_heatmap = out_heatmaps[i].detach().cpu().numpy().transpose(1,2,0)
                axes[1, 0].imshow(min_max_normalize(pred_heatmap, 0.5))
                axes[1, 0].set_title('pred_heatmap')

                # 네 번째 서브플롯 - pred_image
                axes[1, 1].imshow(min_max_normalize(image, 0.5))
                axes[1, 1].scatter(out_nodes[i][:, 1], out_nodes[i][:, 0], color='r')
                axes[1, 1].set_title('pred_image')

                # 서브플롯 간 간격 조절 (선택 사항)
                plt.tight_layout()
                
                save_path = f'{show_dir}/{start_idx+i}.png'
                plt.savefig(save_path)
                plt.close()
