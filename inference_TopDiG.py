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
from scipy.optimize import linear_sum_assignment
import networkx as nx
import geopandas as gpd
from shapely.geometry import Polygon
import rasterio
from rasterio.transform import from_origin
from sklearn.metrics import f1_score, accuracy_score, jaccard_score


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

def scores_to_permutations(scores): # 인퍼런스용 함수
    """
    Input a batched array of scores and returns the hungarian optimized 
    permutation matrices.
    """
    B, N, N = scores.shape

    scores = scores.detach().cpu().numpy()
    perm = np.zeros_like(scores)
    for b in range(B):
        # sinkhorn 알고리즘은 비공개된 트레인 파일에 있을 듯
        # 반복적인 노멀라이제이션(100번)을 위해 싱크홀 알고리즘 사용
        # 인퍼런스시에 헝가리안 알고리즘으로 linear sum assignment result 뽑는다
        r, c = linear_sum_assignment(-scores[b]) # 점수가 높을 수록 페어일 확률이 높으므로 -를 붙여서 최소 찾는 문제로 바꾼다.
        perm[b,r,c] = 1 # 헝가리안 알고리즘이 찾은 칸은 1로 아니면 0인 permutation matrix (B N N) 만든다
    return torch.tensor(perm) # 텐서로 바꿔주기

def cycles_to_geodf(G, cycles):
    polygons = []
    for cycle in cycles:
        if len(cycle) > 2:  # Polygon needs at least 3 points
            points = [G.nodes[node]['pos'] for node in cycle]  # Assuming 'pos' contains coordinates
            polygons.append(Polygon(points))
    
    return gpd.GeoDataFrame(geometry=polygons)

def compute_pixel(pred_nodes, pred_edges, gt_nodes, gt_edges):
    pred_graph = nx.DiGraph()
    for i in range(len(pred_nodes)):
        pred_graph.add_node(i)
    pred_graph.add_edges_from(pred_edges)
    pos_pred = {i: (node[1], 320-node[0]) for i, node in enumerate(pred_nodes)}
    for node, position in pos_pred.items():
        pred_graph.nodes[node]['pos'] = position
    
    gt_graph = nx.DiGraph()
    for i in range(len(gt_nodes)):
        gt_graph.add_node(i)
    gt_graph.add_edges_from(gt_edges)
    pos_gt = {i: (node[1], 320-node[0]) for i, node in enumerate(gt_nodes)}
    for node, position in pos_gt.items():
        gt_graph.nodes[node]['pos'] = position

    # Convert cycles to a GeoDataFrame
    cycles_pred = list(nx.simple_cycles(pred_graph))
    cycles_gt = list(nx.simple_cycles(gt_graph))
    geo_df_pred = cycles_to_geodf(pred_graph, cycles_pred)
    geo_df_gt = cycles_to_geodf(gt_graph, cycles_gt)

    pixel_size = 1
    x_min, y_min, x_max, y_max = 0, 0, 320, 320
    width = int((x_max - x_min) / pixel_size)
    height = int((y_max - y_min) / pixel_size)

    # Create a transform (assumes north-up and no rotation)
    transform = from_origin(x_min, y_max, pixel_size, pixel_size)

    # Convert the GeoDataFrame to a raster array
    shapes = ((geom, 1) for geom in geo_df_pred.geometry)
    rasterized_pred = rasterio.features.rasterize(shapes=shapes, out_shape=(height, width), transform=transform)

    shapes = ((geom, 1) for geom in geo_df_gt.geometry)
    rasterized_gt = rasterio.features.rasterize(shapes=shapes, out_shape=(height, width), transform=transform)

    rasterized_pred, rasterized_gt = rasterized_pred.flatten(), rasterized_gt.flatten()

    accuracy = accuracy_score(rasterized_pred, rasterized_gt)
    f1 = f1_score(rasterized_pred, rasterized_gt, average='macro')  # 'macro' for unweighted mean across classes
    miou = jaccard_score(rasterized_pred, rasterized_gt, average='macro')
    building_iou = jaccard_score(rasterized_pred, rasterized_gt)

    return accuracy, f1, miou, building_iou


if __name__ == "__main__":
    dir_name_ = 'epochs_'
    for i in range(1, 2):
        # TODO 경로 변수 명령어 인자로 받게 수정하기
        dir_name = dir_name_ + str(20)
        config_file = "/nas/tsgil/TopDiG_train/runs/baseline_TopDiG_train_epoch20_sinkhorn_K-N_10/config.yaml"
        ckpt_path = f"/nas/tsgil/TopDiG_train/runs/baseline_TopDiG_train_epoch20_sinkhorn_K-N_10/models/{dir_name}.pth"
        show_dir = f'/nas/tsgil/gil/infer_TopDiG/exp_sinkhorn_K_N/{dir_name}'
        metric_file = '/nas/tsgil/gil/infer_TopDiG/exp_sinkhorn_K_N/metrics_log.txt'

        with open(metric_file, "a") as file:
            file.write(f'CONFIG: {config_file}\n\n')

        with open(config_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        config = dict2obj(config)

        dataset = CrowdAI(
            images_directory='/nas/tsgil/dataset/Inria_building/cocostyle_inria_test/images',
            annotations_path='/nas/tsgil/dataset/Inria_building/cocostyle_inria_test/annotation.json'
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
        
        device = torch.device("cuda")
        model = model.to(device)
        model.train()

        checkpoint = torch.load(ckpt_path, map_location='cpu') # 학습한 TopDiG 불러오기
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))

        if not os.path.isdir(show_dir):
            os.makedirs(show_dir)

        pixel_results = {'acc': [], 'f1': [], 'miou':[], 'building_iou':[]}
        iteration = 0
        for idx, (images, heatmaps, nodes, edges) in enumerate(tqdm(val_loader, total=len(val_loader))):
            iteration = iteration+1
            # if iteration == 2:
            #     break
            images = images.cuda()

            with torch.no_grad():
                out = model(images)
                out_nodes = (out['pred_nodes']*320).detach().cpu().numpy()
                out_heatmaps = out['pred_heatmaps']
            scores = out['scores1'].sigmoid() + out['scores2'].transpose(1,2).sigmoid()
            permu = scores_to_permutations(scores)

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
                axes[1, 1].scatter(out_nodes[i][:, 1], out_nodes[i][:, 0], color='g')
                mat = permu[i].numpy()
                pred_edges = []
                for j in range(len(mat)):
                    for k in range(len(mat)):
                        if mat[j][k] == 1:
                            if j != k:
                                pred_edges.append((j,k))
                for x, _ in pred_edges:
                    plt.scatter(out_nodes[i][x][1], out_nodes[i][x][0], color='b')
                for e in pred_edges:
                    connect = np.stack([out_nodes[i][e[0]], out_nodes[i][e[1]]], axis=0)
                    plt.plot(connect[:,1], connect[:,0])
                    plt.annotate("", xy=out_nodes[i][e[0]][::-1], xytext=out_nodes[i][e[1]][::-1], 
                                arrowprops=dict(arrowstyle="->", lw=1.5, color='r'))
                axes[1, 1].set_title('pred_image')

                # 매트릭 계산
                acc, f1, miou, b_iou = compute_pixel(out_nodes[i], pred_edges, nodes_i, edges_i)
                pixel_results['acc'].append(acc)
                pixel_results['f1'].append(f1)
                pixel_results['miou'].append(miou)
                pixel_results['building_iou'].append(b_iou)
                # print(f'{start_idx+i}.png acc: {acc:.5f}, f1: {f1:.5f}, miou: {miou:.5f}, b_iou: {b_iou:.5f}')
                plt.suptitle(f'Accuracy: {acc:.3f}, F1 Score: {f1:.3f}, mIoU: {miou:.3f}, building_iou: {b_iou:.3f}')
                with open(metric_file, "a") as file:
                    file.write(f'{start_idx+i}.png acc: {acc}, f1: {f1}, miou: {miou}, b_iou: {b_iou}\n')

                # 서브플롯 간 간격 조절 (선택 사항)
                plt.tight_layout()

                save_path = f'{show_dir}/{start_idx+i}.png'
                plt.savefig(save_path)
                plt.close()
        with open(metric_file, "a") as file:
            file.write(f"\nTotal average metrics\n")
            for key, result in pixel_results.items():
                pixel_array=np.array(result)
                print(f'{key}: {pixel_array.mean(0)}')
                file.write(f'{key}: {pixel_array.mean(0)}\n')
            file.write("------------\n")

