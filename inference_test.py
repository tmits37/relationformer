import os
import yaml
import sys
sys.path.append("..")
import json
import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from skimage.morphology import skeletonize_3d
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import text, patheffects
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from monai.data import DataLoader
import torch
from torch import nn
import networkx as nx
from dataset_road_network import Sat2GraphDataLoader
from dataset_road_network import build_road_network_data
from evaluator import build_evaluator
from trainer import build_trainer
from models import build_model
from utils import image_graph_collate_road_network
from models.matcher import build_matcher
from inference import relation_infer
from losses import SetCriterion
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from tqdm import tqdm
import argparse


class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)
        
def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)


def plot_test_sample(image, points, edges):

    H, W = image.shape[0], image.shape[1]
    fig, ax = plt.subplots(figsize=(12,5), dpi=150)
    gridspec.GridSpec(1,5)
    ax = plt.subplot2grid((1,4), (0,0), colspan=2, rowspan=1)

    # Displaying the image
    ax.imshow(image*std+mean)
    ax.axis('off')
    
    ax = plt.subplot2grid((1,4), (0,2), colspan=2, rowspan=1)
    
    border_nodes = np.array([[0,0],[0,H],[W,H],[W,0]])
    border_edges = [(0,1),(1,2),(2,3),(3,0)]
    
    G1 = nx.Graph()
    G1.add_nodes_from(list(range(len(border_nodes))))
    coord_dict = {}
    tmp = [coord_dict.update({i:(pts[1],pts[0])}) for i, pts in enumerate(border_nodes)]
    for n, p in coord_dict.items():
        G1.nodes[n]['pos'] = p
    G1.add_edges_from(border_edges)
    
    pos = nx.get_node_attributes(G1,'pos')
    #pos = nx.rescale_layout_dict(pos)
    #pos = nx.circular_layout(G)
    nx.draw(G1, pos, ax=ax, node_size=1, node_color='darkgrey', edge_color='darkgrey', width=1.5, font_size=12, with_labels=False)

    
    G = nx.Graph()
    edges = [tuple(rel) for rel in edges]
    nodes = list(np.unique(np.array(edges)))
    coord_dict = {}
    tmp = [coord_dict.update({i:(W*pts[1],H- H*pts[0])}) for i, pts in enumerate(points)]
    G.add_nodes_from(list(range(len(points))))
    for n, p in coord_dict.items():
        G.nodes[n]['pos'] = p
    G.add_edges_from(edges)
    pos = nx.get_node_attributes(G,'pos')
    #pos = nx.rescale_layout_dict(pos)
    #pos = nx.circular_layout(G)
    nx.draw(G, pos, ax=ax, node_size=10, node_color='lightcoral', edge_color='mediumorchid', width=1.5, font_size=12, with_labels=False)
    # nx.draw_networkx_edge_labels(G, pos, ax=ax, font_size=12, label_pos=0.5, rotate=False)
    
    plt.show()
    

def plot_val_rel_sample(image, seg, points1, edges1, points2, edges2, attn_map=None, relative_coords=True):
    H, W = image.shape[0], image.shape[1]
    fig, ax = plt.subplots(1,4, figsize=(20,5), dpi=150)

    # Displaying the image, 인풋 이미지
    ax[0].imshow(image*std+mean)
    ax[0].axis('off')
    
    # 인풋 seg
    ax[1].imshow(seg)
    ax[1].axis('off')
    
    border_nodes = np.array([[1,1],[1,H-1],[W-1,H-1],[W-1,1]])
    border_edges = [(0,1),(1,2),(2,3),(3,0)]
    
    # 
    G1 = nx.Graph()
    G1.add_nodes_from(list(range(len(border_nodes))))
    coord_dict = {}
    tmp = [coord_dict.update({i:(pts[1],pts[0])}) for i, pts in enumerate(border_nodes)]
    for n, p in coord_dict.items():
        G1.nodes[n]['pos'] = p
    G1.add_edges_from(border_edges)
    
    pos = nx.get_node_attributes(G1,'pos')
    #pos = nx.rescale_layout_dict(pos)
    #pos = nx.circular_layout(G)
    nx.draw(G1, pos, ax=ax[2], node_size=1, node_color='darkgrey', edge_color='darkgrey', width=1.5, font_size=12, with_labels=False)
    # nx.draw_networkx_edge_labels(G, pos, ax=ax, font_size=12, label_pos=0.5, rotate=False)
    nx.draw(G1, pos, ax=ax[3], node_size=1, node_color='darkgrey', edge_color='darkgrey', width=1.5, font_size=12, with_labels=False)

    G = nx.Graph()
    edges = [tuple(rel) for rel in edges1]
    nodes = list(np.unique(np.array(edges)))
    coord_dict = {}
    tmp = [coord_dict.update({nodes[i]:(W*pts[1], H- H*pts[0])}) for i, pts in enumerate(points1[nodes,:])]
    G.add_nodes_from(nodes)
    for n, p in coord_dict.items():
        G.nodes[n]['pos'] = p
    G.add_edges_from(edges)
    
    pos = nx.get_node_attributes(G,'pos')
    #pos = nx.rescale_layout_dict(pos)
    #pos = nx.circular_layout(G)
    nx.draw(G, pos, ax=ax[2], node_size=10, node_color='lightcoral', edge_color='mediumorchid', width=1.5, font_size=12, with_labels=False)
    # nx.draw_networkx_edge_labels(G, pos, ax=ax, font_size=12, label_pos=0.5, rotate=False)
    
    G = nx.Graph()
    edges = [tuple(rel) for rel in edges2]
    nodes = list(np.unique(np.array(edges)))
    coord_dict = {}
    tmp = [coord_dict.update({nodes[i]:(W*pts[1],H- H*pts[0])}) for i, pts in enumerate(points2[nodes,:])]
    G.add_nodes_from(nodes)
    for n, p in coord_dict.items():
        G.nodes[n]['pos'] = p
    G.add_edges_from(edges)
    pos = nx.get_node_attributes(G,'pos')
    #pos = nx.rescale_layout_dict(pos)
    #pos = nx.circular_layout(G)
    nx.draw(G, pos, ax=ax[3], node_size=10, node_color='lightcoral', edge_color='mediumorchid', width=1.5, font_size=12, with_labels=False)
    # nx.draw_networkx_edge_labels(G, pos, ax=ax, font_size=12, label_pos=0.5, rotate=False)
    
    plt.show()


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


if __name__ == "__main__":

    # config_file = "/nas/k8s/dev/research/doyoungi/git/relationformer/configs/road_rgb_2D.yaml"
    # ckpt_path = "/nas/k8s/dev/research/doyoungi/git/relationformer/work_dirs/road_rgb_2D/runs/baseline_road_resnet_def_detr_final_dec_4_10/models/epochs_70.pth"
    # show_dir = f'/nas/k8s/dev/research/doyoungi/git/relationformer/show_dirs/road_rgb_2D'

    args = parse_args()
    config_file = args.config
    ckpt_path = args.checkpoint
    show_dir = args.show_dir

    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = dict2obj(config)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    val_ds, file_infos = build_road_network_data( # 이 함수는 폴더에서 셔플해서 ds 생성함
        config, mode='test', loadXYN=True
    )

    val_loader = DataLoader(val_ds, # 4분 소요
                            batch_size=config.DATA.BATCH_SIZE,
                            shuffle=False,
                            num_workers=config.DATA.NUM_WORKERS,
                            collate_fn=image_graph_collate_road_network,
                            pin_memory=True)
    print(len(val_ds), len(val_loader)) # 테스트셋 전체 지역 패치: 25036개, 훈련셋 지역 21: 631개

    model = build_model(config)
    device = torch.device("cuda")
    model = model.to(device)

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
    if len(missing_keys) > 0:
        print('Missing Keys: {}'.format(missing_keys))
    if len(unexpected_keys) > 0:
        print('Unexpected Keys: {}'.format(unexpected_keys))


    model.eval()
    if not os.path.isdir(show_dir):
        os.makedirs(show_dir)
        os.makedirs(show_dir + "/patch")
        os.makedirs(show_dir + "/seg")
        os.makedirs(show_dir + "/graph")
    else:
        if not os.path.isdir(show_dir + "/patch"):
            os.makedirs(show_dir + "/patch")
        if not os.path.isdir(show_dir + "/seg"):
            os.makedirs(show_dir + "/seg")
        if not os.path.isdir(show_dir + "/graph"):
            os.makedirs(show_dir + "/graph")
        print("The path is already ready.")

    iteration = 0
    for idx, (images, seg, points, edges) in enumerate(tqdm(val_loader, total=len(val_loader))):
        iteration = iteration+1
        images = images.cuda()

        seg = seg.cuda()
        h, out, _ = model(images, seg=False)
        pred_nodes, pred_edges, pred_nodes_box, pred_nodes_box_score, pred_nodes_box_class, pred_edges_box_score, pred_edges_box_class = relation_infer(
                    h.detach(), out, model.relation_embed, config.MODEL.DECODER.OBJ_TOKEN, config.MODEL.DECODER.RLN_TOKEN,
                    nms=False, map_=True
                    )

        start_idx = idx*config.DATA.BATCH_SIZE
        for i in range(len(images)): # 0~31
            file_name = file_infos[start_idx+i]
            sample_id, x, y, region_num = file_name.split('_')
            # img of patch
            # NumPy 배열을 PIL 이미지로 변환
            image = images[i].cpu().numpy()
            min_value = np.min(image)
            max_value = np.max(image)
            image = (image - min_value) / (max_value - min_value)
            image = (image * 255).astype('uint8')
            image = image.transpose(1,2,0)
            image = Image.fromarray(image, 'RGB')
            # 이미지 파일로 저장할 경로 지정
            save_path = f'{show_dir}/patch/{file_name}.png'
            # 이미지 파일로 저장
            image.save(save_path)
            # seg
            image = seg[i].cpu().numpy()
            image = image+0.5
            image = (image * 255).astype('uint8')
            image = np.squeeze(image, axis=0)
            image = Image.fromarray(image, 'L')
            save_path = f'{show_dir}/seg/{file_name}.png'
            image.save(save_path)
            # gt node edge
            gt_node, gt_edge = points[i].cpu().numpy(), edges[i].cpu().numpy()
            # pred node edge
            pred_node, pred_edge = pred_nodes[i].cpu().numpy(), pred_edges[i]
            gt_node_list = gt_node.tolist()
            gt_edge_list = gt_edge.tolist()
            pred_node_list = pred_node.tolist()
            pred_edge_list = pred_edge.tolist()
            data = {
                "gt_node": gt_node_list,
                "gt_edge": gt_edge_list,
                "pred_node": pred_node_list,
                "pred_edge": pred_edge_list
            }

            with open(f'{show_dir}/graph/{file_name}.json', 'w') as json_file:
                json.dump(data, json_file, indent=2)



