import cv2
import os
import numpy as np
import random
import imageio
import torch
import pyvista
from shapely.geometry import LineString
import geopandas as gpd
from collections import Counter
from torch.utils.data import Dataset
import torchvision.transforms.functional as tvf
from .albumentation_aug import AlbumentationsAugmentation
from .directedgraph_builder import generate_directed_graph_and_sorting, gdf_to_nodes_and_edges_linestring


def get_pts_gnn_classes(node, edge):
    node_counter = Counter()
    # Count each occurrence of nodes in the edge array
    for e in edge.flatten():
        node_counter[e] += 1
    # Identify terminal nodes (nodes that appear only once)
    terminal_nodes = [n for n, count in node_counter.items() if count == 1]

    classes = np.zeros(node.shape[0])
    classes[terminal_nodes] = 1
    return classes


def is_point_on_boundary(point, img_shape, boundary_width=10):
    h, w = img_shape
    x, y = point

    if x <= boundary_width:
        return True

    if x >= w - boundary_width:
        return True

    if y <= boundary_width:
        return True

    if y >= h - boundary_width:
        return True
    return False


def get_pts_classes(lines, coordinates, img_shape, boundary_ratio=0.10):
    img_shape = img_shape[:2]
    boundary_width = int(img_shape[0] * boundary_ratio)

    unique, counts = np.unique(lines, return_counts=True)
    counts_dict = dict(zip(unique, counts))
    intersection_index = [int(k) for k, v in counts_dict.items() if int(v) > 2]

    end_index_candidate = [int(k) for k, v in counts_dict.items() if int(v) == 1]
    end_index_candidate_coords = [coordinates[x] for x in end_index_candidate]

    is_pts_on_boundary = [is_point_on_boundary(x, img_shape, boundary_width) for x in end_index_candidate_coords]
    end_index = [x for i, x in enumerate(end_index_candidate) if is_pts_on_boundary[i] == True]

    pts_class = np.zeros(len(coordinates), dtype=np.int64)

    for pt_id in intersection_index:
        pts_class[pt_id] = 1
    for pt_id in end_index:
        pts_class[pt_id] = 2

    # removal end_index
    # pts_class[pts_class == 3] = 1
    # pts_class[pts_class == 2] = 1 
    
    return pts_class


class Sat2GraphDataLoader(Dataset):
    """[summary]

    Args:
        Dataset ([type]): [description]
    """
    def __init__(self, data, transform):
        """[summary]

        Args:
            data ([type]): [description]
            transform ([type]): [description]
        """
        self.data = data
        self.transform = transform
        self.albumentation_transform = AlbumentationsAugmentation(self.transform)

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def prepare_seg_sat2graph(self, seg):
        seg = (seg > 127).astype('float32')
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        dilated_seg = cv2.dilate(seg.astype('uint8'), kernel, iterations=2)
        return dilated_seg
    
    def __len__(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """[summary]

        Args:
            idx ([type]): [description]

        Returns:
            [type]: [description]
        """
        data = self.data[idx]
        image_data = imageio.imread(data['img'])
        image_data = self.albumentation_transform(image_data)

        image_data = torch.tensor(image_data, dtype=torch.float).permute(2,0,1)
        image_data = image_data/255.0
        vtk_data = pyvista.read(data['vtp'])

        seg_data = imageio.imread(data['seg'])
        seg_data = self.prepare_seg_sat2graph(seg_data)
        seg_data = torch.tensor(seg_data, dtype=torch.long).unsqueeze(0)

        image_data = image_data.clone().detach().to(dtype=torch.float)
        image_data = tvf.normalize(
            image_data,
            mean=self.mean, 
            std=self.std)
        
        coordinates = np.float32(np.asarray(vtk_data.points))[:, :2]
        lines = np.asarray(vtk_data.lines.reshape(-1, 3))[:, 1:]

        # Re-indexing the coordinates and edges order
        coord_lines = []
        for l in lines:
            coord_lines.append(LineString([coordinates[l[0]], coordinates[l[1]]]))
        gdf = gpd.GeoDataFrame(geometry=coord_lines)
        p_gdf = generate_directed_graph_and_sorting(gdf)
        nodes, edges = gdf_to_nodes_and_edges_linestring(p_gdf)
        nodes = np.float32(nodes[:,::-1])
        edges = np.array(edges)

        # coordinates = torch.tensor(coordinates, dtype=torch.float)
        # lines = torch.tensor(lines, dtype=torch.int64)
        coordinates = torch.tensor(nodes, dtype=torch.float)
        lines = torch.tensor(edges, dtype=torch.int64)
        
        # pts_labels
        # coords_pts = np.round(np.float32(np.asarray(vtk_data.points))[:,:2] * image_data.shape[1]).astype('int64')
        # pts_labels = get_pts_classes(
        #     np.asarray(vtk_data.lines.reshape(-1, 3))[:,1:], 
        #     coords_pts, image_data.shape[1:], boundary_ratio=0.075)
        pts_labels = get_pts_gnn_classes(nodes, edges)
        pts_labels = torch.tensor(pts_labels, dtype=torch.int64)

        return image_data, seg_data, coordinates, lines, pts_labels


def build_road_network_data(config, mode="train", split=0.95, loadXYN=False):
    """[summary]

    Args:
        data_dir (str, optional): [description]. Defaults to ''.
        mode (str, optional): [description]. Defaults to 'train'.
        split (float, optional): [description]. Defaults to 0.8.

    Returns:
        [type]: [description]
    """
    # Augmentations
    if config.DATA.AUGMENTATIONS:
        train_transform = [
            # dict(type='RandomShadow', p=0.3),
            # dict(type='RandomFog', p=0.3),
            # dict(type='RandomGamma', p=0.3),
            dict(type='ColorJitter', brightness=0.2, contrast=0, saturation=0, p=0.3),
            dict(type='ColorJitter', brightness=0, contrast=0.2, saturation=0, p=0.3),
            dict(type='ColorJitter', brightness=0, contrast=0, saturation=0.2, p=0.3),
            dict(type='CLAHE', p=0.3),
        ]
    else:
        train_transform = []

    print("train augmentations:", train_transform)
    val_transform = []
    if mode == "train":
        img_folder = os.path.join(config.DATA.DATA_PATH, "raw")
        seg_folder = os.path.join(config.DATA.DATA_PATH, "seg")
        vtk_folder = os.path.join(config.DATA.DATA_PATH, "vtp")
        img_files = []
        vtk_files = []
        seg_files = []
        # Ex) sample_000001_5_748_21_data.png

        for file_ in os.listdir(img_folder):
            file_ = file_[:-8]
            img_files.append(os.path.join(img_folder, file_ + "data.png"))
            vtk_files.append(os.path.join(vtk_folder, file_ + "graph.vtp"))
            seg_files.append(os.path.join(seg_folder, file_ + "seg.png"))

        data_dicts = [
            {"img": img_file, "vtp": vtk_file, "seg": seg_file}
            for img_file, vtk_file, seg_file in zip(img_files, vtk_files, seg_files)
        ]
        ds = Sat2GraphDataLoader(
            data=data_dicts,
            transform=train_transform,
        )
        return ds
    elif mode == "split":
        img_folder = os.path.join(config.DATA.DATA_PATH, "raw")
        seg_folder = os.path.join(config.DATA.DATA_PATH, "seg")
        vtk_folder = os.path.join(config.DATA.DATA_PATH, "vtp")
        img_files = []
        vtk_files = []
        seg_files = []

        for file_ in os.listdir(img_folder):
            file_ = file_[:-8]
            img_files.append(os.path.join(img_folder, file_ + "data.png"))
            vtk_files.append(os.path.join(vtk_folder, file_ + "graph.vtp"))
            seg_files.append(os.path.join(seg_folder, file_ + "seg.png"))

        data_dicts = [
            {"img": img_file, "vtp": vtk_file, "seg": seg_file}
            for img_file, vtk_file, seg_file in zip(img_files, vtk_files, seg_files)
        ]
        # 원래 cities20 데이터에 대해 훈련셋을 랜덤하게 나눠서 훈련과 val 가지고 훈련하는 모드
        if config.DATA.DATASET == "US20-road-network-2D":
            random.seed(config.DATA.SEED)
            random.shuffle(data_dicts)
            train_split = int(split * len(data_dicts))
            train_files, val_files = data_dicts[:train_split], data_dicts[train_split:]
            train_ds = Sat2GraphDataLoader(
                data=train_files,
                transform=train_transform,
            )
            val_ds = Sat2GraphDataLoader(
                data=val_files,
                transform=val_transform,
            )
            return train_ds, val_ds
        # spacenet3 데이터는 val폴더를 참조하여 train val 훈련해야 함
        if config.DATA.DATASET == "spacenet3":
            # 작성중
            train_ds = Sat2GraphDataLoader(
                data=data_dicts,
                transform=train_transform,
            )
            img_folder = os.path.join(config.DATA.VAL_DATA_PATH, "raw")
            seg_folder = os.path.join(config.DATA.VAL_DATA_PATH, "seg")
            vtk_folder = os.path.join(config.DATA.VAL_DATA_PATH, "vtp")
            img_files = []
            vtk_files = []
            seg_files = []

            for file_ in os.listdir(img_folder):
                file_ = file_[:-8]
                img_files.append(os.path.join(img_folder, file_ + "data.png"))
                vtk_files.append(os.path.join(vtk_folder, file_ + "graph.vtp"))
                seg_files.append(os.path.join(seg_folder, file_ + "seg.png"))

            data_dicts = [
                {"img": img_file, "vtp": vtk_file, "seg": seg_file}
                for img_file, vtk_file, seg_file in zip(img_files, vtk_files, seg_files)
            ]
            val_ds = Sat2GraphDataLoader(
                data=data_dicts,
                transform=val_transform,
            )
            return train_ds, val_ds
        raise  # debugging
    elif mode == "test":  # 테스트 모드만 loadXYN 고려한다.
        print("test_mode")
        img_folder = os.path.join(config.DATA.TEST_DATA_PATH, "raw")
        seg_folder = os.path.join(config.DATA.TEST_DATA_PATH, "seg")
        vtk_folder = os.path.join(config.DATA.TEST_DATA_PATH, "vtp")
        img_files = []  # 경로 리스트
        vtk_files = []
        seg_files = []
        file_infos = []

        for file_ in os.listdir(img_folder):
            file_ = file_[:-8]
            img_files.append(os.path.join(img_folder, file_ + "data.png"))
            vtk_files.append(os.path.join(vtk_folder, file_ + "graph.vtp"))
            seg_files.append(os.path.join(seg_folder, file_ + "seg.png"))
            file_ = file_.split("_")[1:5]
            file_ = "_".join(file_)
            file_infos.append(file_)

        data_dicts = [
            {"img": img_file, "vtp": vtk_file, "seg": seg_file}
            for img_file, vtk_file, seg_file in zip(img_files, vtk_files, seg_files)
        ]
        ds = Sat2GraphDataLoader(
            data=data_dicts,
            transform=val_transform,
        )
        if loadXYN:
            return ds, file_infos  # 파일 인포를 넘겨주고 이후 save할때 지역을 명시한다. 그 지역이름으로 패치 합치기 진행
        return ds