import scipy
import os
import sys
import numpy as np
import random
import pickle
import json
import scipy.ndimage
import imageio
import math
import torch
import pyvista
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
import torchvision.transforms.functional as tvf
from albumentation_aug import AlbumentationsAugmentation


# train_transform = [
#         dict(type='HorizontalFlip', p=0.5),
#         dict(type='RandomShadow', p=0.3),
#         dict(type='RandomFog', p=0.3),
#         dict(type='CLAHE', p=0.3),
#         dict(type='RandomGamma', p=0.3),
#         dict(type='ColorJitter', brightness=0.2, contrast=0, saturation=0, p=0.3),
#         dict(type='ColorJitter', brightness=0, contrast=0.2, saturation=0, p=0.3),
#         dict(type='ColorJitter', brightness=0, contrast=0, saturation=0.2, p=0.3),
#     ]
# val_transform = []


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
        seg_data = seg_data/np.max(seg_data)
        seg_data = torch.tensor(seg_data, dtype=torch.long).unsqueeze(0)

        image_data = image_data.clone().detach().to(dtype=torch.float)
        image_data = tvf.normalize(
            image_data,
            mean=self.mean, 
            std=self.std)
        
        coordinates = torch.tensor(np.float32(np.asarray(vtk_data.points)), dtype=torch.float)
        lines = torch.tensor(np.asarray(vtk_data.lines.reshape(-1, 3)), dtype=torch.int64)

        return image_data, seg_data, coordinates[:,:2], lines[:,1:]


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