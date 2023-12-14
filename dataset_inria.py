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
from skimage.transform import resize
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
import torchvision.transforms.functional as tvf

train_transform = [] # 증강 넣을 수도 있음
val_transform = []

class Img2HeatmapDataLoader(Dataset): # 데이터셋 인스턴스
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
        image_data = resize(image_data, (320, 320, 3), anti_aliasing=True, preserve_range=True)
        image_data = torch.tensor(image_data, dtype=torch.float).permute(2,0,1)
        image_data = image_data/255.0
        # htm_data = pyvista.read(data['vtp']) # htm는 좌표 값 파일

        htm_data = imageio.imread(data['htm'])
        htm_data = resize(htm_data, (320, 320), anti_aliasing=True, preserve_range=True)
        htm_data = torch.tensor(htm_data, dtype=torch.int).unsqueeze(0)
        htm_data = htm_data/255.0
        # print(torch.unique(htm_data))

        ann_data = imageio.imread(data['ann'])
        ann_data = ann_data/np.max(ann_data)
        ann_data = torch.tensor(ann_data, dtype=torch.int).unsqueeze(0)

        image_data = image_data.clone().detach().to(dtype=torch.float)
        image_data = tvf.normalize(
            image_data,
            mean=self.mean, 
            std=self.std)

        return image_data, ann_data, htm_data


def build_inria_data(config, mode="train", split=0.95, loadXYN=False):
    """[summary]

    Args:
        data_dir (str, optional): [description]. Defaults to ''.
        mode (str, optional): [description]. Defaults to 'train'.
        split (float, optional): [description]. Defaults to 0.8.

    Returns:
        [type]: [description]
    """
    if mode == "train":
        img_folder = os.path.join(config.DATA.DATA_PATH, "img")
        ann_folder = os.path.join(config.DATA.DATA_PATH, "ann")
        htm_folder = os.path.join(config.DATA.DATA_PATH, "heatmap")
        img_files = []
        htm_files = []
        ann_files = []
        # Ex) sample_000001_5_748_21_data.png
        # 1_1_1_1.png

        for file_ in os.listdir(img_folder):
            img_files.append(os.path.join(img_folder, file_))
            htm_files.append(os.path.join(htm_folder, file_))
            ann_files.append(os.path.join(ann_folder, file_))

        data_dicts = [
            {"img": img_file, "htm": htm_file, "ann": ann_file}
            for img_file, htm_file, ann_file in zip(img_files, htm_files, ann_files)
        ]
        ds = Img2HeatmapDataLoader(
            data=data_dicts,
            transform=train_transform,
        )
        return ds
    elif mode == "split":
        img_folder = os.path.join(config.DATA.DATA_PATH, "img")
        ann_folder = os.path.join(config.DATA.DATA_PATH, "ann")
        htm_folder = os.path.join(config.DATA.DATA_PATH, "heatmap")
        img_files = []
        htm_files = []
        ann_files = []

        for file_ in os.listdir(img_folder):
            img_files.append(os.path.join(img_folder, file_))
            htm_files.append(os.path.join(htm_folder, file_))
            ann_files.append(os.path.join(ann_folder, file_))

        data_dicts = [
            {"img": img_file, "htm": htm_file, "ann": ann_file}
            for img_file, htm_file, ann_file in zip(img_files, htm_files, ann_files)
        ]
        # 원래 cities20 데이터에 대해 훈련셋을 랜덤하게 나눠서 훈련과 val 가지고 훈련하는 모드
        if config.DATA.DATASET == "Inria" or config.DATA.DATASET == "US20-road-network-2D":
            random.seed(config.DATA.SEED)
            random.shuffle(data_dicts)
            train_split = int(split * len(data_dicts))
            train_files, val_files = data_dicts[:train_split], data_dicts[train_split:]
            train_ds = Img2HeatmapDataLoader(
                data=train_files,
                transform=train_transform,
            )
            val_ds = Img2HeatmapDataLoader(
                data=val_files,
                transform=val_transform,
            )
            return train_ds, val_ds
        # spacenet3 데이터는 val폴더를 참조하여 train val 훈련해야 함
        if config.DATA.DATASET == "spacenet3":
            # 작성중
            train_ds = Img2HeatmapDataLoader(
                data=data_dicts,
                transform=train_transform,
            )
            img_folder = os.path.join(config.DATA.VAL_DATA_PATH, "img")
            ann_folder = os.path.join(config.DATA.VAL_DATA_PATH, "ann")
            htm_folder = os.path.join(config.DATA.VAL_DATA_PATH, "heatmap")
            img_files = []
            htm_files = []
            ann_files = []

            for file_ in os.listdir(img_folder):
                img_files.append(os.path.join(img_folder, file_))
                htm_files.append(os.path.join(htm_folder, file_))
                ann_files.append(os.path.join(ann_folder, file_))

            data_dicts = [
                {"img": img_file, "htm": htm_file, "ann": ann_file}
                for img_file, htm_file, ann_file in zip(img_files, htm_files, ann_files)
            ]
            val_ds = Img2HeatmapDataLoader(
                data=data_dicts,
                transform=val_transform,
            )
            return train_ds, val_ds
        raise  # debugging
    elif mode == "test":  # 테스트 모드만 loadXYN 고려한다.
        img_folder = os.path.join(config.DATA.TEST_DATA_PATH, "img")
        ann_folder = os.path.join(config.DATA.TEST_DATA_PATH, "ann")
        htm_folder = os.path.join(config.DATA.TEST_DATA_PATH, "heatmap")
        img_files = []  # 경로 리스트
        htm_files = []
        ann_files = []
        file_infos = []

        for file_ in os.listdir(img_folder):
            img_files.append(os.path.join(img_folder, file_))
            htm_files.append(os.path.join(htm_folder, file_))
            ann_files.append(os.path.join(ann_folder, file_))
            # file_ = file_.split("_")[1:5] # TODO 파일 인포 넘기는 부분 하드 코딩된거 수정하기
            # file_ = "_".join(file_)
            # file_infos.append(file_)

        data_dicts = [
            {"img": img_file, "htm": htm_file, "ann": ann_file}
            for img_file, htm_file, ann_file in zip(img_files, htm_files, ann_files)
        ]
        ds = Img2HeatmapDataLoader(
            data=data_dicts,
            transform=val_transform,
        )
        if loadXYN:
            # 파일 인포를 넘겨주고 이후 save할때 지역을 명시한다. 그 지역이름으로 패치 합치기 진행
            return ds, file_infos
        return ds