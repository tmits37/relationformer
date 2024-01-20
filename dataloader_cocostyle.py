import os
import random
import numpy as np
import pandas as pd
import geopandas as gpd
from skimage import io
from skimage.transform import resize
from shapely.geometry import Polygon

import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO

from dataset_preparing import get_coords_from_densifing_points, generate_heatmap


# dense에 대한 간격이나 표준편차를 하이퍼 파라미터로 조정가능한 코드. 히트맵 생성 때문에 속도는 좀 걸릴 수 있음
# Inria 데이터 크기 조정하여 coco 포맷으로 맞춰준 데이터 처리가능
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

# 딕셔너리로 된 것을 텐서로 네 묶음을 만들어서 리스트에 넣어서 리턴
def image_graph_collate_road_network_coco(batch):
    images = torch.stack([item['image'] for item in batch], 0).contiguous()
    heatmap = torch.stack([item['heatmap'] for item in batch], 0).contiguous()
    points = [item['nodes'] for item in batch]
    edges = [item['edges'] for item in batch]

    return [images, heatmap, points, edges]


def create_polygon(segmentation):
    # COCO segmentation format is [x1, y1, x2, y2, ..., xn, yn]
    # We need to reshape it to [(x1, y1), (x2, y2), ..., (xn, yn)]
    points = list(zip(segmentation[::2], segmentation[1::2]))
    return Polygon(points)


def gdf_to_nodes_and_edges(gdf):
    nodes = []
    for _, row in gdf.iterrows():
        polygon = row['geometry']
        if polygon.geom_type == 'Polygon':
            for x, y in polygon.exterior.coords:
                nodes.append((x, y))
        elif polygon.geom_type == 'MultiPolygon':
            for part in polygon:
                for x, y in part.exterior.coords:
                    nodes.append((x, y))
        else:
            raise AttributeError

    # Remove duplicates if necessary
    nodes = list(set(nodes))

    # Create a DataFrame for nodes with unique indices
    node_df = pd.DataFrame(nodes, columns=['x', 'y'])
    node_df['node_id'] = range(len(node_df))

    edges = []
    for _, row in gdf.iterrows():
        polygon = row['geometry']
        if polygon.geom_type == 'Polygon':
            coords = polygon.exterior.coords[:-1]  # Exclude closing vertex
            edge = [(node_df[(node_df['x'] == x) & (node_df['y'] == y)].index[0], 
                    node_df[(node_df['x'] == coords[(i+1)%len(coords)][0]) & (node_df['y'] == coords[(i+1)%len(coords)][1])].index[0]) 
                    for i, (x, y) in enumerate(coords)]
            edges.extend(edge)
        elif polygon.geom_type == 'MultiPolygon':
            for part in polygon:
                coords = part.exterior.coords[:-1]
                edge = [(node_df[(node_df['x'] == x) & (node_df['y'] == y)].index[0], 
                        node_df[(node_df['x'] == coords[(i+1)%len(coords)][0]) & (node_df['y'] == coords[(i+1)%len(coords)][1])].index[0]) 
                        for i, (x, y) in enumerate(coords)]
                edges.extend(edge)

    return node_df[['y', 'x']].values, edges


class CrowdAI(Dataset):
    """A dataset class for handling and processing data from the CrowdAI dataset.

    Attributes:
        IMAGES_DIRECTORY (str): Directory containing the images.
        ANNOTATIONS_PATH (str): File path for the annotations.
        coco (COCO): COCO object to handle COCO annotations.
        max_points (int): Maximum number of points to consider (default 256).
        gap_distance (float): Distance between interpolated points.
        sigma (float): Standard deviation for Gaussian kernel used in heatmap generation.

    Args:
        images_directory (str): Directory where the dataset images are stored.
        annotations_path (str): File path for the COCO format annotations.
        gap_distance (int, optional): Gap distance for densifying points. Defaults to 20.
        sigma (float, optional): Sigma value for Gaussian blur in heatmap. Defaults to 1.5.
    """

    def __init__(self, 
                 images_directory, 
                 annotations_path,
                 gap_distance=20,
                 sigma=1.5):

        self.IMAGES_DIRECTORY = images_directory
        self.ANNOTATIONS_PATH = annotations_path
        self.coco = COCO(self.ANNOTATIONS_PATH)
        self.image_ids = self.coco.getImgIds(catIds=self.coco.getCatIds())

        self.len = len(self.image_ids)

        self.max_points = 256 # TODO: It should be restricted the number when gt points over the max points limit
        self.gap_distance = gap_distance
        self.sigma = sigma

        print("Built Dataset Options:")
        print(f"--Num.of images: {self.image_ids}")
        print(f"--Gap Distance: {self.gap_distance}", f"--Sigma: {self.sigma}")

    def prepare_annotations(self, img):
        """Prepares annotations for an image.
        Args:
            img (dict): A dictionary containing image metadata.

        Returns:
            GeoDataFrame: A GeoDataFrame containing the geometrical data of annotations.
        """
        annotation_ids = self.coco.getAnnIds(imgIds=img['id'])
        annotations = self.coco.loadAnns(annotation_ids)
        random.shuffle(annotations)

        data = []
        for ann in annotations:
            polygon = create_polygon(ann['segmentation'][0])
            data.append({'id': ann['id'], 'geometry': polygon})
        gdf = gpd.GeoDataFrame(data, geometry='geometry')
        return gdf

    def loadSample(self, idx):
        """Loads a sample for a given index.

        Args:
            idx (int): The index of the sample to load.

        Returns:
            dict: A dictionary containing the sample data.
                'image' (torch.Tensor of shape [3, H, W], torch.float32): 
                    The image tensor normalized to [0, 1].
                'image_idx' (torch.Tensor of shape [1], torch.long): 
                    The index of the image.
                'heatmap' (torch.Tensor of shape [H, W], torch.float32): 
                    The heatmap tensor for the image normalized to [0, 1]. 
                'nodes' (torch.Tensor of shape [N, 2], torch.float): 
                    The nodes tensor representing points in the image.
                    nodes are normalized to [0, 1]
                'edges' (torch.Tensor of shape [E, 2], torch.long): 
                    The edges tensor representing connections between nodes.
        """
        idx = self.image_ids[idx]

        img = self.coco.loadImgs(idx)[0]
        image_path = os.path.join(self.IMAGES_DIRECTORY, img['file_name'])
        image = io.imread(image_path)

        origin_gdf = self.prepare_annotations(img)
        coords, gdf = get_coords_from_densifing_points(origin_gdf, gap_distance=self.gap_distance) # [N, 2]
        heatmap = generate_heatmap(coords, image.shape[:2], sigma=self.sigma)

        nodes, edges = gdf_to_nodes_and_edges(gdf)
        nodes = nodes / image.shape[0]

        image_idx = torch.tensor([idx])
        image = resize(image, (320, 320, 3), anti_aliasing=True, preserve_range=True) # TODO 일단 300->320, 변형 정도 체크
        image = torch.from_numpy(image)
        image = image.float()
        image = image.permute(2,0,1) / 255.0
        heatmap = resize(heatmap, (320, 320, 1), anti_aliasing=True, preserve_range=True)
        heatmap = torch.from_numpy(heatmap)
        heatmap = heatmap.float()
        heatmap = heatmap.permute(2,0,1) / 255.0
        # if len(nodes) > 256: # 정답 노드가 256개 초과인 경우
        #     print("num_nodes:", len(nodes))
        try:
            nodes = torch.tensor(nodes, dtype=torch.float32)
        except:
            # print("Error! nodes:", nodes) # 노드가 0개인 에러
            pass
        edges = torch.tensor(edges, dtype=torch.long)

        sample = {
            'image': image, 
            'image_idx': image_idx, 
            'heatmap': heatmap,
            'nodes': nodes,
            'edges': edges,
            }
        return sample

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        sample = self.loadSample(idx)
        number_of_nodes = len(sample['nodes']) # 1~256
        while number_of_nodes == 0 or number_of_nodes > 256: # 0 or > 256
            idx_new = random.randint(0, self.len-1)
            # print("Pick new one")
            sample = self.loadSample(idx_new)
            number_of_nodes = len(sample['nodes'])
        return sample


def build_inria_coco_data(config, mode='train'):
    if mode == 'train':
        ds = CrowdAI(
            images_directory=config.DATA.COCO_IMAGE_DIR,
            annotations_path=config.DATA.COCO_ANNOT_PATH,
            gap_distance=config.DATA.GAP_DISTANCE,
            sigma=config.DATA.SIGMA
        )
    elif mode == 'test':
        ds = CrowdAI(
            images_directory=config.DATA.TEST_COCO_IMAGE_DIR,
            annotations_path=config.DATA.TEST_COCO_ANNOT_PATH,
            gap_distance=config.DATA.GAP_DISTANCE,
            sigma=config.DATA.SIGMA
        )
    else:
        raise AssertionError
    return ds


if __name__ == '__main__':
    dataset = CrowdAI(images_directory='/nas/tsgil/dataset/Inria_building/cocostyle/images',
                      annotations_path='/nas/tsgil/dataset/Inria_building/cocostyle/annotation.json')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=6, collate_fn=image_graph_collate_road_network_coco)
    
    print(next(iter(dataloader))[0].shape) # image

    data = next(iter(dataloader))

    image = data[0][1].detach().cpu().numpy().transpose(1,2,0)
    heatmap = data[1][1].detach().cpu().numpy()
    nodes = data[2][1].detach().cpu().numpy() * image.shape[0]
    edges = data[3][1].detach().cpu().numpy()

    nodes = nodes.astype('int64')

    # Visualize
    import matplotlib.pyplot as plt
    plt.imshow(min_max_normalize(image, 0))
    plt.scatter(nodes[:,1], nodes[:,0], color='r')

    for e in edges:
        connect = np.stack([nodes[e[0]], nodes[e[1]]], axis=0)
        plt.plot(connect[:,1], connect[:,0])
    plt.show()