import os
import random
import numpy as np
import pandas as pd
import geopandas as gpd
from skimage import io
from skimage.transform import resize
from shapely.geometry import Polygon, LineString

import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO

from dataset_preparing import get_coords_from_densifing_points, generate_heatmap
from dataloader_cocostyle import gdf_to_nodes_and_edges


def create_linestring(segmentation):
    # COCO segmentation format is [x1, y1, x2, y2, ..., xn, yn]
    # We need to reshape it to [(x1, y1), (x2, y2), ..., (xn, yn)]
    # points = list(zip(segmentation[::2], segmentation[1::2]))
    if len(segmentation[0]) == 1:
        return None
    else:
        return LineString(segmentation)


def gdf_to_nodes_and_edges_linestring(gdf):
    nodes = []
    for _, row in gdf.iterrows():
        polygon = row['geometry']
        if polygon.geom_type == 'LineString':
            for x, y in polygon.coords:
                nodes.append((x, y))
        elif polygon.geom_type == 'MultiLineString':
            for part in polygon:
                for x, y in part.coords:
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
        if polygon.geom_type == 'LineString':
            coords = polygon.coords  # Exclude closing vertex
            edge = [(node_df[(node_df['x'] == x) & (node_df['y'] == y)].index[0], 
                    node_df[(node_df['x'] == coords[(i+1)%len(coords)][0]) & (node_df['y'] == coords[(i+1)%len(coords)][1])].index[0]) 
                    for i, (x, y) in enumerate(coords)]
            edges.extend(edge)
        elif polygon.geom_type == 'MultiLineString':
            for part in polygon:
                coords = part.coords
                edge = [(node_df[(node_df['x'] == x) & (node_df['y'] == y)].index[0], 
                        node_df[(node_df['x'] == coords[(i+1)%len(coords)][0]) & (node_df['y'] == coords[(i+1)%len(coords)][1])].index[0]) 
                        for i, (x, y) in enumerate(coords)]
                edges.extend(edge)

    return node_df[['y', 'x']].values, edges


class CrowdAIRoad(Dataset):
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
                 gap_distance=10,
                 sigma=1.0,
                 nms=False):

        self.IMAGES_DIRECTORY = images_directory
        self.ANNOTATIONS_PATH = annotations_path
        self.coco = COCO(self.ANNOTATIONS_PATH)
        self.image_ids = self.coco.getImgIds(catIds=self.coco.getCatIds())

        self.len = len(self.image_ids)

        self.max_points = 256 # TODO: It should be restricted the number when gt points over the max points limit
        self.gap_distance = gap_distance
        self.sigma = sigma
        self.nms = nms

        print("Built Dataset Options:")
        print(f"--Num.of images: {self.len}")
        print(f"--Gap Distance: {self.gap_distance}", f"--Sigma: {self.sigma}", f"--nms: {self.nms}")

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
            # print(ann['segmentation'][0], ann['segmentation'][1])
            polygon = create_linestring(ann['segmentation'])
            if polygon is None:
                pass
            else:
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
        # coords, gdf = get_coords_from_densifing_points(origin_gdf, gap_distance=self.gap_distance, nms=self.nms) # [N, 2]
        p_gdf = origin_gdf.copy()
        origin_coordinates = []
        for i, row in p_gdf.iterrows():
            geom = row.geometry

            x, y = geom.xy
            x, y = np.array(x), np.array(y)
            pos = np.stack([x,y], axis=1)
            pos = np.round(pos).astype('uint16')
            origin_coordinates.append(pos)
        coords = np.concatenate(origin_coordinates, axis=0) # .tolist()

        heatmap = generate_heatmap(coords, image.shape[:2], sigma=self.sigma)

        nodes, edges = gdf_to_nodes_and_edges_linestring(origin_gdf)
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
        if len(nodes) > 256: # 정답 노드가 256개 초과인 경우
            print("num_nodes:", len(nodes))
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
        # while number_of_nodes == 0:
            idx_new = random.randint(0, self.len-1)
            # print("Pick new one")
            sample = self.loadSample(idx_new)
            number_of_nodes = len(sample['nodes'])
        return sample

def build_road_coco_data(config, mode='train'):
    if mode == 'train':
        ds = CrowdAIRoad(
            images_directory=config.DATA.COCO_IMAGE_DIR,
            annotations_path=config.DATA.COCO_ANNOT_PATH,
            gap_distance=config.DATA.GAP_DISTANCE,
            sigma=config.DATA.SIGMA,
            nms=config.DATA.NMS,
        )
    elif mode == 'test':
        ds = CrowdAIRoad(
            images_directory=config.DATA.TEST_COCO_IMAGE_DIR,
            annotations_path=config.DATA.TEST_COCO_ANNOT_PATH,
            gap_distance=config.DATA.GAP_DISTANCE,
            sigma=config.DATA.SIGMA,
            nms=config.DATA.NMS,
        )
    else:
        raise AssertionError
    return ds

if __name__ == "__main__":
    dataset = CrowdAIRoad(images_directory='/nas/k8s/dev/research/doyoungi/dataset/sat2graph/cocostyle/images',
                    annotations_path='/nas/k8s/dev/research/doyoungi/dataset/sat2graph/cocostyle/annotation.json',
                    gap_distance=10,
                    sigma=0.8,
                    nms=False)
    
    from tqdm import tqdm 
    for i in tqdm(range(len(dataset))):
        d = dataset[i]
