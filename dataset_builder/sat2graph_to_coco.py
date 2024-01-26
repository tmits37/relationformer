from pycocotools.coco import COCO
import os
import numpy as np
from skimage import io
import json
from tqdm import tqdm
from itertools import groupby
from shapely.geometry import Polygon, mapping
from skimage.measure import label as ski_label
from skimage.measure import regionprops
from shapely.geometry import box, LineString
import cv2
import glob
import math
import rasterio
import geopandas as gpd
from shapely import affinity
import imageio.v2 as imageio
import pickle

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from shapely.geometry import Polygon, MultiPolygon
from Inria_heatmap import raster_to_polygons
from inria_to_coco import crop2patch, lt_crop

import sys
sys.path.append("/nas/k8s/dev/research/doyoungi/git/relationformer")
from generate_20cities_data import prepare_spacenet3, prepare_sat2graph
from generate_data import convert_graph


def original_processing(args, test_input_args):
    input_image_path = args['input_image_path']
    input_gt_path =args['input_gt_path']
    save_path = args['save_path']
    cities = args['cities']
    val_set = args['val_set']
    output_im_train = args['output_im_train']
    patch_width = args['patch_width']
    patch_height = args['patch_height']
    patch_overlap = args['patch_overlap']
    patch_size = args['patch_size']
    output_data_train = args['output_data_train']

    train_ob_id = 0
    train_im_id = 0
    # read in data with npy format
    input_label = os.listdir(input_gt_path)
    # print(input_label)
    for i, input_args in enumerate(tqdm(test_input_args)):
        _, file_name, raw_file, seg_file, vtk_file, image_id, dense, dataset, cfg_options = input_args.values()
        try:
            image_data = imageio.imread(raw_file + ".png")
        except:
            image_data = imageio.imread(raw_file + ".jpg")
        try:
            vtk_file = vtk_file.replace('gt_graph_dense.p', 'gt_graph.p')
            with open(vtk_file, "rb") as f:
                graph = pickle.load(f)
            node_array, edge_array = convert_graph(graph)
            if dataset == "spacenet":
                node_array[:, 0] = 400 - node_array[:, 0]
        except:
            pass
        gt_im_data = imageio.imread(seg_file)
        line_strings = [LineString([node_array[start, ::-1], node_array[end, ::-1]]) for start, end in edge_array]
        geom = gpd.GeoDataFrame(geometry=line_strings)

        patch_list = crop2patch(image_data.shape, patch_width, patch_height, patch_overlap)
        for pid, pa in enumerate(patch_list):
            x_ul, y_ul, pw, ph = pa
            p_gt = gt_im_data[y_ul:y_ul+patch_height, x_ul:x_ul+patch_width]
            p_im = image_data[y_ul:y_ul+patch_height, x_ul:x_ul+patch_width, :]
            p_gts = []
            p_ims = []
            p_im_rd, p_gt_rd = lt_crop(p_im, p_gt, patch_size)
            p_gts.append(p_gt_rd)
            p_ims.append(p_im_rd)

            bbox = box(x_ul, y_ul, x_ul+pw-0.5, y_ul+ph-0.5)
            transform_params = (1, 0, 0, 1, -x_ul, -y_ul)
            intersect_gdf = geom[geom.intersects(bbox)]
            valid_intersect_gdf = intersect_gdf[intersect_gdf['geometry'].is_valid]
            intersection_gdf = gpd.GeoDataFrame(geometry=valid_intersect_gdf.intersection(bbox))
            intersection_gdf['geometry'] = intersection_gdf['geometry'].apply(lambda geom: affinity.affine_transform(geom, transform_params))
            p_lines = intersection_gdf.geometry.to_list()

            if np.sum(p_gt > 0) > 5:
                for poly in p_lines:
                    p_bbox = [poly.bounds[0], poly.bounds[1],
                                poly.bounds[2]-poly.bounds[0], poly.bounds[3]-poly.bounds[1]]
                    p_seg = []
                    coor_list = mapping(poly)['coordinates']
                    for part_poly in coor_list:
                        p_seg.append(np.asarray(part_poly).ravel().tolist())

                    anno_info = {
                        'id': train_ob_id,
                        'image_id': train_im_id,
                        'segmentation': p_seg,
                        'area': poly.area,
                        'bbox': p_bbox,
                        'category_id': 100,
                        'iscrowd': 0
                    }
                    output_data_train['annotations'].append(anno_info)
                    train_ob_id += 1

                # get patch info
                p_name = file_name + '_' + str(train_im_id).zfill(4) + '.png'
                patch_info = {'id': train_im_id, 'file_name': p_name, 'width': patch_size, 'height': patch_size}
                output_data_train['images'].append(patch_info)
                # save patch image
                io.imsave(os.path.join(output_im_train, p_name), p_im)
                train_im_id += 1

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, 'annotation.json'), 'w') as f_json:
        json.dump(output_data_train, f_json)
    return None


if __name__ == '__main__':

    input_image_path = '/nas/Dataset/inria/AerialImageDataset/train/images'
    input_gt_path = '/nas/Dataset/inria/AerialImageDataset/train/gt'
    save_path = '/nas/k8s/dev/research/doyoungi/dataset/sat2graph/cocostyle'

    cities = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
    val_set = ['10', '20']

    os.makedirs(save_path, exist_ok=True)
    output_im_train = os.path.join(save_path, 'images')
    if not os.path.exists(output_im_train):
        os.makedirs(output_im_train)

    patch_width = 300 # 725
    patch_height = 300 # 725
    patch_overlap = 0
    patch_size = 300 # 512
    # rotation_list = [22.5, 45, 67.5]

    # main dict for annotation file
    output_data_train = {
        'info': {'district': 'Sat2graph', 'description': 'road_networks', 'contributor': 'whu'},
        'categories': [{'id': 100, 'name': 'road'}],
        'images': [],
        'annotations': [],
    }

    args = {
        'input_image_path': input_image_path,
        'input_gt_path': input_gt_path,
        'save_path': save_path,
        'cities': cities,
        'val_set': val_set,
        'output_im_train': output_im_train,
        'patch_width': patch_width,
        'patch_height': patch_height,
        'patch_overlap': patch_overlap,
        'patch_size': patch_size,
        'output_data_train': output_data_train,
    }

    test_input_args, train_input_args = prepare_sat2graph(p=300, dense=True)
    options = 'HiSup'
    print(options)
    if options == 'HiSup':
        original_processing(args, train_input_args)
        pass

    print('Done!')