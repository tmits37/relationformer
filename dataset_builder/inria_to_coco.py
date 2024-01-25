# Transform Inria gt dataset (binary image) to COCO format
# Using cv2.findcontours and polygon simplify to convert raster label to vector label
#
# The first 5 images are kept as validation set

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
from shapely.geometry import box
import cv2
import glob
import math
import rasterio
import geopandas as gpd
from shapely import affinity

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from shapely.geometry import Polygon, MultiPolygon
from Inria_heatmap import raster_to_polygons


def add_to_gt_polygon():
    rootdir = "/nas/k8s/dev/research/doyoungi/dataset/Inria_building/Inria_ImageDataset_PolygonLabel_FrameField/raw/train/gt_polygons"
    rootdir_json = "/nas/k8s/dev/research/doyoungi/dataset/Inria_building/Inria_ImageDataset_PolygonLabel_FrameField/raw/train/gt_polygonized"
    rdr_list = os.listdir(rootdir)
    rdr_list = [x[:-4] for x in rdr_list]
    rdr_json_list = os.listdir(rootdir_json)
    rdr_json_list = [x[:-8] for x in rdr_json_list]
    except_list = [x for x in rdr_json_list if not x in rdr_list]
    print(len(os.listdir(rootdir)), len(os.listdir(rootdir_json)), len(except_list))

    for exc in except_list:
        gdf = gpd.read_file(os.path.join(rootdir_json, f'{exc}.geojson'))
        aligned_poly = gdf.iloc[0].geometry
        polygons = [geom for geom in aligned_poly.geoms if isinstance(geom, Polygon)]
        extracted_gdf = gpd.GeoDataFrame(geometry=polygons)

        vertices_list = [np.array(polygon.exterior.coords) for polygon in polygons]
        vertices_array = np.array(vertices_list, dtype=object)

        np.save(os.path.join(rootdir, f'{exc}.npy'), vertices_array)

def explode_multipolygon(geometry):
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    else:
        return [geometry]


def canvasPoly2WorldCoord(gdf, geo_transform):
    x_gsd, _, tl_x, _, y_gsd, tl_y, _, _, _ = geo_transform
    def _transform_geom(geom):
        return affinity.affine_transform(geom, [x_gsd, 0, 0, y_gsd, tl_x, tl_y])
    gdf['geometry'] = gdf['geometry'].apply(_transform_geom)
    return gdf


def worldCoordPoly2CanvasPolygon2(gdf, geo_transform):
    x_gsd, _, tl_x, _, y_gsd, tl_y, _, _, _ = geo_transform

    inv_x_gsd = 1 / x_gsd
    inv_y_gsd = 1 / y_gsd

    inv_tl_x = -tl_x / x_gsd
    inv_tl_y = -tl_y / y_gsd

    def _transform_geom(geom):
        return affinity.affine_transform(geom, [inv_x_gsd, 0, 0, inv_y_gsd, inv_tl_x, inv_tl_y])
    gdf['geometry'] = gdf['geometry'].apply(_transform_geom)
    return gdf


def polygon2hbb(poly):
    """
    Get horizontal bounding box (match COCO)
    """
    p_x = poly[:, 0]
    p_y = poly[:, 1]
    hbb_x = np.min(p_x)
    hbb_y = np.min(p_y)
    hbb_w = np.around(np.max(p_x) - hbb_x, decimals=2)
    hbb_h = np.around(np.max(p_y) - hbb_y, decimals=2)
    hbox = [hbb_x, hbb_y, hbb_w, hbb_h]
    return [float(i) for i in hbox]

def clip_by_bound(poly, im_h, im_w):
    """
    Bound poly coordinates by image shape
    """
    p_x = poly[:, 0]
    p_y = poly[:, 1]
    p_x = np.clip(p_x, 0.0, im_w-1)
    p_y = np.clip(p_y, 0.0, im_h-1)
    return np.concatenate((p_x[:, np.newaxis], p_y[:, np.newaxis]), axis=1)

def crop2patch(im_p, p_h, p_w, p_overlap):
    """
    Get coordinates of upper-left point for image patch
    return: patch_list [X_upper-left, Y_upper-left, patch_width, patch_height]
    """
    im_h, im_w, _ = im_p
    x = np.arange(0, im_w-p_w, p_w-p_overlap)
    x = np.append(x, im_w-p_w)
    y = np.arange(0, im_h-p_h, p_h-p_overlap)
    y = np.append(y, im_h-p_h)
    X, Y = np.meshgrid(x, y)
    patch_list = [[i, j, p_w, p_h] for i, j in zip(X.flatten(), Y.flatten())]
    return patch_list

def polygon_in_bounding_box(polygon, bounding_box):
    """
    Returns True if all vertices of polygons are inside bounding_box
    :param polygon: [N, 2]
    :param bounding_box: [row_min, col_min, row_max, col_max]
    :return:
    """
    result = np.all(
        np.logical_and(
            np.logical_and(bounding_box[0] <= polygon[:, 0], polygon[:, 0] <= bounding_box[0] + bounding_box[2]),
            np.logical_and(bounding_box[1] <= polygon[:, 1], polygon[:, 1] <= bounding_box[1] + bounding_box[3])
        )
    )
    return result

def transform_poly_to_bounding_box(polygon, bounding_box):
    """
    Transform the original coordinates of polygon to bbox
    :param polygon: [N, 2]
    :param bounding_box: [row_min, col_min, row_max, col_max]
    :return:
    """
    transformed_polygon = polygon.copy()
    transformed_polygon[:, 0] -= bounding_box[0]
    transformed_polygon[:, 1] -= bounding_box[1]
    return transformed_polygon

def bmask_to_poly(b_im, simplify_ind, tolerance=1.8, ):
    """
    Convert binary mask to polygons
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    try:
        label_img = ski_label(b_im > 0)
    except:
        print('error')
    props = regionprops(label_img)
    for prop in props:
        prop_mask = np.zeros_like(b_im)
        prop_mask[prop.coords[:, 0], prop.coords[:, 1]] = 1
        padded_binary_mask = np.pad(prop_mask, pad_width=1, mode='constant', constant_values=0)
        contours, hierarchy = cv2.findContours(padded_binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 1:
            intp = []
            for contour, h in zip(contours, hierarchy[0]):
                contour = np.array([c.reshape(-1).tolist() for c in contour])
                # subtract pad
                contour -= 1
                contour = clip_by_bound(contour, b_im.shape[0], b_im.shape[1])
                if len(contour) > 3:
                    closed_c = np.concatenate((contour, contour[0].reshape(-1, 2)))
                    if h[3] < 0:
                        extp = [tuple(i) for i in closed_c]
                    else:
                        if cv2.contourArea(closed_c.astype(int)) > 10:
                            intp.append([tuple(i) for i in closed_c])
            poly = Polygon(extp, intp)
            if simplify_ind:
                poly = poly.simplify(tolerance=tolerance, preserve_topology=False)
                if isinstance(poly, Polygon):
                    polygons.append(poly)
                else:
                    for idx in range(len(poly.geoms)):
                        polygons.append(poly.geoms[idx])
        elif len(contours) == 1:
            contour = np.array([c.reshape(-1).tolist() for c in contours[0]])
            contour -= 1
            contour = clip_by_bound(contour, b_im.shape[0], b_im.shape[1])
            if len(contour) > 3:
                closed_c = np.concatenate((contour, contour[0].reshape(-1, 2)))
                poly = Polygon(closed_c)

            # simply polygon vertex
                if simplify_ind:
                    poly = poly.simplify(tolerance=tolerance, preserve_topology=False)
                if isinstance(poly, Polygon):
                    polygons.append(poly)
                else:
                    for idx in range(len(poly.geoms)):
                        polygons.append(poly.geoms[idx])
            # print(np.array(poly.exterior.coords).ravel().tolist())
            # in case that after "simplify", one polygon turn to multiply polygons
            # (pixels in polygon) are not connected
    return polygons

def rotate_image(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result

def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )

def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]

def rotate_crop(im, gt, crop_size, angle):
    h, w = im.shape[0:2]
    im_rotated = rotate_image(im, angle)
    gt_rotated = rotate_image(gt, angle)
    if largest_rotated_rect(w, h, math.radians(angle))[0] > crop_size:
        im_cropped = crop_around_center(im_rotated, crop_size, crop_size)
        gt_cropped = crop_around_center(gt_rotated, crop_size, crop_size)
    else:
        print('error')
        im_cropped = crop_around_center(im, crop_size, crop_size)
        gt_cropped = crop_around_center(gt, crop_size, crop_size)
    return im_cropped, gt_cropped

def lt_crop(im, gt, crop_size):
    im_cropped = im[0:crop_size, 0:crop_size, :]
    gt_cropped = gt[0:crop_size, 0:crop_size]
    return im_cropped, gt_cropped

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def original_processing(args):
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
    for g_id, label in enumerate(tqdm(input_label)):
        # read data
        label_info = [''.join(list(g)) for k, g in groupby(label, key=lambda x: x.isdigit())]
        label_name = label_info[0] + label_info[1]
        image_data = io.imread(os.path.join(input_image_path, label_name + '.tif'))
        gt_im_data = io.imread(os.path.join(input_gt_path, label_name + '.tif'))
        im_h, im_w, _ = image_data.shape

        if label_info[1] not in val_set and label_info[0] in cities:
            # for training set, split image to 512x512
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

                for p_im, p_gt in zip(p_ims, p_gts):
                    if np.sum(p_gt > 0) > 5:
                        p_polygons = bmask_to_poly(p_gt, 1)
                        for poly in p_polygons:
                            p_area = round(poly.area, 2)
                            if p_area > 0:
                                p_bbox = [poly.bounds[0], poly.bounds[1],
                                          poly.bounds[2]-poly.bounds[0], poly.bounds[3]-poly.bounds[1]]
                                if p_bbox[2] > 5 and p_bbox[3] > 5:
                                    p_seg = []
                                    coor_list = mapping(poly)['coordinates']
                                    for part_poly in coor_list:
                                        p_seg.append(np.asarray(part_poly).ravel().tolist())
                                    anno_info = {
                                        'id': train_ob_id,
                                        'image_id': train_im_id,
                                        'segmentation': p_seg,
                                        'area': p_area,
                                        'bbox': p_bbox,
                                        'category_id': 100,
                                        'iscrowd': 0
                                    }
                                    output_data_train['annotations'].append(anno_info)
                                    train_ob_id += 1
                    # get patch info
                    p_name = label_name + '-' + str(train_im_id) + '.tif'
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


def inria_htm_processing(args):
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
    for g_id, label in enumerate(tqdm(input_label)):
        # read data
        label_info = [''.join(list(g)) for k, g in groupby(label, key=lambda x: x.isdigit())]
        label_name = label_info[0] + label_info[1]
        raster = rasterio.open(os.path.join(input_image_path, label_name + '.tif'))
        raster_gt = rasterio.open(os.path.join(input_gt_path, label_name + '.tif'))
        im_h, im_w = raster.shape
        raster_shape = (im_h, im_w, 3)

        # not in 
        if label_info[1] not in val_set and label_info[0] in cities:
            # for training set, split image to 512x512
            patch_list = crop2patch(raster_shape, patch_width, patch_height, patch_overlap)
            print(label_name)
            for pid, pa in enumerate(patch_list):
                x_ul, y_ul, pw, ph = pa

                window = rasterio.windows.Window(x_ul, y_ul, pw, ph)
                p_gt = raster_gt.read(window=window)[0]
                p_im = raster.read(window=window).transpose(1,2,0)
                p_gts = []
                p_ims = []
                p_im_rd, p_gt_rd = lt_crop(p_im, p_gt, patch_size)
                p_gts.append(p_gt_rd)
                p_ims.append(p_im_rd)

                for p_im, p_gt in zip(p_ims, p_gts):
                    if np.sum(p_gt > 0) > 5:
                        gdf = raster_to_polygons(raster_gt, window, min_patch_objs=3, tolerance=1.2)
                        
                        if gdf is None or gdf.empty:
                            pass
                        else:
                            p_polygons = gdf.geometry.to_list()
                            for poly in p_polygons:
                                p_area = round(poly.area, 2)
                                if p_area > 0:
                                    p_bbox = [poly.bounds[0], poly.bounds[1],
                                            poly.bounds[2]-poly.bounds[0], poly.bounds[3]-poly.bounds[1]]
                                    if p_bbox[2] > 5 and p_bbox[3] > 5:
                                        p_seg = []
                                        coor_list = mapping(poly)['coordinates']
                                        for part_poly in coor_list:
                                            p_seg.append(np.asarray(part_poly).ravel().tolist())
                                        anno_info = {
                                            'id': train_ob_id,
                                            'image_id': train_im_id,
                                            'segmentation': p_seg,
                                            'area': p_area,
                                            'bbox': p_bbox,
                                            'category_id': 100,
                                            'iscrowd': 0
                                        }
                                        output_data_train['annotations'].append(anno_info)
                                        train_ob_id += 1
                    # get patch info
                    p_name = label_name + '-' + str(train_im_id) + '.tif'
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


def prepare_framefield_polygon(aligned_poly, window, transform):

    polygons = [Polygon(coords[:, ::-1]) for coords in aligned_poly if len(coords) > 3]
    extracted_gdf = gpd.GeoDataFrame(geometry=polygons)

    bbox = rasterio.windows.bounds(window, transform)
    bbox_polygon = box(*bbox)

    gdf_ori = canvasPoly2WorldCoord(extracted_gdf, transform)
    intersect_gdf = gdf_ori[gdf_ori.intersects(bbox_polygon)]
    valid_intersect_gdf = intersect_gdf[intersect_gdf['geometry'].is_valid]
    intersection_gdf = gpd.GeoDataFrame(geometry=valid_intersect_gdf.intersection(bbox_polygon))

    window_transform = rasterio.windows.transform(window, transform)
    subset_gdf = worldCoordPoly2CanvasPolygon2(intersection_gdf, window_transform)

    # Exploding a MultiPolygon into Polygons
    exploded_gdf = gpd.GeoDataFrame(columns=subset_gdf.columns)
    for index, row in subset_gdf.iterrows():
        geometries = explode_multipolygon(row.geometry)
        for geom in geometries:
            exploded_gdf = exploded_gdf.append(row, ignore_index=True)
            exploded_gdf.at[exploded_gdf.index[-1], 'geometry'] = geom

    area_gdf = exploded_gdf[exploded_gdf.area > 10]
    polys = area_gdf.geometry.tolist()
    return polys


def framefield_processing(args):
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
    framefield_aligned_path = args['framefield_aligned_path']

    train_ob_id = 0
    train_im_id = 0
    input_label = os.listdir(input_gt_path)
    for g_id, label in enumerate(tqdm(input_label)):
        # read data
        label_info = [''.join(list(g)) for k, g in groupby(label, key=lambda x: x.isdigit())]
        label_name = label_info[0] + label_info[1]
        # image_data = io.imread(os.path.join(input_image_path, label_name + '.tif'))
        # gt_im_data = io.imread(os.path.join(input_gt_path, label_name + '.tif'))
        raster = rasterio.open(os.path.join(input_image_path, label_name + '.tif'))
        raster_gt = rasterio.open(os.path.join(input_gt_path, label_name + '.tif'))
        aligned_poly = np.load(os.path.join(framefield_aligned_path, label_name + '.npy'), allow_pickle=True)
        im_h, im_w = raster.shape
        raster_shape = (im_h, im_w, 3)

        print(label_name)
        if label_info[1] not in val_set and label_info[0] in cities:
            # for training set, split image to 512x512
            patch_list = crop2patch(raster_shape, patch_width, patch_height, patch_overlap)
            for pid, pa in enumerate(patch_list):
                    x_ul, y_ul, pw, ph = pa
                    window = rasterio.windows.Window(x_ul, y_ul, pw, ph)

                    p_gt = raster_gt.read(window=window)[0]
                    p_im = raster.read(window=window).transpose(1,2,0)
                    p_gts = []
                    p_ims = []
                    p_im_rd, p_gt_rd = lt_crop(p_im, p_gt, patch_size)
                    p_gts.append(p_gt_rd)
                    p_ims.append(p_im_rd)
                    
                    for p_im, p_gt in zip(p_ims, p_gts):
                        if np.sum(p_gt > 0) > 5:
                            # maximum number of the nodes should be p_w-1
                            p_polygons = prepare_framefield_polygon(aligned_poly, 
                                                                    rasterio.windows.Window(x_ul, y_ul, pw-0.5, ph-0.5), 
                                                                    raster.transform)
                            for poly in p_polygons:
                                p_area = round(poly.area, 2)
                                if p_area > 0:
                                    p_bbox = [poly.bounds[0], poly.bounds[1],
                                            poly.bounds[2]-poly.bounds[0], poly.bounds[3]-poly.bounds[1]]
                                    if p_bbox[2] > 5 and p_bbox[3] > 5:
                                        p_seg = []
                                        coor_list = mapping(poly)['coordinates']
                                        for part_poly in coor_list:
                                            p_seg.append(np.asarray(part_poly).ravel().tolist())
                                        anno_info = {
                                            'id': train_ob_id,
                                            'image_id': train_im_id,
                                            'segmentation': p_seg,
                                            'area': p_area,
                                            'bbox': p_bbox,
                                            'category_id': 100,
                                            'iscrowd': 0
                                        }
                                        output_data_train['annotations'].append(anno_info)
                                        train_ob_id += 1
                        # get patch info
                        p_name = label_name + '-' + str(train_im_id) + '.tif'
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
    save_path = '/nas/k8s/dev/research/doyoungi/dataset/Inria_building/cocostyle_inria'

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
        'info': {'district': 'Inria', 'description': 'building footprints', 'contributor': 'whu'},
        'categories': [{'id': 100, 'name': 'building'}],
        'images': [],
        'annotations': [],
    }

    framefield_aligned_path = '/nas/k8s/dev/research/doyoungi/dataset/Inria_building/Inria_ImageDataset_PolygonLabel_FrameField/raw/train/gt_polygons'
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
        'framefield_aligned_path': framefield_aligned_path,
    }

    options = 'Inria_Htm'
    print(options)
    if options == 'HiSup':
        original_processing(args)
    elif options == 'FrameField':
        framefield_processing(args)
    elif options == 'Inria_Htm':
        inria_htm_processing(args)
    else:
        pass

    print('Done!')