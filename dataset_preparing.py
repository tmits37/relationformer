import os
import matplotlib.pyplot as plt
import json
import numpy as np
from glob import glob
import rasterio

import rasterio
from rasterio.features import shapes
import geopandas as gpd

from shapely import affinity
from shapely.geometry import Polygon, LineString

from scipy.ndimage import gaussian_filter
from PIL import Image

from multiprocessing import Pool
from functools import partial
from tqdm import tqdm


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


def get_seed(raster, length, overlap):
    diff = length - overlap

    X_seed = np.arange(raster.height // diff + 1) * diff
    X_seed[-1] = raster.height - length

    Y_seed = np.arange(raster.width // diff + 1) * diff
    Y_seed[-1] = raster.width - length
    if abs(X_seed[-1] - X_seed[-2]) < int(length / 2):
        X_seed = np.delete(X_seed, [-2])
    if abs(Y_seed[-1] - Y_seed[-2]) < int(length / 2):
        Y_seed = np.delete(Y_seed, [-2])
    X_seed, Y_seed = np.meshgrid(X_seed, Y_seed)

    seed = []
    for x, y in zip(X_seed.ravel(), Y_seed.ravel()):
        seed.append(np.array([x, y]))
    return seed


def worldCoordPoly2CanvasPolygon(geom, geo_transform, xoff=0, yoff=0):
    # Extract the geo_transform parameters
    x_gsd, _, tl_x, _, y_gsd, tl_y, _, _, _ = list(geo_transform)

    # Invert the scale factors
    inv_x_gsd = 1 / x_gsd
    inv_y_gsd = 1 / y_gsd

    # Invert the translation, taking into account the offset
    inv_tl_x = -tl_x / x_gsd - xoff
    inv_tl_y = -tl_y / y_gsd - yoff

    # Apply the inverse affine transformation
    geom = affinity.affine_transform(
        geom, [inv_x_gsd, 0, 0, inv_y_gsd, inv_tl_x, inv_tl_y])

    return geom


def add_sample_points_to_edges(polygon, gap):
    # Initialize a list to hold the new points
    new_points = []

    # Iterate over edges of the polygon
    coords = list(polygon.exterior.coords)
    for i in range(len(coords) - 1):
        start_point = coords[i]
        end_point = coords[i + 1]
        
        # Create a line segment for the current edge
        line = LineString([start_point, end_point])

        # Add the start point of the edge
        new_points.append(start_point)

        # Calculate points along the edge
        current_distance = gap
        while current_distance < line.length:
            point = line.interpolate(current_distance)
            new_points.append((point.x, point.y))
            current_distance += gap

    # Ensure the polygon is closed by adding the start point at the end
    new_points.append(coords[0])

    # Create a new polygon with the additional points
    new_polygon = Polygon(new_points)

    return new_polygon


def non_max_suppression(coords, scores, threshold):
    """
    Perform non-maximum suppression on a set of points.

    :param coords: NumPy array of shape [N, 2] containing point coordinates.
    :param scores: NumPy array of length N containing scores for each point.
    :param threshold: Distance threshold for suppression.
    :return: Array of indices of points that are kept.
    """
    # Sort points by scores in descending order
    sorted_indices = np.argsort(-scores)
    selected_indices = []

    while len(sorted_indices) > 0:
        # Select the point with the highest score and remove it from the list
        current_index = sorted_indices[0]
        selected_indices.append(current_index)
        sorted_indices = sorted_indices[1:]

        # Compute distances from the current point to all others
        distances = np.linalg.norm(coords[sorted_indices] - coords[current_index], axis=1)

        # Only keep points that are further away than the threshold
        sorted_indices = sorted_indices[distances > threshold]

    return np.array(selected_indices)


def raster_to_polygons(raster, window, min_patch_objs=3, tolerance=1.2):
    ann = raster.read(1, window=window)
    window_transform = rasterio.windows.transform(window, raster.transform)

    results = ({'properties': {'raster_val': v}, 'geometry': s} for i, (s, v)  in enumerate(
                shapes(ann, mask=(ann==255), transform=window_transform)))

    geoms = list(results)
    
    if len(geoms) < min_patch_objs:
        return None
    else:        
        gdf  = gpd.GeoDataFrame.from_features(geoms)
        polys = gdf['geometry'].to_list()
        geom_poly = [worldCoordPoly2CanvasPolygon(p, window_transform).simplify(tolerance) for p in polys]
        gdf_ori = gpd.GeoDataFrame(geometry=geom_poly)
    return gdf_ori





def get_coords_from_densifing_points(gdf, gap_distance=10):
    p_gdf = gdf.copy()
    origin_coordinates = []
    polys = []
    for i, row in gdf.iterrows():
        geom = row.geometry
        new_geom = add_sample_points_to_edges(geom, gap_distance)
        polys.append(new_geom)

        x, y = geom.exterior.xy
        x, y = np.array(x), np.array(y)
        pos = np.stack([x,y], axis=1)
        pos = np.round(pos).astype('uint16')
        origin_coordinates.append(pos)
        
    origin_coordinates = np.concatenate(origin_coordinates, axis=0) # .tolist()
    p_gdf['geometry'] = polys

    coordinates = []
    polygons = []
    for i, row in p_gdf.iterrows():
        geom = row.geometry
        x, y = geom.exterior.xy
        x, y = np.array(x), np.array(y)
        pos = np.stack([x,y], axis=1)
        pos = np.round(pos).astype('uint16')

        coords_score = []
        for pt in pos:
            if pt.tolist() in origin_coordinates:
                coords_score.append(1)
            else:
                coords_score.append(100)
        coords_score = np.array(coords_score)

        nms_idx = non_max_suppression(pos, coords_score, int(gap_distance // 2))
        nms_idx = sorted(nms_idx)
        nms_coords = pos[nms_idx]
        # nms_coords = pos
        coordinates.append(nms_coords)
        
        nms_coords = np.concatenate([nms_coords, nms_coords[:1]], axis=0).tolist()
        nms_poly = Polygon(nms_coords)
        polygons.append(nms_poly)

    coordinates = np.concatenate(coordinates, axis=0)
    p_gdf['geometry'] = polygons
    return coordinates, p_gdf


def generate_heatmap(coords, shape, sigma=1):
    heatmap = np.zeros(shape[:2]).astype('uint8')
    y_coords, x_coords = coords[:,1], coords[:,0]
    y_coords[y_coords>=shape[0]] = shape[0] - 1
    x_coords[x_coords>=shape[1]] = shape[1] - 1
    heatmap[y_coords, x_coords] = 1

    heatmap = heatmap.astype('float32')
    blurred_heatmap = gaussian_filter(heatmap, sigma=sigma)

    cpo = blurred_heatmap.copy()
    cpo[y_coords, x_coords] = 0
    coords_value = max(np.min(blurred_heatmap[y_coords, x_coords]), np.max(cpo))

    blurred_heatmap[y_coords, x_coords] = coords_value
    results_heatmap = min_max_normalize(blurred_heatmap[...,np.newaxis], percentile=0)

    return results_heatmap


def run(rootdir, savedir, cfg_options, basename):
    # params
    min_patch_objs = cfg_options['min_patch_objs']
    tolerance = cfg_options['tolerance']
    gap_distance = cfg_options['gap_distance']
    sigma = cfg_options['sigma']
    length = cfg_options['length']
    overlap = cfg_options['length'] - cfg_options['stride']


    raster = rasterio.open(os.path.join(os.path.join(rootdir, 'gt'), basename))
    raster_img = rasterio.open(os.path.join(os.path.join(rootdir, 'images'), basename))
    seed = get_seed(raster, length=length, overlap=overlap)
    seed = np.stack(seed, axis=0)

    for sd in seed:
        x0, y0 = sd
        window = rasterio.windows.Window(x0, y0, length, length)
        img = raster_img.read(window=window)
        ann = raster.read(1, window=window)
        shape = img.shape[1:]

        gdf = raster_to_polygons(raster, window=window, min_patch_objs=min_patch_objs, tolerance=tolerance)
        if gdf is None:
            continue

        coords = get_coords_from_densifing_points(gdf, gap_distance=gap_distance)
        heatmap = generate_heatmap(coords, shape, sigma=sigma)

        # Saving
        save_basename = f"{basename[:-4]}_{x0}_{y0}.png"
        im = Image.fromarray(img.transpose(1,2,0))
        im_ann = Image.fromarray(ann)
        im_save = Image.fromarray(heatmap[...,0])

        im.save(os.path.join(savedir, 'img', save_basename))
        im_ann.save(os.path.join(savedir, 'ann', save_basename))
        im_save.save(os.path.join(savedir, 'heatmap', save_basename))
        


if __name__ == "__main__":

    cfg_options = {
        'min_patch_objs': 3,
        'tolerance': 1.2,
        'gap_distance': 10,
        'sigma': 1,
        'length': 300,
        'stride': 300,
    }

    rootdir = "/nas/Dataset/inria/AerialImageDataset/train"
    filelist = glob(os.path.join(rootdir, 'images') + '/*.tif')
    filelist = [os.path.basename(x) for x in filelist]
    cities = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
    val_set = [10, 20]
    test_filelist = []
    for city in cities:
        for val_num in val_set:
            test_filelist.append(f'{city}{val_num}.tif')
    train_filelist = [x for x in filelist if not x in test_filelist]


    # Train:
    train_savedir = "/nas/k8s/dev/research/doyoungi/dataset/Inria_building/train"
    os.makedirs(os.path.dirname(train_savedir), exist_ok=True)
    os.makedirs(train_savedir, exist_ok=True)
    os.makedirs(os.path.join(train_savedir, 'img'), exist_ok=True)
    os.makedirs(os.path.join(train_savedir, 'ann'), exist_ok=True)
    os.makedirs(os.path.join(train_savedir, 'heatmap'), exist_ok=True)

    train_partial_function = partial(run, rootdir, train_savedir, cfg_options)
    pool = Pool(4)
    for _ in tqdm(pool.imap_unordered(train_partial_function, train_filelist), total=len(train_filelist)):
        pass
    pool.close()
    pool.join()


    # test:
    test_savedir = "/nas/k8s/dev/research/doyoungi/dataset/Inria_building/test"
    os.makedirs(os.path.dirname(test_savedir), exist_ok=True)
    os.makedirs(test_savedir, exist_ok=True)
    os.makedirs(os.path.join(test_savedir, 'img'), exist_ok=True)
    os.makedirs(os.path.join(test_savedir, 'ann'), exist_ok=True)
    os.makedirs(os.path.join(test_savedir, 'heatmap'), exist_ok=True)

    test_partial_function = partial(run, rootdir, test_savedir, cfg_options)
    pool = Pool(4)
    for _ in tqdm(pool.imap_unordered(test_partial_function, train_filelist), total=len(train_filelist)):
        pass
    pool.close()
    pool.join()
