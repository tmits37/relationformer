import math
import imageio.v2 as imageio
import pyvista
import numpy as np
import pickle
import os
from tqdm import tqdm
from multiprocessing import Pool


def angle(v1, v2):
    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    return np.arccos(np.clip(dot_product, a_min=-1, a_max=1))


def convert_graph(graph):
    node_list = []
    edge_list = []
    for n, v in graph.items():
        node_list.append(n)
    node_array = np.array(node_list)

    for ind, (n, v) in enumerate(graph.items()):
        for nei in v:
            idx = node_list.index(nei)
            edge_list.append(np.array((ind, idx)))
    edge_array = np.array(edge_list)
    return node_array, edge_array


def neighbor_transpos(n_in):
    n_out = {}

    for k, v in n_in.items():
        nk = (k[1], k[0])
        nv = []

        for _v in v:
            nv.append((_v[1], _v[0]))

        n_out[nk] = nv
    return n_out


def neighbor_to_integer(n_in):
    n_out = {}

    for k, v in n_in.items():
        nk = (int(k[0]), int(k[1]))

        if nk in n_out:
            nv = n_out[nk]
        else:
            nv = []

        for _v in v:
            new_n_k = (int(_v[0]), int(_v[1]))

            if new_n_k in nv:
                pass
            else:
                nv.append(new_n_k)

        n_out[nk] = nv
    return n_out


def save_input(
    path,
    idx,
    lefttop_x,
    lefttop_y,
    region_name,
    patch,
    patch_seg,
    patch_coord,
    patch_edge,
):
    """[summary]

    Args:
        patch ([type]): [description]
        patch_coord ([type]): [description]
        patch_edge ([type]): [description]
    """
    imageio.imwrite(
        path
        + "raw/sample_"
        + str(idx).zfill(6)
        + "_"
        + str(lefttop_x)
        + "_"
        + str(lefttop_y)
        + "_"
        + str(region_name)
        + "_data.png",
        patch,  # 넘파이 어레이
    )
    imageio.imwrite(
        path
        + "seg/sample_"
        + str(idx).zfill(6)
        + "_"
        + str(lefttop_x)
        + "_"
        + str(lefttop_y)
        + "_"
        + str(region_name)
        + "_seg.png",
        patch_seg,
    )

    # vertices, faces, _, _ = marching_cubes_lewiner(patch)
    # vertices = vertices/np.array(patch.shape)
    # faces = np.concatenate((np.int32(3*np.ones((faces.shape[0],1))), faces), 1)

    # mesh = pyvista.PolyData(vertices)
    # mesh.faces = faces.flatten()
    # mesh.save(path+'mesh/sample_'+str(idx).zfill(4)+'_segmentation.stl')

    patch_edge = np.concatenate(
        (np.int32(2 * np.ones((patch_edge.shape[0], 1))), patch_edge), 1
    )
    mesh = pyvista.PolyData(patch_coord)
    # print(patch_edge.shape)
    mesh.lines = patch_edge.flatten()
    mesh.save(
        path
        + "vtp/sample_"
        + str(idx).zfill(6)
        + "_"
        + str(lefttop_x)
        + "_"
        + str(lefttop_y)
        + "_"
        + str(region_name)
        + "_graph.vtp"
    )


def prune_patch(patch_coord_list, patch_edge_list, dense=False, MAX_TOKENS=80):
    """[summary]

    Args:
        patch_list ([type]): [description]
        patch_coord_list ([type]): [description]
        patch_edge_list ([type]): [description]

    Returns:
        [type]: [description]
    """
    mod_patch_coord_list = []
    mod_patch_edge_list = []

    for coord, edge in zip(patch_coord_list, patch_edge_list):
        # find largest graph segment in graph and in skeleton and see if they match
        dist_adj = np.zeros((coord.shape[0], coord.shape[0]))
        dist_adj[edge[:, 0], edge[:, 1]] = np.sum(
            (coord[edge[:, 0], :] - coord[edge[:, 1], :]) ** 2, 1
        )
        dist_adj[edge[:, 1], edge[:, 0]] = np.sum(
            (coord[edge[:, 0], :] - coord[edge[:, 1], :]) ** 2, 1
        )

        # straighten the graph by removing redundant nodes
        start = True
        node_mask = np.ones(coord.shape[0], dtype=np.bool_)
        while start:
            degree = (dist_adj > 0).sum(1)
            deg_2 = list(np.where(degree == 2)[0])
            if len(deg_2) == 0:
                start = False
            for n, idx in enumerate(deg_2):
                deg_2_neighbor = np.where(dist_adj[idx, :] > 0)[0]

                p1 = coord[idx, :]
                p2 = coord[deg_2_neighbor[0], :]
                p3 = coord[deg_2_neighbor[1], :]
                l1 = p2 - p1
                l2 = p3 - p1
                node_angle = angle(l1, l2) * 180 / math.pi

                if (dense==True) and (len(patch_coord_list[0]) < MAX_TOKENS):
                    if n == len(deg_2) - 1:
                        start = False
                else:
                    if node_angle > 175: # default = 160
                        node_mask[idx] = False
                        dist_adj[deg_2_neighbor[0], deg_2_neighbor[1]] = np.sum(
                            (p2 - p3) ** 2
                        )
                        dist_adj[deg_2_neighbor[1], deg_2_neighbor[0]] = np.sum(
                            (p2 - p3) ** 2
                        )

                        dist_adj[idx, deg_2_neighbor[0]] = 0.0
                        dist_adj[deg_2_neighbor[0], idx] = 0.0
                        dist_adj[idx, deg_2_neighbor[1]] = 0.0
                        dist_adj[deg_2_neighbor[1], idx] = 0.0
                        break
                    elif n == len(deg_2) - 1:
                        start = False

        new_coord = coord[node_mask, :]
        new_dist_adj = dist_adj[np.ix_(node_mask, node_mask)]
        new_edge = np.array(np.where(np.triu(new_dist_adj) > 0)).T

        mod_patch_coord_list.append(new_coord)
        mod_patch_edge_list.append(new_edge)

    return mod_patch_coord_list, mod_patch_edge_list


def patch_extract(save_path, image, seg, mesh, dense=False, cfg_options={}):
    """[summary]

    Args:
        image ([type]): [description]
        coordinates ([type]): [description]
        lines ([type]): [description]
        patch_size (tuple, optional): [description]. Defaults to (64,64,64).
        num_patch (int, optional): [description]. Defaults to 2.

    Returns:
        [type]: [description]
    """
    
    image_id = cfg_options['image_id']
    region_name = cfg_options['region_name']
    stride = cfg_options['stride']
    patch_size = cfg_options['patch_size']
    pad = cfg_options['pad']
    MAX_TOKENS = cfg_options['MAX_TOKENS']

    p_h, p_w, _ = patch_size  # 512, 512, 1
    pad_h, pad_w, _ = pad  # 5, 5, 0

    p_h = p_h - 2 * pad_h  # 502
    p_w = p_w - 2 * pad_w  # 502

    h, w, d = image.shape  # 2048, 2048

    # 좌표 간격 계산
    a, b = 5, h - 5 - p_h
    c = (b - a) // stride
    c += 1

    x_ = np.int32(np.linspace(5, h - 5 - p_h, c))
    y_ = np.int32(np.linspace(5, w - 5 - p_w, c))

    ind = np.meshgrid(x_, y_, indexing="ij")
    # Center Crop based on foreground

    for start in list(np.array(ind).reshape(2, -1).T):
        # print(image.shape, seg.shape)
        start = np.array((start[0], start[1], 0))
        end = start + np.array(patch_size) - 1 - 2 * np.array(pad)

        patch = np.pad(
            image[start[0] : start[0] + p_h, start[1] : start[1] + p_w, :],
            ((pad_h, pad_h), (pad_w, pad_w), (0, 0)),
        )
        patch_list = [patch]

        patch_seg = np.pad(
            seg[
                start[0] : start[0] + p_h,
                start[1] : start[1] + p_w,
            ],
            ((pad_h, pad_h), (pad_w, pad_w)),
        )
        seg_list = [patch_seg]

        # collect all the nodes
        bounds = [start[0], end[0], start[1], end[1], -0.5, 0.5]

        # When crinkle=True, There will be no additional points 
        # on boundary edge lines
        clipped_mesh = mesh.clip_box(bounds, invert=False, crinkle=False)

        patch_coordinates = np.float32(np.asarray(clipped_mesh.points))
        patch_edge = clipped_mesh.cells[
            np.sum(clipped_mesh.celltypes == 1) * 2 :
        ].reshape(-1, 3)

        patch_coord_ind = np.where(
            (
                np.prod(patch_coordinates >= start, 1)
                * np.prod(patch_coordinates <= end, 1)
            )
            > 0.0
        )
        patch_coordinates = patch_coordinates[
            patch_coord_ind[0], :
        ]  # all coordinates inside the patch
        patch_edge = [
            tuple(l)
            for l in patch_edge[:, 1:]
            if l[0] in patch_coord_ind[0] and l[1] in patch_coord_ind[0]
        ]

        temp = np.array(
            patch_edge
        ).flatten()  # flatten all the indices of the edges which completely lie inside patch
        temp = [
            np.where(patch_coord_ind[0] == ind) for ind in temp
        ]  # remap the edge indices according to the new order
        patch_edge = np.array(temp).reshape(
            -1, 2
        )  # reshape the edge list into previous format

        if patch_coordinates.shape[0] < 2 or patch_edge.shape[0] < 1:
            continue
        # concatenate final variables
        patch_coordinates = (patch_coordinates - start + np.array(pad)) / np.array(
            patch_size
        )
        patch_coord_list = [patch_coordinates]  # .to(device))
        patch_edge_list = [patch_edge]  # .to(device))

        if dense:
            if len(patch_coord_list[0]) < 6:
                # Densify graph 사용시 patch 내에 sample points 수가 임계값보다 적을경우
                # Class imbalance 문제에 대응하기 위하여 제거한다.
                continue

        mod_patch_coord_list, mod_patch_edge_list = prune_patch(
            patch_coord_list, patch_edge_list, dense=dense, MAX_TOKENS=MAX_TOKENS
        )

        if (len(mod_patch_coord_list[0]) > MAX_TOKENS) or (len(mod_patch_edge_list[0]) > MAX_TOKENS):
            img_path = "raw/sample_"  + str(image_id).zfill(6) + "_" + str(start[0]) + "_" + str(start[1]) + "_" + str(region_name) + "_data.png"
            print(len(mod_patch_coord_list[0]), len(mod_patch_edge_list[0]), img_path)
            continue

        # save data
        for patch, patch_seg, patch_coord, patch_edge in zip(
            patch_list, seg_list, mod_patch_coord_list, mod_patch_edge_list
        ):  # 반복 한번짜리 반복문
            save_input(
                save_path,
                image_id,  # 1부터 단순 넘버링
                start[0],  # x0
                start[1],  # y0
                region_name,  # region_name
                patch,
                patch_seg,
                patch_coord,
                patch_edge,
            )


def generate_patches(input_args):
    save_path, file_name, raw_file, seg_file, vtk_file, image_id, dense, dataset, cfg_options = input_args.values()
    try:
        sat_img = imageio.imread(raw_file + ".png")
    except:
        sat_img = imageio.imread(raw_file + ".jpg")
    try:
        with open(vtk_file, "rb") as f:
            graph = pickle.load(f)
        node_array, edge_array = convert_graph(graph)
        if dataset == "spacenet":
            node_array[:, 0] = 400 - node_array[:, 0]

        gt_seg = imageio.imread(seg_file)
        patch_coord = np.concatenate(  # 에러가 나는 부분
            (node_array, np.int32(np.zeros((node_array.shape[0], 1)))), 1
        )
        mesh = pyvista.PolyData(patch_coord)
        patch_edge = np.concatenate(
            (np.int32(2 * np.ones((edge_array.shape[0], 1))), edge_array), 1
        )
        mesh.lines = patch_edge.flatten()
        region_name = "-".join(file_name.split("_")[1:4])  # spacenet 지역 이름 기준

        cfg_options['image_id'] = image_id
        cfg_options['region_name'] = region_name

        patch_extract(save_path, sat_img, gt_seg, mesh, dense, cfg_options=cfg_options)
    except KeyboardInterrupt as e:
        print(e)
        return
    except Exception as e:
        print("Error file name:", raw_file)
        print("Error:", e)
        return


def run(input_args, multiproc=True):
    if multiproc:
        print("Multiprocessing...")
        num_thread = 8
        with Pool(processes=num_thread) as pool:
            for _ in tqdm(
                pool.imap_unordered(generate_patches, input_args), total=len(input_args)
            ):
                pass
    else:
        for input_arg in input_args:
            generate_patches(input_arg)


