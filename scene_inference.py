import os
import json
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm


def scale_coordinates(coord, image_size, x0, y0):
    return int(coord[0] * image_size[0] + x0), int(coord[1] * image_size[1] + y0)


def to_img_and_seg_from_results(filelist, rootdir, image_size=(2048, 2048), patch_size=(128,128)):
    p = 5
    patch_side = patch_size[0]

    tot_img = np.zeros((image_size[0], image_size[1], 3))
    tot_seg = np.zeros((image_size[0], image_size[1], 1))
    tot_dat = np.zeros((image_size[0], image_size[1], 1))

    for filename in filelist:
        img = np.array(Image.open(os.path.join(rootdir, 'patch', filename)))
        seg = np.array(Image.open(os.path.join(rootdir, 'seg', filename)))
        # seg = np.array(Image.open(os.path.join(rootdir, 'seg_pred', filename)))
        # print(filename, seg.dtype, seg.shape, np.unique(seg))

        x0 = int(filename.split('_')[1])
        y0 = int(filename.split('_')[2])

        # print(x0, y0)
        tot_img[x0+p:x0+patch_side-p, y0+p:y0+patch_side-p] = img[p:-p,p:-p]
        tot_seg[x0+p:x0+patch_side-p, y0+p:y0+patch_side-p] += seg[p:-p,p:-p,np.newaxis]
        tot_dat[x0+p:x0+patch_side-p, y0+p:y0+patch_side-p] += 1

    tot_dat[tot_dat == 0] = 1
    tot_seg = tot_seg / tot_dat
    tot_seg = tot_seg.astype('uint8')
    tot_img = tot_img.astype('uint8')
    return tot_seg, tot_img


def to_graph_from_results(tot_img, 
                          filelist, 
                          rootdir,
                          patch_size=(128,128), 
                          option='pred', 
                          overwrite=False, 
                          single_patch=False):
    blank_image = tot_img.copy()

    filelist = sorted(filelist, key=lambda x: int(x.split('_')[2]))
    for filename in filelist:
        with open(os.path.join(rootdir, 'graph', filename)) as f:
            data = json.load(f)

        # if filename == 'data.json':
        #     continue
        if single_patch:
            blank_image = tot_img.copy()

        x0 = int(filename.split('_')[1])
        y0 = int(filename.split('_')[2])

        nodes = data[f'{option}_node']
        edges = data[f'{option}_edge']

        if overwrite:
            for edge in edges:
                node1 = scale_coordinates(nodes[edge[0]], patch_size, x0, y0)
                node2 = scale_coordinates(nodes[edge[1]], patch_size, x0, y0)
                cv2.line(blank_image, node1[::-1], node2[::-1], (255, 0, 0), 2) 

            for node in nodes:
                scaled_node = scale_coordinates(node, patch_size, x0, y0)
                cv2.circle(blank_image, scaled_node[::-1], 3, (0, 255, 0), -1)
                cv2.circle(blank_image, scaled_node[::-1], 3, (255, 255, 0), 1)


        else:
            palette_image = tot_img[x0:x0+patch_size[0], y0:y0+patch_size[1]].copy()
            for edge in edges:
                node1 = scale_coordinates(nodes[edge[0]], patch_size, 0, 0)
                node2 = scale_coordinates(nodes[edge[1]], patch_size, 0, 0)
                cv2.line(palette_image, node1[::-1], node2[::-1], (255, 0, 0), 2) 

            for node in nodes:
                scaled_node = scale_coordinates(node, patch_size, 0, 0)
                cv2.circle(palette_image, scaled_node[::-1], 3, (0, 255, 0), -1)
                cv2.circle(palette_image, scaled_node[::-1], 3, (255, 255, 0), 1)

            blank_image[int(x0):int(x0)+int(patch_size[0]), int(y0):int(y0)+int(patch_size[1])] = palette_image

        #  save
        if single_patch:
            os.makedirs(os.path.join(rootdir, 'inference_plt', 'single_patch'), exist_ok=True)
            blank_image = blank_image.astype('uint8')
            fig, ax = plt.subplots(1, 1, figsize=(9,9))
            # fig, ax = plt.subplots(2, 1, figsize=(18,36))
            ax.imshow(blank_image[x0:x0+patch_size[0], y0:y0+patch_size[0]])
            savename = os.path.join(rootdir, 'inference_plt', 'single_patch', f'{region_num}_{x0}_{y0}_result.png')
            plt.savefig(savename, dpi=200, facecolor='#eeeeee')


    blank_image = blank_image.astype('uint8')
    return blank_image


def parse_args():
    parser = argparse.ArgumentParser(
        description='relationformer inference model')
    parser.add_argument('--show-dir', type=str, required=True, help='savedir')
    parser.add_argument('--dataset', type=str, choices=['sat2graph', 'spacenet'],
                        default='sat2graph', help='define dataset')
    parser.add_argument('--patch-size', type=int, default=128)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--single-patch', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    rootdir = "/nas/k8s/dev/research/doyoungi/git/relationformer/show_dirs/road_rgb_2D"
    rootdir = args.show_dir
    filelist_p = os.listdir(os.path.join(rootdir, 'patch'))
    filelist_g = os.listdir(os.path.join(rootdir, 'graph'))

    if args.dataset == 'sat2graph':
        # region_num_list = ['05JUL15WV031100015JUL05162954']
        image_size = (2048, 2048)
        patch_size = (args.patch_size, args.patch_size)
        region_num_list = [8,9,19,28,29,39,48,49,59,68,69,79,88,89,99,108,109,119,128,129,139,148,149,159,168,169,179]
    elif args.dataset == 'spacenet':
        with open("/nas/k8s/dev/research/doyoungi/git/relationformer/data/spacenet3/data_split.json", "r") as file:
            split_info = json.load(file)
        image_size = (400, 400)
        patch_size = (args.patch_size, args.patch_size)
        region_num_list = ['-'.join(x.split('_')[1:]) for x in split_info['test']]
    else:
        raise AttributeError

    os.makedirs(os.path.join(rootdir, 'inference_plt'), exist_ok=True)
    for region_num in tqdm(region_num_list):
        tmp = []
        for a in filelist_p:
            if str(a.split('.')[0].split('_')[-1]) == str(region_num):
                tmp.append(a)

        filelist_p_region = tmp
        tmp = []
        for b in filelist_g:
            if str(b.split('.')[0].split('_')[-1]) == str(region_num):
                tmp.append(b)
                
        filelist_g_region = tmp

        tot_seg, tot_img = to_img_and_seg_from_results(filelist_p_region, rootdir, image_size, patch_size)
        tot_graph = to_graph_from_results(tot_img, 
                                          filelist_g_region, 
                                          rootdir, 
                                          patch_size, 
                                          option='pred', 
                                          overwrite=args.overwrite,
                                          single_patch=args.single_patch)

        if not args.single_patch:
            gt_tot_graph = to_graph_from_results(tot_img, 
                                                 filelist_g_region, 
                                                 rootdir, 
                                                 patch_size, 
                                                 option='gt', 
                                                 overwrite=True,
                                                 single_patch=False)

            fig, ax = plt.subplots(1, 2, figsize=(18,9))
            # fig, ax = plt.subplots(2, 1, figsize=(18,36))
            ax[0].imshow(tot_graph)
            ax[1].imshow(gt_tot_graph)
            savename = os.path.join(rootdir, 'inference_plt', f'{region_num}_result.png')
            plt.savefig(savename, dpi=200, facecolor='#eeeeee')

            # fig, ax = plt.subplots(1, 2, figsize=(18,9))
            # ax[0].imshow(tot_seg)
            # ax[1].imshow(gt_tot_graph)
            # os.makedirs(os.path.join(rootdir, 'inference_plt_seg'), exist_ok=True)
            # savename = os.path.join(rootdir, 'inference_plt_seg', f'{region_num}_result.png')
            # plt.savefig(savename, dpi=200, facecolor='#eeeeee')