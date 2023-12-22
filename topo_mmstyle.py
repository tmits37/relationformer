import os
import json
import shutil
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool

import rasterio
import sys
sys.path.append("/nas/k8s/dev/research/doyoungi/mmsegmentation_REST02_IE/tools/apls/")

# from apls import xy2latlon, connection_dicts_of_rngdet_aplsformat
from topo import create_graph
import topo_utils as topo


MIN_LAT = 41.0 
MAX_LON = -71.0 


def from_edge_to_connectivity(edge):
    connectivity = {}
    for e in edge:
        node1, node2 = e
        if node1 in connectivity:
            connectivity[node1].append(node2)
        else:
            connectivity[node1] = [node2]
        if node2 in connectivity:
            connectivity[node2].append(node1)
        else:
            connectivity[node2] = [node1]            
    return connectivity


def caculate_topo_for_relationformer(filename, tmp_dir, patch_size=256):
    # fixed hyperparamters
    topo_interval = 0.00005
    matching_threshold = 0.00010
    r = 0.00150 # around 150 meters

    basename = os.path.basename(filename)
    topotxt_path = os.path.join(tmp_dir, 'topo', os.path.splitext(basename)[0] + '.txt')
    with open(filename, 'rb') as f:
        js = json.load(f)

    # prepare label-gt
    gt_node = np.array(js['gt_node']) * patch_size
    gt_node = np.round(gt_node).astype('int64')
    gt_edge = js['gt_edge']
    gt_node_connections = from_edge_to_connectivity(gt_edge)
    label_connection_dicts = dict()
    for k, v in gt_node_connections.items():
        label_connection_dicts[tuple(gt_node[k])] = [tuple(gt_node[x]) for x in v]

    # prepare pred
    pred_node = np.array(js['pred_node']) * patch_size
    pred_node = np.round(pred_node).astype('int64')
    pred_edge = js['pred_edge']
    pred_node_connections = from_edge_to_connectivity(pred_edge)
    pred_connection_dicts = dict()
    for k, v in pred_node_connections.items():
        pred_connection_dicts[tuple(pred_node[k])] = [tuple(pred_node[x]) for x in v]

    pred_graph, min_lat, max_lon = create_graph(pred_connection_dicts, MIN_LAT, MAX_LON)
    gt_graph, _, _= create_graph(label_connection_dicts, MIN_LAT, MAX_LON)

    region = [min_lat-300 * 1.0/111111.0, 
                MAX_LON-500 * 1.0/111111.0, 
                MIN_LAT+300 * 1.0/111111.0, 
                max_lon+500 * 1.0/111111.0]

    pred_graph.region = region
    gt_graph.region = region

    losm = topo.TOPOGenerateStartingPoints(gt_graph, 
                                            region=region,
                                            image="NULL", 
                                            check=False, 
                                            direction=False, 
                                            metaData=None)

    lmap = topo.TOPOGeneratePairs(pred_graph, 
                                    gt_graph,
                                    losm, 
                                    threshold=0.00010, 
                                    region=region)

    topoResult =  topo.TOPOWithPairs(pred_graph, 
                                        gt_graph, 
                                        lmap, 
                                        losm, 
                                        r=r, 
                                        step=topo_interval, 
                                        threshold=matching_threshold, 
                                        outputfile=topotxt_path, 
                                        one2oneMatching=True, 
                                        metaData=None)
    return None


def map_function(input_arg):
    return caculate_topo_for_relationformer(**input_arg)


def parse_args():
    parser = argparse.ArgumentParser(
        description='relationformer test (and eval) a model')
    parser.add_argument('--show-dir', help='folder contains inferenced imgs')
    parser.add_argument('--patch-size', default=128, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # jsonpath = '/nas/k8s/dev/research/doyoungi/git/relationformer/show_dirs/edge/graph/000001_209_73_2-Vegas-1173.json'
    # tmp_dir = "/nas/k8s/dev/research/doyoungi"
    # caculate_topo_for_relationformer(jsonpath, tmp_dir)

    args = parse_args()

    filelist = glob(os.path.join(args.show_dir, 'graph') + f'/*.json')
    print("Number of inferenced files: {}".format(len(filelist)))

    tmp_dir = os.path.join(args.show_dir, 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(os.path.join(tmp_dir, 'topo'), exist_ok=True)

    input_args = list()
    for idx, filename in enumerate(filelist):
        input_args.append({'filename': filename,
                          'tmp_dir': tmp_dir,
                          'patch_size': args.patch_size})

    pool = Pool(8)
    for _ in tqdm(pool.imap_unordered(map_function, input_args), total=len(input_args)):
        pass
    pool.close()
    pool.join()

    precision = []
    recall = []
    for file_name in os.listdir(os.path.join(tmp_dir, 'topo')):
        if '.txt' not in file_name:
            continue
        with open(os.path.join(tmp_dir, 'topo', file_name)) as f:
            lines = f.readlines()

        p = float(lines[-1].split(' ')[0].split('=')[-1])
        r = float(lines[-1].split(' ')[-1].split('=')[-1])
        if p + r:
            precision.append(p)
            recall.append(r)

    topo_f1 = 2*np.mean(precision)*np.mean(recall)/(np.mean(precision)+np.mean(recall))

    print("This file does not consider any files with an TOPO value of zero.")
    print('TOPO',topo_f1,'Precision',np.mean(precision),'Recall',np.mean(recall))
    with open(os.path.join(tmp_dir, 'topo.json'),'w') as jf:
        json.dump({'mean topo':[topo_f1,np.mean(precision),np.mean(recall)],
                   'prec':precision,'recall':recall,'f1':topo_f1}, jf)
        
    shutil.rmtree(os.path.join(tmp_dir, 'topo'))

