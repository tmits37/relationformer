import os
import json
import shutil
import argparse
import subprocess
import numpy as np
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool

import sys
sys.path.append("/nas/k8s/dev/research/doyoungi/mmsegmentation_REST02_IE/tools/apls/")
from apls import xy2latlon


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


def relationformer_convert_to_rngdet_aplsformat(nodes, edges, patch_size=128):
    nodes = np.array(nodes) * patch_size
    nodes = np.round(nodes).astype('int64')
    node_connections = from_edge_to_connectivity(edges)
    connection_dicts = dict()
    for k, v in node_connections.items():
        connection_dicts[tuple(nodes[k])] = [tuple(nodes[x]) for x in v]

    nodes, edges = [], []
    nodemap, edge_map = {}, {}

    for k, v in connection_dicts.items():
        nodemap[k] = len(nodes)
        lat1,lon1 = xy2latlon(k[0], k[1])
        nodes.append([lat1,lon1])

    for k, v in connection_dicts.items():
        n1 = k 
        for n2 in v:
            if (n1,n2) in edge_map or (n2,n1) in edge_map:
                continue
            else:
                edge_map[(n1,n2)] = True 
            edges.append([nodemap[n1], nodemap[n2]])
    return nodes, edges


def caculate_apls_for_relationformer(
        filename, 
        tmp_dir, 
        patch_size=128
        ):
    
    global gopath
    basename = os.path.basename(filename)
    with open(filename, 'rb') as f:
        js = json.load(f)

    nodes_pred, edges_pred = relationformer_convert_to_rngdet_aplsformat(
                                js['pred_node'], js['pred_edge'], patch_size=patch_size)
    nodes, edges = relationformer_convert_to_rngdet_aplsformat(
                                js['gt_node'], js['gt_edge'], patch_size=patch_size)

    predjsonname = os.path.join(tmp_dir, 'apls_inter_processing', os.path.splitext(basename)[0] + ".json")
    json.dump([nodes_pred,edges_pred], open(predjsonname, "w"), indent=2)
    
    labeljsonname = os.path.join(tmp_dir, 'apls_inter_processing', os.path.splitext(basename)[0] + "label.json")
    json.dump([nodes,edges], open(labeljsonname, "w"), indent=2)
    
    aplstxt_path = os.path.join(tmp_dir, 'apls', os.path.splitext(basename)[0] + '.txt')

    command = ["go", "run", f"{gopath}", 
            f"{labeljsonname}", f"{predjsonname}", f"{aplstxt_path}"]
    
    subprocess.run(command)

    return None


def map_function(input_arg):
    return caculate_apls_for_relationformer(**input_arg)


def parse_args():
    parser = argparse.ArgumentParser(
        description='relationformer test (and eval) a model')
    parser.add_argument('--show-dir', help='folder contains inferenced imgs')
    parser.add_argument('--patch-size', default=128, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    gopath = "/nas/k8s/dev/research/doyoungi/mmsegmentation_REST02_IE/tools/apls/apls.go"

    filelist = glob(os.path.join(args.show_dir, 'graph') + f'/*.json')
    print("Number of inferenced files: {}".format(len(filelist)))

    tmp_dir = os.path.join(args.show_dir, 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(os.path.join(tmp_dir, 'apls'), exist_ok=True)
    os.makedirs(os.path.join(tmp_dir, 'apls_inter_processing'), exist_ok=True)

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

    # this for debug
    # for input_arg in tqdm(input_args):
    #     map_function(input_arg)
    #     break

    # concatenate and mean the apls
    apls, output_apls = [], []
    name_list = os.listdir(os.path.join(tmp_dir, 'apls'))
    for txtname in name_list :
        with open(os.path.join(tmp_dir, 'apls', txtname)) as f:
            lines = f.readlines()
        if 'NaN' in lines[0]:
            pass
        else:
            apls.append(float(lines[0].split(' ')[-1]))
            output_apls.append([txtname,float(lines[0].split(' ')[-1])])

    print("APLS over 0 / APLS 0 / Total files")
    print("{} / {} / {}".format(len(output_apls), len(filelist) - len(output_apls), len(filelist)))
    print("This file does not consider any files with an APLS value of zero.")
    print('APLS',np.sum(apls)/len(apls))
    with open(os.path.join(tmp_dir, 'apls.json'),'w') as jf:
        json.dump({'apls':output_apls,'final_APLS':np.mean(apls)},jf)

    shutil.rmtree(os.path.join(tmp_dir, 'apls'))
    shutil.rmtree(os.path.join(tmp_dir, 'apls_inter_processing'))