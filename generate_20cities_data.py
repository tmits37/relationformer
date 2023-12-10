import json
import os
import argparse

from generate_data import run 


def generate_data_folder_spacenet3(root_dir, file_list, mode="train", dense=False, multiproc=True, cfg_options={}):
    patch_length = cfg_options['patch_length']
    # 저장 경로 만들기
    path = f"./data/spacenet3/{mode}_data_{patch_length}_dense/"  # spacenet
    if not os.path.isdir(path):
        os.makedirs(path)
        os.makedirs(path + "/raw")
        os.makedirs(path + "/seg")
        os.makedirs(path + "/vtp")
    else:
        if not os.path.isdir(path + "/raw"):
            os.makedirs(path + "/raw")
        if not os.path.isdir(path + "/seg"):
            os.makedirs(path + "/seg")
        if not os.path.isdir(path + "/vtp"):
            os.makedirs(path + "/vtp")
        # raise Exception(f"{mode.capitalize()} folder already exists")
        print(f"{mode.capitalize()} folder already exists")
    print(f"Preparing {mode.capitalize()} Data")
    # args 리스트 만들기
    input_args = list()
    for idx, file_name in enumerate(file_list):
        raw_file = root_dir + file_name + "__rgb"
        # seg_file = root_dir + file_name + "__gt_graph_dense.png"  # 둘 중 하나로
        seg_file = root_dir + file_name + "__gt.png"
        vtk_file = root_dir + file_name + "__gt_graph_dense.p"
        input_args.append(
            {
                "save_path": path,
                "file_name": file_name,
                "raw_file": raw_file,
                "seg_file": seg_file,
                "vtk_file": vtk_file,
                "idx": idx,
                "dense": dense,
                "dataset": 'spacenet',
                "cfg_options": cfg_options,
            }
        )
    return input_args


def generate_data_folder_sat2graph(root_dir, file_list, mode="train", dense=False, multiproc=True, cfg_options={}):
    patch_length = cfg_options['patch_length']
    # 저장 경로 만들기
    path = f"./data/20cities/{mode}_data_{patch_length}_dense/"  # spacenet
    # path = f"./data/20cities/test/"
    if not os.path.isdir(path):
        os.makedirs(path)
        os.makedirs(path + "/raw")
        os.makedirs(path + "/seg")
        os.makedirs(path + "/vtp")
    else:
        if not os.path.isdir(path + "/raw"):
            os.makedirs(path + "/raw")
        if not os.path.isdir(path + "/seg"):
            os.makedirs(path + "/seg")
        if not os.path.isdir(path + "/vtp"):
            os.makedirs(path + "/vtp")
        # raise Exception(f"{mode.capitalize()} folder already exists")
        print(f"{mode.capitalize()} folder already exists")
    print(f"Preparing {mode.capitalize()} Data")
    # args 리스트 만들기
    input_args = list()
    for idx, file_name in enumerate(file_list):
        raw_file = root_dir + file_name + "_sat"
        seg_file = root_dir + file_name + "_gt.png"
        vtk_file = root_dir + file_name + "_refine_gt_graph.p"
        input_args.append(
            {
                "save_path": path,
                "file_name": file_name,
                "raw_file": raw_file,
                "seg_file": seg_file,
                "vtk_file": vtk_file,
                "image_id": idx+1,
                "dense": dense,
                "dataset": 'sat2graph',
                "cfg_options": cfg_options
            })

    return input_args


def prepare_sat2graph(p=256, dense=False):
    cfg_options={
        'MAX_TOKENS': 128,
        'patch_size': [p, p, 1],
        'pad': [5,5,0],
        'patch_length': p,
        'stride': int(p * 0.5) 
        }
    print(cfg_options)

    indrange_train = []
    indrange_test = []

    for x in range(180):
        if x % 10 < 8:
            indrange_train.append(x)

        if x % 10 == 9:
            indrange_test.append(x)

        if x % 20 == 18:
            indrange_train.append(x)

        if x % 20 == 8:
            indrange_test.append(x)

    root_dir = "./data/20cities/"

    filelist_train = [f"region_{x}" for x in indrange_train]
    filelist_test = [f"region_{x}" for x in indrange_test]

    test_input_args = generate_data_folder_sat2graph(root_dir, filelist_test, mode="test", dense=dense, cfg_options=cfg_options)
    train_input_args = generate_data_folder_sat2graph(root_dir, filelist_train, mode="train", dense=dense, cfg_options=cfg_options)

    return test_input_args, train_input_args


def prepare_spacenet3(p=256, dense=False):

    with open("./data/spacenet3/data_split.json", "r") as file:
        split_info = json.load(file)

    cfg_options={
        'MAX_TOKENS': 128,
        'patch_size': [p, p, 1],
        'pad': [5,5,0],
        'patch_length': p,
        'stride': int(p * 0.5) 
        }

    root_dir = "./data/spacenet3/rgb/"

    filelist_train = []
    filelist_test = []
    for split_type in split_info:
        if split_type == "test":
            filelist_test = split_info[split_type]
        elif split_type == "train" or split_type == "validation":
            print(len(filelist_train))  # check
            filelist_train.extend(split_info[split_type])
            print(len(filelist_train))  # check

    test_input_args = generate_data_folder_spacenet3(root_dir, filelist_test, mode="test", dense=dense, cfg_options=cfg_options)
    train_input_args = generate_data_folder_spacenet3(root_dir, filelist_train, mode="train", dense=dense, cfg_options=cfg_options)

    return test_input_args, train_input_args


def parse_args():
    parser = argparse.ArgumentParser(
        description='generate_relationformer_models')
    parser.add_argument('--patch-size', type=int, default=128)
    parser.add_argument('--dataset', type=str, choices=['sat2graph', 'spacenet'],
                        default='sat2graph', help='define dataset')
    parser.add_argument('--dense', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    if args.dataset == 'sat2graph':
        test_input_args, train_input_args = prepare_sat2graph(p=args.patch_size, dense=args.dense)
    elif args.dataset == 'spacenet':
        test_input_args, train_input_args = prepare_spacenet3(p=args.patch_size, dense=args.dense)
    else:
        raise AttributeError

    run(test_input_args, multiproc=True)
    run(train_input_args, multiproc=True)
