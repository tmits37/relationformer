CONFIG=$1
CHECKPOINT=$2
SHOW_DIR=$3
PATCH_SIZE=$4
DATASET=$5

# pip3 uninstall -y scikit-image 

# pip3 install -r requirements.txt
# pip3 install tensorboard

# cd ./models/ops
# rm -rf build
# rm -rf dist
# rm -rf MultiScaleDeformableAttention.egg-info
# python3 setup.py install
# cd ../../

python3 inference.py $CONFIG $CHECKPOINT --show-dir $SHOW_DIR
python3 scene_inference.py --show-dir $SHOW_DIR --dataset $DATASET --patch-size $PATCH_SIZE ${@:6}