CONFIG=$1
GPUS=$2

pip3 uninstall -y scikit-image 

pip3 install -r requirements.txt
pip3 install tensorboard

cd ./models/ops
rm -rf build
rm -rf dist
rm -rf MultiScaleDeformableAttention.egg-info
python3 setup.py install
cd ../../

# segmentations...
cd /nas/k8s/dev/research/doyoungi/mmsegmentation_REST02_IE
pip3 install openmim && 
mim uninstall -y mmdet mmrotate mmcv-full &&
mim install mmcv-full==1.6.2 && 
mim install -e .
cd /nas/k8s/dev/research/doyoungi/git/relationformer

python3 -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    train.py \
    --config $CONFIG \