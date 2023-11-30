CONFIG=$1
GPUS=$2

pip3 uninstall -y scikit-image 

pip3 install -r pl_requirements.txt
pip3 install tensorboard

cd ./models/ops
rm -rf build
rm -rf dist
rm -rf MultiScaleDeformableAttention.egg-info
python3 setup.py install
cd ../../

python3 -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    pl_train.py \
    --config $CONFIG \
    # --launcher pytorch ${@:3}
