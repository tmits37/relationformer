CONFIG=$1
GPUS=$2

pip3 uninstall -y scikit-image

pip3 install -r requirements.txt
pip3 install tensorboard

python3 -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    train_TopDiG.py --config $CONFIG ${@:3}
