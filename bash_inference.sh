CONFIG=$1
CHECKPOINT=$2

pip3 uninstall -y scikit-image
pip3 install -r requirements.txt
pip3 install tensorboard

python3 inference_TopDiG.py $CONFIG $CHECKPOINT ${@:3}
