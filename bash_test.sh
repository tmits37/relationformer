CONFIG=$1
CHECKPOINT=$2

pip3 uninstall -y scikit-image 

pip3 install -r requirements.txt
pip3 install tensorboard

cd ./models/ops
rm -rf build
rm -rf dist
rm -rf MultiScaleDeformableAttention.egg-info
python3 setup.py install
cd ../../

python3 test.py $CONFIG $CHECKPOINT