apt-get update 
apt-get install -y git libgl1-mesa-glx libglib2.0-0 
pip install -r requirements.txt
# download models from modelscope
modelscope download --model facebook/dinov2-giant --local_dir ./pretrained_models/dinov2/giant
modelscope download --model google/siglip-so400m-patch14-384 --local_dir ./pretrained_models/google-siglip-so400m-patch14-384