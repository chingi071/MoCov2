import os
import requests
import tarfile
import gdown


data_file = 'data1'
model_file ='weights'

if not os.path.exists(data_file):
    os.makedirs(data_file)

if not os.path.exists(model_file):
    os.makedirs(model_file)
    
imagenette2_url = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz'
tgz_name = os.path.split(imagenette2_url)[-1]
target_path = os.path.join(data_file, tgz_name)

response = requests.get(imagenette2_url, stream=True)
if response.status_code == 200:
    with open(target_path, 'wb') as f:
        f.write(response.raw.read())
        
tar = tarfile.open(target_path, 'r:gz')
tar.extractall(path=data_file)

moco_v2_200ep_model = 'https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_200ep/moco_v2_200ep_pretrain.pth.tar'
model_path = os.path.join(model_file, "moco_v2_200ep_pretrain.pth.tar")
gdown.download(moco_v2_200ep_model, model_path)