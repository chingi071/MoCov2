import os
import requests
import tarfile
import gdown
from tqdm import tqdm

data_file = 'data'
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
        for data in tqdm(response.iter_content(1024)):
            f.write(data)
print("imagenette2 has been downloaded.")
        
with tarfile.open(target_path, 'r:gz') as tar:
    for member in tqdm(tar.getmembers()):
        tar.extract(member=member, path=data_file)
print("imagenette2 has been decompressed.")

moco_v2_200ep_model = 'https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_200ep/moco_v2_200ep_pretrain.pth.tar'
model_path = os.path.join(model_file, "moco_v2_200ep_pretrain.pth.tar")
gdown.download(moco_v2_200ep_model, model_path)
print("model has been downloaded.")