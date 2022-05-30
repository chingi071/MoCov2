# MoCov2

## Dependency Setup

**Create new conda virtual environment**

```
conda create --name ssl python=3.8 -y
conda activate ssl
```

**Installation**

```
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 -c pytorch -y
git clone https://github.com/chingi071/MoCov2_Pytorch
pip install pandas matplotlib gdown
```

## Data preprocessing & pre-trained weight download

```
python get_data_model.py
```

## Unsupervised Training

```
python train.py --data-path data/imagenette2 --batch-size 256
```

## Linear 

```
python linear.py --data-path data/imagenette2 --batch-size 256 --pretrained weights/moco_v2_200ep_pretrain.pth.tar
```

## Reference

https://github.com/facebookresearch/moco
