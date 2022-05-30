import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class MoCov2(nn.Module):
    def __init__(self, arch='resnet50', feature_dim=128, moco_momentum=0.999, mlp=True):
        super(MoCo, self).__init__()
        
        self.m = moco_momentum                
        
        self.encoder_q = models.__dict__[arch](num_classes=feature_dim)
        self.encoder_k = models.__dict__[arch](num_classes=feature_dim)
                
        if mlp:
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)
            
        for (param_q, param_k) in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        
    def forward(self, im_q, im_k):
        query = self.encoder_q(im_q)
        query = nn.functional.normalize(query, dim=1)
        
        with torch.no_grad():
            for (param_q, param_k) in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_k.data * self.m + param_q.data * (1. - self.m))
            
            # shuffle BN
            idx_shuffle = torch.randperm(im_k.size(0)).cuda()
            idx_unshuffle = torch.argsort(idx_shuffle)
            
            key = self.encoder_k(im_k[idx_shuffle])
            key = nn.functional.normalize(key, dim=1)
            key = key[idx_unshuffle]
            
        return query, key
