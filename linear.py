import os
import argparse
import pandas as pd
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
from collections import OrderedDict

import utils

def train_epoch(epoch, model, train_loader, optimizer, criterion):
    model.eval()
    
    running_loss, running_acc1, running_acc5, running_count = 0.0, 0.0, 0.0, 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        
        output = model(data)
        loss = criterion(output, target)
        
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        
        running_count += data.shape[0]
        running_loss += loss.item() * data.shape[0]
        running_acc1 += acc1.item() * data.shape[0]
        running_acc5 += acc5.item() * data.shape[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    train_loss_value = running_loss / running_count
    train_acc1_value = running_acc1 / running_count
    train_acc5_value = running_acc5 / running_count
                    
    return train_loss_value, train_acc1_value, train_acc5_value

def valid_epoch(epoch, model, valid_loader, criterion):
    global best_valid_acc
    model.eval()
    
    running_loss, running_acc1, running_acc5, running_count = 0.0, 0.0, 0.0, 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(valid_loader):
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            output = model(data)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

            running_count += data.shape[0]
            running_loss += loss.item() * data.shape[0]
            running_acc1 += acc1.item() * data.shape[0]
            running_acc5 += acc5.item() * data.shape[0]
    
    valid_loss_value = running_loss / running_count
    valid_acc1_value = running_acc1 / running_count
    valid_acc5_value = running_acc5 / running_count
    
    if epoch % 10 == 0:
        model_path = os.path.join(args.model_saved_path, 'model_{:04d}.pth'.format(epoch))
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict(),}, model_path)
            
    if valid_acc1_value > best_valid_acc:
        model_path = os.path.join(args.model_saved_path, 'model_best.pth')
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict(),}, model_path)
        best_valid_acc = valid_acc1_value
                    
    return valid_loss_value, valid_acc1_value, valid_acc5_value
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-path', default='', type=str,
                        help='path to dataset')
    parser.add_argument('--num-epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--model-saved-path', default='workdirs_linear', type=str,
                        help='Specify the file path where the model is saved (default: workdirs_linear)')
    parser.add_argument('--batch-size', default=256, type=int,
                        help='Specify batch size for training (default: 256).')
    parser.add_argument('--num-workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--lr', default=30.0, type=float,
                        help='Specify initial learning rate (default: 30).')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Specify initial lmomentum (default: 0.9).')
    parser.add_argument('--weight-decay', default=0., type=float,
                        help='Specify initial weight decay (default: 0).')
    parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int,
                        help='learning rate schedule (when to drop lr by 10x, default: [60, 80])')
    parser.add_argument('--arch', default='resnet50', type=str,
                        help='Specify model architecture (default: resnet50).')
    parser.add_argument('--cos', default=0, type=int,
                        help='use cosine lr schedule (default: 0)')  
    parser.add_argument('--pretrained', default='', type=str,
                        help='path to moco pretrained checkpoint')
   
    args = parser.parse_args()
    
    if args.data_path == '':
        print("data_path is null...")
        assert args.data_path
        
    try:
        if not os.path.exists(args.model_saved_path):
            os.mkdir(args.model_saved_path)
    except:
        pass
    
    train_data_path = os.path.join(args.data_path, 'train')
    valid_data_path = os.path.join(args.data_path, 'val')
    
    train_dataset = torchvision.datasets.ImageFolder(train_data_path, 
                                                     transform=transforms.Compose([
                                                    transforms.RandomResizedCrop(224),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                ]))
    
    valid_dataset = torchvision.datasets.ImageFolder(valid_data_path, 
                                                     transform=transforms.Compose([
                                                    transforms.Resize(256),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                ]))     
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                num_workers=args.num_workers, pin_memory=True)
    
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                                num_workers=args.num_workers, pin_memory=True)
    
    model = models.__dict__[args.arch]().cuda()
    
    if args.pretrained:
        if os.path.exists('moco_v2_200ep_pretrain.pth.tar'):
            state_dict = OrderedDict()

            checkpoint = torch.load(args.pretrained, map_location="cpu")
            checkpoint_state_dict = checkpoint['state_dict']

            for k, v in checkpoint_state_dict.items():
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    state_dict[k[len('module.encoder_q.'):]] = v

            model.load_state_dict(state_dict, strict=False)

            print('load pre-trained model {}.'.format(args.pretrained))
            
        else:
            print('no pre-trained model {} found.'.format(args.pretrained))
            
    # freeze fc
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
            
    model.fc.weight.data.normal_(mean=0.0, std=1.0)
    model.fc.bias.data.zero_()
    
    # model.fc.weight.data.uniform_(0.0, 1.0)
    # model.fc.bias.data.fill_(0.001)
    
    criterion = nn.CrossEntropyLoss().cuda()

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.SGD(parameters, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    train_acc1 = []
    train_acc5 = []
    train_loss = []
    
    valid_acc1 = []
    valid_acc5 = []
    valid_loss = []
    best_valid_acc = 0
    print("======== start training ========")
    
    for epoch in range(args.num_epochs):
        utils.adjust_learning_rate(optimizer, epoch, args)
        
        train_loss_value, train_acc1_value, train_acc5_value = train_epoch(epoch, model, train_loader, optimizer, criterion)
        train_loss.append(train_loss_value)
        train_acc1.append(train_acc1_value)
        train_acc5.append(train_acc5_value)
        
        print("Epoch {:d}, train_loss_value: {:.4f}, train_acc1_value: {:.4f}, train_acc5_value: {:.4f}".format(
              epoch, train_loss_value, train_acc1_value, train_acc5_value))
        
        valid_loss_value, valid_acc1_value, valid_acc5_value = valid_epoch(epoch, model, valid_loader, criterion)
        
        valid_loss.append(valid_loss_value)
        valid_acc1.append(valid_acc1_value)
        valid_acc5.append(valid_acc5_value)
        
        print("Epoch {:d}, valid_loss_value: {:.4f}, valid_acc1_value: {:.4f}, valid_acc5_value: {:.4f}".format(
              epoch, valid_loss_value, valid_acc1_value, valid_acc5_value))
        
        print("=======================")
    
    utils.visualization(args.num_epochs, [train_acc1, valid_acc1], 'Accuracy Top 1', args.model_saved_path)
    utils.visualization(args.num_epochs, [train_acc5, valid_acc5], 'Accuracy Top 5', args.model_saved_path)
    utils.visualization(args.num_epochs, [train_loss, valid_loss], 'Loss', args.model_saved_path)
    