import os
import argparse
import pandas as pd
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import aug
import utils
import model

def train_epoch(epoch, moco_model, train_loader, optimizer, criterion):
    global memory_queue
    moco_model.train()

    running_loss, running_acc1, running_acc5, running_count = 0.0, 0.0, 0.0, 0
    for batch_idx, (data, target) in enumerate(train_loader):
        im_q = data[0].cuda(non_blocking=True)
        im_k = data[1].cuda(non_blocking=True)

        query, key = moco_model(im_q, im_k)
        
        l_pos = torch.einsum('nc,nc->n', [query, key]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [query, memory_queue.clone().detach()])

        output = torch.cat([l_pos, l_neg], dim=1)
        output /= args.temperature

        target = torch.zeros(output.shape[0], dtype=torch.long).cuda()

        loss = criterion(output, target)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        memory_queue = torch.cat((memory_queue, key.T), dim=1)[:, key.shape[0]:]

        running_count += im_q.shape[0]
        running_loss += loss.item() * im_q.shape[0]
        running_acc1 += acc1.item() * im_q.shape[0]
        running_acc5 += acc5.item() * im_q.shape[0]
 
        # if batch_idx % 10 == 0:
        #     print("Train Epoch: {}/{} [iter:{}/{}],acc:{}, loss:{}\
        #         ".format(epoch+1, args.num_epochs, batch_idx+1, len(train_loader),
        #             running_acc / running_count,
        #            running_loss / running_count))
            
    
    train_loss_value = running_loss / running_count
    train_acc1_value = running_acc1 / running_count
    train_acc5_value = running_acc5 / running_count
                    
    return train_loss_value, train_acc1_value, train_acc5_value

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-path', default='', type=str,
                        help='path to dataset')
    parser.add_argument('--num-epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--model-saved-path', default='workdirs', type=str,
                        help='Specify the file path where the model is saved (default: workdirs)')
    parser.add_argument('--batch-size', default=256, type=int,
                        help='Specify batch size for training (default: 256).')
    parser.add_argument('--num-workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--lr', default=0.03, type=float,
                        help='Specify initial learning rate (default: 0.03).')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Specify initial lmomentum (default: 0.9).')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='Specify initial weight decay (default: 1e-4).')
    
    # moco model setting
    parser.add_argument('--arch', default='resnet50', type=str,
                        help='Specify model architecture (default: resnet50).')
    parser.add_argument('--feature-dim', default=128, type=float,
                        help='Specify initial feature dim (default: 128).')
    parser.add_argument('--queue-size', default=65536, type=float,
                        help='Specify initial queue size (default: 65536).')
    parser.add_argument('--moco-momentum', default=0.999, type=float,
                        help='Specify moco momentum of updating key encoder (default: 0.999).')
    parser.add_argument('--temperature', default=0.2, type=float,
                        help='Specify temperature for training (default: 0.07).')
    
    parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                        help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument('--cos', action='store_false',
                        help='use cosine lr schedule (default: True)')   
    
    args = parser.parse_args()
        
    try:
        if not os.path.exists(args.model_saved_path):
            os.mkdir(args.model_saved_path)
    except:
        pass
    
    train_data_path = os.path.join(args.data_path, 'train')    
    train_dataset = torchvision.datasets.ImageFolder(train_data_path, transform=aug.TwoCropsTransform(aug.train_transform))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                num_workers=args.num_workers, pin_memory=True, drop_last=True)
        
    moco_model = model.MoCov2(arch=args.arch, feature_dim=args.feature_dim, moco_momentum=args.moco_momentum).cuda()
        
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(moco_model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    memory_queue = nn.functional.normalize(torch.randn(128, 65536), dim=0).cuda()

    train_acc1 = []
	train_acc5 = []
    train_loss = []
    best_train_acc = 0
    print("======== start training ========")
    
    for epoch in range(args.num_epochs):
        utils.adjust_learning_rate(optimizer, epoch, args)
        
        train_loss_value, train_acc1_value, train_acc5_value = train_epoch(epoch, moco_model, train_loader, optimizer, criterion)
        train_loss.append(train_loss_value)
        train_acc1.append(train_acc1_value)
		train_acc5.append(train_acc5_value)
        
        print("Epoch {:d}, train_loss_value: {:.4f}, train_acc1_value: {:.4f}, train_acc5_value: {:.4f}".format(
              epoch, train_loss_value, train_acc1_value, train_acc5_value))
        
        print("=======================")
        
        if epoch % 10 == 0:
            model_path = os.path.join(args.model_saved_path, 'model_{:04d}.pth'.format(epoch))
            torch.save({'epoch': epoch, 'state_dict': moco_model.state_dict(), 'optimizer' : optimizer.state_dict(),}, model_path)
            
        if train_acc1_value > best_train_acc:
            model_path = os.path.join(args.model_saved_path, 'model_best.pth')
            torch.save({'epoch': epoch, 'state_dict': moco_model.state_dict(), 'optimizer' : optimizer.state_dict(),}, model_path)
            best_train_acc = train_acc1_value

    
    utils.visualization(args.num_epochs, [train_acc1], 'Accuracy Top 1', args.model_saved_path)
    utils.visualization(args.num_epochs, [train_acc5], 'Accuracy Top 5', args.model_saved_path)
    utils.visualization(args.num_epochs, [train_loss], 'Loss', args.model_saved_path)
    