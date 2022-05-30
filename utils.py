import os
import torch
import math
import matplotlib.pyplot as plt

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.num_epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def visualization(num_epochs, value_list, title, model_saved_path):
    plt.plot(range(num_epochs), value_list[0], 'b-', label=f'Training_{title}')
    title_name = 'Training ' + title 
    
    if len(value_list) == 2:
        plt.plot(range(num_epochs), value_list[1], 'g-', label=f'validation{title}')
        title_name = 'Training & Validation ' + title 
        
    plt.title(title_name)
    plt.xlabel('Number of epochs')
    plt.ylabel(title)
    plt.legend()
    plt.savefig(os.path.join(model_saved_path, f"{title}.jpg"))
    plt.close()
    