import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pickle
from einops import rearrange, reduce, repeat
from torch import einsum


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def update_pseudo_label_temperature(epoch, initial_temperature=1.0, min_temperature=0.1, total_epochs=100) -> float:
    temperature_decrese = (initial_temperature - min_temperature) / total_epochs
    new_temperature = initial_temperature - temperature_decrese * epoch
    return new_temperature

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def len2mask(length, shape):
    batch, max_length = shape
    mask = torch.lt(torch.arange(max_length, device=length.device).unsqueeze(0).expand(shape),
                    length.unsqueeze(1))
    return mask


@torch.no_grad()
def accuracy(logit, oris, refs, phrase_mask, topk=(1,), iouThreshold=0.5):
    b, q, k, d, c = *logit.shape, 4, max(topk)
    # check shape
    rearrange(oris, 'b k d -> b k d', b=b, k=k, d=d)
    rearrange(refs, 'b q d -> b q d', b=b, q=q, d=d)
    rearrange(phrase_mask, 'b q -> b q', b=b, q=q)
    # assure bool type
    phrase_mask = phrase_mask.bool()
    # topk box
    _, topidx = torch.topk(logit, c, dim=-1)
    top_one_hot = F.one_hot(topidx, k).float()
    top_box = einsum('b q c k, b k d -> b q c d', top_one_hot, oris)
    # iou
    n = phrase_mask.sum()
    preds = rearrange(top_box[phrase_mask], 'n c d -> (n c) d', n=n, c=c, d=d)
    labels = repeat(refs[phrase_mask], 'n d -> (n c) d', n=n, c=c, d=d)
    iou_score = iou(preds, labels)
    iou_score = rearrange(iou_score, '(n c) -> n c', n=n, c=c)
    # acc
    matches, _ = torch.cummax(iou_score >= iouThreshold, dim=-1)
    acc = reduce(matches.float(), 'n c -> c', 'mean')
    # ret
    ret = []
    for k in topk:
        ret.append(acc[k - 1].item())
    return ret


def iou(box1, box2):
    """
    :param box1: (n,4)
    :param box2: (n,4)
    :return: (n,)
    """
    n, _ = box1.shape
    assert _ == 4, "wrong box shape"
    (box1_left_x, box1_top_y, box1_right_x, box1_bottom_y) = box1.T
    box1_w = box1_right_x - box1_left_x + 1
    box1_h = box1_bottom_y - box1_top_y + 1

    (box2_left_x, box2_top_y, box2_right_x, box2_bottom_y) = box2.T
    box2_w = box2_right_x - box2_left_x + 1
    box2_h = box2_bottom_y - box2_top_y + 1

    # get intersecting boxes
    intersect_left_x = torch.maximum(box1_left_x, box2_left_x)
    intersect_top_y = torch.maximum(box1_top_y, box2_top_y)
    intersect_right_x = torch.minimum(box1_right_x, box2_right_x)
    intersect_bottom_y = torch.minimum(box1_bottom_y, box2_bottom_y)

    # compute area of intersection
    # the "0" lower bound is to handle cases where box1 and box2 don't overlap
    overlap_x = intersect_right_x - intersect_left_x + 1
    overlap_x = torch.maximum(overlap_x, torch.zeros_like(overlap_x))
    overlap_y = intersect_bottom_y - intersect_top_y + 1
    overlap_y = torch.maximum(overlap_y, torch.zeros_like(overlap_y))
    intersect = overlap_x * overlap_y

    # get area of union
    union = (box1_w * box1_h) + (box2_w * box2_h) - intersect

    # return iou
    return intersect * 1.0 / union
