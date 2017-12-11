import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Source: Focal Loss for dense object detection
# https://arxiv.org/abs/1708.02002
# Adapted from https://github.com/kuangliu/pytorch-retinanet/blob/master/loss.py


def one_hot_embedding(labels, num_classes):
    '''Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''
    y = torch.eye(num_classes)  # [D,D]
    return y[labels]            # [N,D]


class FocalLoss(nn.Module):
    def __init__(self, num_classes, cuda):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.cuda = cuda

    def forward(self, x, y):
        '''Focal loss.
        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].
        Return:
          (tensor) focal loss.
        '''
        alpha = 0.25
        gamma = 2

        t = one_hot_embedding(y.data.cpu(), self.num_classes)
        if self.cuda:
            t = Variable(t).cuda()

        p = x.sigmoid()
        pt = p * t + (1 - p) * (1 - t)         # pt = p if t > 0 else 1-p
        # w = alpha if t > 0 else 1-alpha
        w = alpha * t + (1 - alpha) * (1 - t)
        w = w * (1 - pt).pow(gamma)
        return F.binary_cross_entropy_with_logits(x, t, w, size_average=False)


class FocalLossAlt(nn.Module):
    def __init__(self, num_classes, cuda):
        super(FocalLossAlt, self).__init__()
        self.num_classes = num_classes
        self.cuda = cuda

    def forward(self, x, y):
        '''Focal loss alternative.
        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].
        Return:
          (tensor) focal loss.
        '''
        alpha = 0.25

        t = one_hot_embedding(y.data.cpu(), self.num_classes)
        if self.cuda:
            t = Variable(t).cuda()

        xt = x * (2 * t - 1)  # xt = x if t > 0 else -x
        pt = (2 * xt + 1).sigmoid()

        w = alpha * t + (1 - alpha) * (1 - t)
        loss = -w * pt.log() / 2
        return loss.sum()
