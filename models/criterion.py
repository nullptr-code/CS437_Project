from torch import nn
import torch


def diceLoss(pred, target, smooth=1):
    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    union = (pred**2).sum() + (target**2).sum()
    loss = 1 - ((2.0 * intersection + smooth) / (union + smooth))
    return loss


def weighted_loss(pred, target, bce_weight=0.5):
    bce = nn.functional.binary_cross_entropy_with_logits(pred, target)

    pred = torch.sigmoid(pred)
    dice = diceLoss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    return loss
