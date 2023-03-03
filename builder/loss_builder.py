
import torch
from utils.lovasz_losses import lovasz_softmax


def build(ignore_label=0):

    ce_loss_func = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)

    return ce_loss_func, lovasz_softmax
