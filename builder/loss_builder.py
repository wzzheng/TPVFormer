
import torch
from utils.lovasz_losses import lovasz_softmax


def build(wce=True, lovasz=True, ignore_label=0):

    loss_funs = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)

    if wce and lovasz:
        return loss_funs, lovasz_softmax
    elif wce and not lovasz:
        return wce
    elif not wce and lovasz:
        return lovasz_softmax
    else:
        raise NotImplementedError
