import numpy as np
import torch


def worker_init_fn(worker_id):
    """The function is designed for pytorch multi-process dataloader.
    Note that we use the pytorch random generator to generate a base_seed.
    Please try to be consistent.

    References:
        https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed

    """
    base_seed = torch.IntTensor(1).random_().item()
    np.random.seed(base_seed + worker_id)
