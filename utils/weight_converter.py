import torch
from copy import deepcopy

ckpt_path = 'ckpts/tpv04_occupancy.pth'
save_path = 'ckpts/tpv04_occupancy_v2.pth'

ckpt_path = 'ckpts/tpv10_lidarseg.pth'
save_path = 'ckpts/tpv10_lidarseg_v2.pth'

lut = {
    'pts_bbox_head': ['tpv_head', 1],
    'transformer.': ['', 1],
    'fusion_head': ['tpv_aggregator', 1],
    'bev': ['tpv', None]
}

ckpt = torch.load(ckpt_path, map_location='cpu')

keys = list(ckpt.keys())
for k in keys:
    new_k = deepcopy(k)
    for old, new in lut.items():
        if new[1] is not None:
            new_k = new_k.replace(old, new[0], new[1])
        else:
            new_k = new_k.replace(old, new[0])
    if new_k == k:
        continue
    ckpt[new_k] = ckpt[k]
    del ckpt[k]

torch.save(ckpt, save_path)
