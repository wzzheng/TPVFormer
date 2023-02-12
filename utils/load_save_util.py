from collections import OrderedDict


def revise_ckpt(state_dict):
    tmp_k = list(state_dict.keys())[0]
    if not tmp_k.startswith('module.'):
        state_dict = OrderedDict(
            {('module.' + k): v
                for k, v in state_dict.items()})
    return state_dict


def revise_ckpt_2(state_dict):
    param_names = list(state_dict.keys())
    for param_name in param_names:
        if 'img_neck.lateral_convs' in param_name or 'img_neck.fpn_convs' in param_name:
            del state_dict[param_name]
    return state_dict