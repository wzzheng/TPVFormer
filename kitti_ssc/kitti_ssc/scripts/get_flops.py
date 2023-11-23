from mmcv.cnn.utils.flops_counter import get_model_complexity_info
from kitti_ssc.models.monoscene import MonoScene
import hydra
from omegaconf import DictConfig
hydra.output_subdir = None
from kitti_ssc.dataset.semantic_kitti.kitti_dm import KittiDataModule
from kitti_ssc.dataset.semantic_kitti.kitti_dataset import KittiDataset
from kitti_ssc.dataset.semantic_kitti.collate import collate_fn
from kitti_ssc.dataset.semantic_kitti.params import (
    semantic_kitti_class_frequencies,
    kitti_class_names,
)
import torch, numpy as np, torch.nn as nn
from functools import partial
from mmcv import Config

@hydra.main(config_name="../config/kitti_ssc.yaml")
def build_model_dm(config: DictConfig):

    if hasattr(config, 'model_cfg'):
    # if config.model_cfg is not None:
        model_cfg = Config.fromfile(config.model_cfg)
    else:
        model_cfg = None
    class_names = kitti_class_names
    max_epochs = 30
    logdir = config.kitti_logdir
    full_scene_size = (256, 256, 32)
    project_scale = model_cfg.project_scale if model_cfg is not None else 2
    feature = model_cfg.feature if model_cfg is not None else 64
    n_classes = 20
    class_weights = torch.from_numpy(
        1 / np.log(semantic_kitti_class_frequencies + 0.001)
    )

    dataset = KittiDataset(
        split="train",
        root=config.kitti_root,
        preprocess_root=config.kitti_preprocess_root,
        project_scale=project_scale,
        frustum_size=config.frustum_size,
        fliplr=0.5,
        color_jitter=(0.4, 0.4, 0.4),
    )

    project_res = ["1"]
    if config.project_1_2:
        project_res.append("2")
    if config.project_1_4:
        project_res.append("4")
    if config.project_1_8:
        project_res.append("8")

    if model_cfg is not None and 'tpv' in config.model_cfg:
        print('importing tpv10')
        from kitti_ssc.tpvformer10.kitti_ssc_tpv import TPV

        model = TPV(
            model_cfg=model_cfg.model,
            dataset=config.dataset,
            frustum_size=config.frustum_size,
            project_scale=project_scale,
            n_relations=config.n_relations,
            fp_loss=config.fp_loss,
            feature=feature,
            full_scene_size=full_scene_size,
            project_res=project_res,
            n_classes=n_classes,
            class_names=class_names,
            context_prior=config.context_prior,
            relation_loss=config.relation_loss,
            CE_ssc_loss=config.CE_ssc_loss,
            sem_scal_loss=config.sem_scal_loss,
            geo_scal_loss=config.geo_scal_loss,
            lr=model_cfg.lr,
            weight_decay=model_cfg.weight_decay,
            class_weights=class_weights,
            decoder_checkpoint=model_cfg.get('decoder_checkpoint', False),
            cos_lr=model_cfg.get('cos_lr', False)
        )
        print(model.net_3d_decoder)
    else:
        from kitti_ssc.models.monoscene import MonoScene
        model = MonoScene(
            dataset=config.dataset,
            frustum_size=config.frustum_size,
            project_scale=project_scale,
            n_relations=config.n_relations,
            fp_loss=config.fp_loss,
            feature=feature,
            full_scene_size=full_scene_size,
            project_res=project_res,
            n_classes=n_classes,
            class_names=class_names,
            context_prior=config.context_prior,
            relation_loss=config.relation_loss,
            CE_ssc_loss=config.CE_ssc_loss,
            sem_scal_loss=config.sem_scal_loss,
            geo_scal_loss=config.geo_scal_loss,
            lr=config.lr,
            weight_decay=config.weight_decay,
            class_weights=class_weights,
        )


    # Initialize MonoScene model
    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    # import pdb; pdb.set_trace()
    # return model, dataset
    flops_count, params_count = get_model_complexity_info(
        model.cuda(), (1200, 370), input_constructor=partial(input_constructor, dataset, feature)
    )
    print(flops_count)
    print(params_count)

def input_constructor(dataset, feature, *args, **kwargs):
    input = [dataset[0]]
    input = collate_fn(input)
    h, w = input['img'].shape[-2:]
    input['img'] = input['img'].cuda()
    img = torch.randn(1, feature, h, w)
    # input['x_rgb'] = {
    #     "1_1": img.cuda(),
    #     "1_2": nn.MaxPool2d(3, 2, 1)(img).cuda(),
    #     "1_4": nn.MaxPool2d(5, 4, 2)(img).cuda(),
    #     "1_8": nn.MaxPool2d(9, 8, 4)(img).cuda(),
    #     "1_16": nn.MaxPool2d(17, 16, 8)(img).cuda(),
    # }
    input['x_rgb'] = [
        img.unsqueeze(1).cuda(),
        nn.MaxPool2d(3, 2, 1)(img).unsqueeze(1).cuda(),
        nn.MaxPool2d(5, 4, 2)(img).unsqueeze(1).cuda(),
        nn.MaxPool2d(9, 8, 4)(img).unsqueeze(1).cuda(),
        # img.cuda(),
        # nn.MaxPool2d(3, 2, 1)(img).cuda(),
        # nn.MaxPool2d(5, 4, 2)(img).cuda(),
        # nn.MaxPool2d(9, 8, 4)(img).cuda(),
    ]

    # import pdb; pdb.set_trace()
    return {'batch': input}

if __name__ == '__main__':
    build_model_dm()
    # import pdb; pdb.set_trace()
    # model, dataset = None
    # flops_count, params_count = get_model_complexity_info(
    #     model, (1200, 370), input_constructor=partial(input_constructor, dataset)
    # )