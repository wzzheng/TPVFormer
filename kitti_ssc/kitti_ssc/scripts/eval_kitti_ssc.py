from pytorch_lightning import Trainer
from kitti_ssc.models.monoscene import MonoScene
from kitti_ssc.dataset.NYU.nyu_dm import NYUDataModule
from kitti_ssc.dataset.semantic_kitti.kitti_dm import KittiDataModule
import hydra
from omegaconf import DictConfig
import torch
import os
from hydra.utils import get_original_cwd

import tensorboard

from mmcv import Config


@hydra.main(config_name="../config/kitti_ssc.yaml")
def main(config: DictConfig):
    torch.set_grad_enabled(False)

    model_cfg = Config.fromfile(config.model_cfg)

    if config.dataset == "kitti":
        config.batch_size = 1
        n_classes = 20
        project_scale = model_cfg.project_scale
        feature = model_cfg.feature
        full_scene_size = (256, 256, 32)
        data_module = KittiDataModule(
            root=config.kitti_root,
            preprocess_root=config.kitti_preprocess_root,
            frustum_size=config.frustum_size,
            batch_size=int(config.batch_size / config.n_gpus),
            num_workers=int(config.num_workers_per_gpu * config.n_gpus),
        )

    elif config.dataset == "NYU":
        config.batch_size = 2
        project_scale = 1
        n_classes = 12
        feature = 200
        full_scene_size = (60, 36, 60)
        data_module = NYUDataModule(
            root=config.NYU_root,
            preprocess_root=config.NYU_preprocess_root,
            n_relations=config.n_relations,
            frustum_size=config.frustum_size,
            batch_size=int(config.batch_size / config.n_gpus),
            num_workers=int(config.num_workers_per_gpu * config.n_gpus),
        )

    trainer = Trainer(
        sync_batchnorm=True, deterministic=True, gpus=config.n_gpus, accelerator="ddp"
    )

    model_path = config.model_path
    print('importing tpv10')
    from kitti_ssc.tpvformer10.kitti_ssc_tpv import TPV
    model = TPV.load_from_checkpoint(
        model_path,
        feature=feature,
        project_scale=project_scale,
        fp_loss=config.fp_loss,
        full_scene_size=full_scene_size,
    )
    model.eval()
    data_module.setup()
    val_dataloader = data_module.val_dataloader()
    trainer.test(model, test_dataloaders=val_dataloader)


if __name__ == "__main__":
    main()
