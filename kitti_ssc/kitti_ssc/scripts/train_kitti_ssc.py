from kitti_ssc.dataset.semantic_kitti.kitti_dm import KittiDataModule
from kitti_ssc.dataset.semantic_kitti.params import (
    semantic_kitti_class_frequencies,
    kitti_class_names,
)
from kitti_ssc.dataset.NYU.params import (
    class_weights as NYU_class_weights,
    NYU_class_names,
)
from kitti_ssc.dataset.NYU.nyu_dm import NYUDataModule
from torch.utils.data.dataloader import DataLoader
# from kitti_ssc.models.monoscene import MonoScene
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
# from pytorch_lightning.strategies import DDPSpawnStrategy
import os
import hydra
from omegaconf import DictConfig
import numpy as np
import torch
from mmcv import Config
import mmcv
logger = mmcv.utils.get_logger('mmcv')
logger.setLevel("WARNING")

hydra.output_subdir = None


@hydra.main(config_name="../config/kitti_ssc.yaml")
def main(config: DictConfig):
    exp_name = config.exp_prefix
    exp_name += "_{}_{}".format(config.dataset, config.run)
    exp_name += "_FrusSize_{}".format(config.frustum_size)
    exp_name += "_nRelations{}".format(config.n_relations)
    exp_name += "_WD{}_lr{}".format(config.weight_decay, config.lr)

    if config.CE_ssc_loss:
        exp_name += "_CEssc"
    if config.geo_scal_loss:
        exp_name += "_geoScalLoss"
    if config.sem_scal_loss:
        exp_name += "_semScalLoss"
    if config.fp_loss:
        exp_name += "_fpLoss"

    if config.relation_loss:
        exp_name += "_CERel"
    if config.context_prior:
        exp_name += "_3DCRP"

    # Setup dataloaders
    if config.dataset == "kitti":
        if config.model_cfg is not None:
            model_cfg = Config.fromfile(config.model_cfg)
        class_names = kitti_class_names
        max_epochs = 30
        logdir = config.kitti_logdir
        full_scene_size = (256, 256, 32)
        project_scale = model_cfg.project_scale
        feature = model_cfg.feature
        n_classes = 20
        class_weights = torch.from_numpy(
            1 / np.log(semantic_kitti_class_frequencies + 0.001)
        )
        data_module = KittiDataModule(
            root=config.kitti_root,
            preprocess_root=config.kitti_preprocess_root,
            frustum_size=config.frustum_size,
            project_scale=project_scale,
            batch_size=int(config.batch_size / config.n_gpus),
            num_workers=int(config.num_workers_per_gpu),
        )

    elif config.dataset == "NYU":
        class_names = NYU_class_names
        max_epochs = 30
        logdir = config.logdir
        full_scene_size = (60, 36, 60)
        project_scale = 1
        feature = 200
        n_classes = 12
        class_weights = NYU_class_weights
        data_module = NYUDataModule(
            root=config.NYU_root,
            preprocess_root=config.NYU_preprocess_root,
            n_relations=config.n_relations,
            frustum_size=config.frustum_size,
            batch_size=int(config.batch_size / config.n_gpus),
            num_workers=int(config.num_workers_per_gpu * config.n_gpus),
        )

    project_res = ["1"]
    if config.project_1_2:
        exp_name += "_Proj_2"
        project_res.append("2")
    if config.project_1_4:
        exp_name += "_4"
        project_res.append("4")
    if config.project_1_8:
        exp_name += "_8"
        project_res.append("8")

    print(exp_name)

    # Initialize MonoScene model
    if 'tpv' in config.model_cfg:
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
    # else:
    #     from kitti_ssc.models.monoscene import MonoScene


    if config.enable_log:
        logger = TensorBoardLogger(save_dir=logdir, name=exp_name, version="")
        lr_monitor = LearningRateMonitor(logging_interval="step")
        checkpoint_callbacks = [
            ModelCheckpoint(
                save_last=True,
                monitor="val/mIoU",
                save_top_k=1,
                mode="max",
                filename="{epoch:03d}-{val/mIoU:.5f}",
            ),
            lr_monitor,
        ]
    else:
        logger = False
        checkpoint_callbacks = False

    model_path = os.path.join(logdir, exp_name, "checkpoints/last.ckpt")
    if os.path.isfile(model_path):
        # Continue training from last.ckpt
        trainer = Trainer(
            callbacks=checkpoint_callbacks,
            resume_from_checkpoint=model_path,
            sync_batchnorm=True,
            deterministic=False,
            max_epochs=max_epochs,
            gpus=config.n_gpus,
            logger=logger,
            check_val_every_n_epoch=1,
            log_every_n_steps=10,
            flush_logs_every_n_steps=100,
            accelerator="ddp",
        )
    else:
        # Train from scratch
        trainer = Trainer(
            callbacks=checkpoint_callbacks,
            sync_batchnorm=True,
            deterministic=False,
            max_epochs=max_epochs,
            gpus=config.n_gpus,
            logger=logger,
            check_val_every_n_epoch=1,
            log_every_n_steps=10,
            flush_logs_every_n_steps=100,
            # accelerator="ddp",
            accelerator="ddp",
            # strategy=DDPSpawnStrategy(find_unused_parameters=False)
        )

    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
