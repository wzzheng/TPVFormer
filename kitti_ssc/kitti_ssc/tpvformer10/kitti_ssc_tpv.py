import pytorch_lightning as pl
import torch
import torch.nn as nn
# from kitti_ssc.models.unet3d_nyu import UNet3D as UNet3DNYU
# from kitti_ssc.models.unet3d_kitti import UNet3D as UNet3DKitti
from kitti_ssc.loss.sscMetrics import SSCMetrics
from kitti_ssc.loss.ssc_loss import sem_scal_loss, CE_ssc_loss, KL_sep, geo_scal_loss
# from kitti_ssc.models.flosp import FLoSP
from kitti_ssc.loss.CRP_loss import compute_super_CP_multilabel_loss
import numpy as np
import torch.nn.functional as F
from kitti_ssc.models.unet2d import UNet2D
from torch.optim.lr_scheduler import MultiStepLR

from kitti_ssc.tpvformer10 import *
from mmdet.models import build_detector
from timm.scheduler import CosineLRScheduler
from .scheduler import MyCosineLRScheduler

class TPV(pl.LightningModule):
    def __init__(
        self,
        model_cfg,
        n_classes,
        class_names,
        feature,
        class_weights,
        project_scale,
        full_scene_size,
        dataset,
        n_relations=4,
        context_prior=True,
        fp_loss=True,
        project_res=[],
        frustum_size=4,
        relation_loss=False,
        CE_ssc_loss=True,
        geo_scal_loss=True,
        sem_scal_loss=True,
        lr=1e-4,
        weight_decay=1e-4,
        decoder_checkpoint=False,
        cos_lr=False,
        **kwargs,
    ):
        super().__init__()

        self.project_res = project_res
        self.fp_loss = fp_loss
        self.dataset = dataset
        # self.context_prior = context_prior
        self.context_prior = False
        self.frustum_size = frustum_size
        self.class_names = class_names
        # self.relation_loss = relation_loss
        self.relation_loss = False
        self.CE_ssc_loss = CE_ssc_loss
        self.sem_scal_loss = sem_scal_loss
        self.geo_scal_loss = geo_scal_loss
        self.project_scale = project_scale
        self.class_weights = class_weights
        self.lr = lr
        self.weight_decay = weight_decay

        self.projects = {}
        self.scale_2ds = [1, 2, 4, 8]  # 2D scales
        # for scale_2d in self.scale_2ds:
        #     self.projects[str(scale_2d)] = FLoSP(
        #         full_scene_size, project_scale=self.project_scale, dataset=self.dataset
        #     )
        # self.projects = nn.ModuleDict(self.projects)

        self.cos_lr = cos_lr

        self.n_classes = n_classes
        if self.dataset == "NYU":
            self.net_3d_decoder = build_detector(model_cfg)
            self.net_3d_decoder.init_weights()
        elif self.dataset == "kitti":
            self.net_3d_decoder = build_detector(model_cfg)
            self.net_3d_decoder.init_weights()
        
        self.net_rgb = UNet2D.build(
            out_feature=feature, 
            use_decoder=True,
            decoder_checkpoint=decoder_checkpoint
        )

        # log hyperparameters
        self.save_hyperparameters()

        self.train_metrics = SSCMetrics(self.n_classes)
        self.val_metrics = SSCMetrics(self.n_classes)
        self.test_metrics = SSCMetrics(self.n_classes)

    def forward_dummy(self, batch):
        img = batch["img"]
        img_metas = batch["img_metas"]
        # print(img_metas)
        res = {}

        # img = self.img_backbone(img)
        img = batch['x_rgb']

        out = self.net_3d_decoder(img=img, img_metas=img_metas)
        # x3d_up_lfull = self.up_l1_lfull(out)

        # ssc_logit_full = self.ssc_head(x3d_up_lfull)

        # res["ssc_logit"] = ssc_logit_full
        res["ssc_logit"] = out

        return res


    def forward(self, batch):

        img = batch["img"]

        out = {}

        x_rgb = self.net_rgb(img)

        # x3ds = []
        # for i in range(bs):
        #     x3d = None
        #     for scale_2d in self.project_res:

        #         # project features at each 2D scale to target 3D scale
        #         scale_2d = int(scale_2d)
        #         projected_pix = batch["projected_pix_{}".format(self.project_scale)][i].cuda()
        #         fov_mask = batch["fov_mask_{}".format(self.project_scale)][i].cuda()

        #         # Sum all the 3D features
        #         if x3d is None:
        #             x3d = self.projects[str(scale_2d)](
        #                 x_rgb["1_" + str(scale_2d)][i],
        #                 projected_pix // scale_2d,
        #                 fov_mask,
        #             )
        #         else:
        #             x3d += self.projects[str(scale_2d)](
        #                 x_rgb["1_" + str(scale_2d)][i],
        #                 projected_pix // scale_2d,
        #                 fov_mask,
        #             )
        #     x3ds.append(x3d)

        input_dict = {
            # "img": [
            #     x_rgb["1_1"].unsqueeze(1), # B, N, C, H, W
            #     x_rgb["1_2"].unsqueeze(1),
            #     x_rgb["1_4"].unsqueeze(1),
            #     x_rgb["1_8"].unsqueeze(1)
            # ],
            "img": [
                x_rgb[0].unsqueeze(1), # B, N, C, H, W
                x_rgb[1].unsqueeze(1),
                x_rgb[2].unsqueeze(1),
                x_rgb[3].unsqueeze(1)
            ],
            "img_metas": batch["img_metas"]
        }

        out = self.net_3d_decoder(**input_dict)

        return {'ssc_logit': out}

    def step(self, batch, step_type, metric):
        bs = len(batch["img"])
        loss = 0
        out_dict = self(batch)
        ssc_pred = out_dict["ssc_logit"]
        target = batch["target"]

        if self.context_prior:
            P_logits = out_dict["P_logits"]
            CP_mega_matrices = batch["CP_mega_matrices"]

            if self.relation_loss:
                loss_rel_ce = compute_super_CP_multilabel_loss(
                    P_logits, CP_mega_matrices
                )
                loss += loss_rel_ce
                self.log(
                    step_type + "/loss_relation_ce_super",
                    loss_rel_ce.detach(),
                    on_epoch=True,
                    sync_dist=True,
                )

        class_weight = self.class_weights.type_as(batch["img"])
        if self.CE_ssc_loss:
            loss_ssc = CE_ssc_loss(ssc_pred, target, class_weight)
            loss += loss_ssc
            self.log(
                step_type + "/loss_ssc",
                loss_ssc.detach(),
                on_epoch=True,
                sync_dist=True,
            )

        if self.sem_scal_loss:
            loss_sem_scal = sem_scal_loss(ssc_pred, target)
            loss += loss_sem_scal
            self.log(
                step_type + "/loss_sem_scal",
                loss_sem_scal.detach(),
                on_epoch=True,
                sync_dist=True,
            )

        if self.geo_scal_loss:
            loss_geo_scal = geo_scal_loss(ssc_pred, target)
            loss += loss_geo_scal
            self.log(
                step_type + "/loss_geo_scal",
                loss_geo_scal.detach(),
                on_epoch=True,
                sync_dist=True,
            )

        if self.fp_loss and step_type != "test":
            frustums_masks = torch.stack(batch["frustums_masks"])
            frustums_class_dists = torch.stack(
                batch["frustums_class_dists"]
            ).float()  # (bs, n_frustums, n_classes)
            n_frustums = frustums_class_dists.shape[1]

            pred_prob = F.softmax(ssc_pred, dim=1)
            batch_cnt = frustums_class_dists.sum(0)  # (n_frustums, n_classes)

            frustum_loss = 0
            frustum_nonempty = 0
            for frus in range(n_frustums):
                frustum_mask = frustums_masks[:, frus, :, :, :].unsqueeze(1).float()
                prob = frustum_mask * pred_prob  # bs, n_classes, H, W, D
                prob = prob.reshape(bs, self.n_classes, -1).permute(1, 0, 2)
                prob = prob.reshape(self.n_classes, -1)
                cum_prob = prob.sum(dim=1)  # n_classes

                total_cnt = torch.sum(batch_cnt[frus])
                total_prob = prob.sum()
                if total_prob > 0 and total_cnt > 0:
                    frustum_target_proportion = batch_cnt[frus] / total_cnt
                    cum_prob = cum_prob / total_prob  # n_classes
                    frustum_loss_i = KL_sep(cum_prob, frustum_target_proportion)
                    frustum_loss += frustum_loss_i
                    frustum_nonempty += 1
            frustum_loss = frustum_loss / frustum_nonempty
            loss += frustum_loss
            self.log(
                step_type + "/loss_frustums",
                frustum_loss.detach(),
                on_epoch=True,
                sync_dist=True,
            )

        y_true = target.cpu().numpy()
        y_pred = ssc_pred.detach().cpu().numpy()
        y_pred = np.argmax(y_pred, axis=1)
        metric.add_batch(y_pred, y_true)

        self.log(step_type + "/loss", loss.detach(), on_epoch=True, sync_dist=True)

        # if step_type == 'train':
        #     print('check unused parameters')
        #     loss.backward()
        #     for name, param in self.named_parameters():
        #         if param.requires_grad and param.grad is None:
        #             print(name)
        #     import pdb; pdb.set_trace()

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train", self.train_metrics)

    def validation_step(self, batch, batch_idx):
        self.step(batch, "val", self.val_metrics)

    def validation_epoch_end(self, outputs):
        metric_list = [("train", self.train_metrics), ("val", self.val_metrics)]

        for prefix, metric in metric_list:
            stats = metric.get_stats()
            for i, class_name in enumerate(self.class_names):
                self.log(
                    "{}_SemIoU/{}".format(prefix, class_name),
                    stats["iou_ssc"][i],
                    sync_dist=True,
                )
            self.log("{}/mIoU".format(prefix), stats["iou_ssc_mean"], sync_dist=True)
            self.log("{}/IoU".format(prefix), stats["iou"], sync_dist=True)
            self.log("{}/Precision".format(prefix), stats["precision"], sync_dist=True)
            self.log("{}/Recall".format(prefix), stats["recall"], sync_dist=True)
            metric.reset()

    def test_step(self, batch, batch_idx):
        self.step(batch, "test", self.test_metrics)

    def test_epoch_end(self, outputs):
        classes = self.class_names
        metric_list = [("test", self.test_metrics)]
        for prefix, metric in metric_list:
            print("{}======".format(prefix))
            stats = metric.get_stats()
            print(
                "Precision={:.4f}, Recall={:.4f}, IoU={:.4f}".format(
                    stats["precision"] * 100, stats["recall"] * 100, stats["iou"] * 100
                )
            )
            print("class IoU: {}, ".format(classes))
            print(
                " ".join(["{:.4f}, "] * len(classes)).format(
                    *(stats["iou_ssc"] * 100).tolist()
                )
            )
            print("mIoU={:.4f}".format(stats["iou_ssc_mean"] * 100))
            metric.reset()

    def configure_optimizers(self):
        if self.dataset == "NYU":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            scheduler = MultiStepLR(optimizer, milestones=[20], gamma=0.1)
            return [optimizer], [scheduler]
        elif self.dataset == "kitti":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            if not self.cos_lr:
                scheduler = MultiStepLR(optimizer, milestones=[25], gamma=0.1)
                sched_config = {
                    'scheduler': scheduler
                }
            else:
                scheduler = MyCosineLRScheduler(
                    optimizer,
                    30 * 1163,
                    lr_min=1e-6,
                    warmup_t=500,
                    warmup_lr_init=1e-5,
                    t_in_epochs=False
                )
                sched_config = {
                    'scheduler': scheduler,
                    'interval': 'step',
                }
            return {
                'optimizer': optimizer,
                'lr_scheduler': sched_config
            }
