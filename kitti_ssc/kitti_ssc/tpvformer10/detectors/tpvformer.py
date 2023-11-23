# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

from tkinter.messagebox import NO
import torch, torch.nn as nn, torch.nn.functional as F
from mmcv.runner import force_fp32, auto_fp16, BaseModule
from mmdet.models import DETECTORS, HEADS, builder
from .grid_mask import GridMask
import time
import copy
import numpy as np
from torch.cuda.amp import autocast

@HEADS.register_module()
class TPVFuser(BaseModule):
    def __init__(
        self, bev_h, bev_w, bev_z, nbr_classes=20, 
        in_dims=64, hidden_dims=128, out_dims=None,
        scale_h=2, scale_w=2, scale_z=2, use_checkpoint=False,
    ):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_z = bev_z
        self.scale_h = scale_h
        self.scale_w = scale_w
        self.scale_z = scale_z
        self.use_checkpoint = use_checkpoint

        out_dims = in_dims if out_dims is None else out_dims

        self.decoder = nn.Sequential(
            nn.Linear(in_dims, hidden_dims),
            nn.Softplus(),
            nn.Linear(hidden_dims, out_dims)
        )

        self.classifier = nn.Linear(out_dims, nbr_classes)
        self.classes = nbr_classes
    
    def forward(self, bev_list, points=None):
        """
        bev_list[0]: bs, h*w, c
        bev_list[1]: bs, z*h, c
        bev_list[2]: bs, w*z, c
        """
        bev_hw, bev_zh, bev_zw = bev_list[0], bev_list[1], bev_list[2]
        bev_hw = bev_hw + bev_hw.mean(dim=1, keepdim=True) * 0.0
        bev_zh = bev_zh + bev_zh.mean(dim=1, keepdim=True) * 0.0
        bev_zw = bev_zw + bev_zw.mean(dim=1, keepdim=True) * 0.0
        bs, _, c = bev_hw.shape
        bev_hw = bev_hw.permute(0, 2, 1).reshape(bs, c, self.bev_h, self.bev_w)
        bev_zh = bev_zh.permute(0, 2, 1).reshape(bs, c, self.bev_z, self.bev_h)
        bev_zw = bev_zw.permute(0, 2, 1).reshape(bs, c, self.bev_w, self.bev_z)

        # NVIDIA STYLE
        if self.scale_h != 1 or self.scale_w != 1:
            bev_hw = F.interpolate(
                bev_hw, 
                size=(self.bev_h*self.scale_h, self.bev_w*self.scale_w),
                # scale_factor=self.scale_h, 
                mode='bilinear'
            )
        if self.scale_z != 1 or self.scale_h != 1:
            bev_zh = F.interpolate(
                bev_zh, 
                size=(self.bev_z*self.scale_z, self.bev_h*self.scale_h),
                # scale_factor=self.scale_w, 
                mode='bilinear'
            )
        if self.scale_w != 1 or self.scale_z != 1:
            bev_zw = F.interpolate(
                bev_zw, 
                size=(self.bev_w*self.scale_w, self.bev_z*self.scale_z),
                # scale_factor=self.scale_z, 
                mode='bilinear'
            )
        
        if points is not None:
            # points: bs, n, 3
            _, n, _ = points.shape
            points = points.reshape(bs, 1, n, 3)
            points[..., 0] = points[..., 0] / (self.bev_w*self.scale_w) * 2 - 1
            points[..., 1] = points[..., 1] / (self.bev_h*self.scale_h) * 2 - 1
            points[..., 2] = points[..., 2] / (self.bev_z*self.scale_z) * 2 - 1
            sample_loc = points[:, :, :, [0, 1]]
            bev_hw_pts = F.grid_sample(bev_hw, sample_loc).squeeze(2) # bs, c, n
            sample_loc = points[:, :, :, [1, 2]]
            bev_zh_pts = F.grid_sample(bev_zh, sample_loc).squeeze(2)
            sample_loc = points[:, :, :, [2, 0]]
            bev_zw_pts = F.grid_sample(bev_zw, sample_loc).squeeze(2)

            bev_hw_vox = bev_hw.unsqueeze(-1).permute(0, 1, 3, 2, 4).expand(-1, -1, -1, -1, self.scale_z*self.bev_z)
            bev_zh_vox = bev_zh.unsqueeze(-1).permute(0, 1, 4, 3, 2).expand(-1, -1, self.scale_w*self.bev_w, -1, -1)
            bev_zw_vox = bev_zw.unsqueeze(-1).permute(0, 1, 2, 4, 3).expand(-1, -1, -1, self.scale_h*self.bev_h, -1)
        
            fused_vox = (bev_hw_vox + bev_zh_vox + bev_zw_vox).flatten(2)
            fused_pts = bev_hw_pts + bev_zh_pts + bev_zw_pts
            fused = torch.cat([fused_vox, fused_pts], dim=-1) # bs, c, whz+n
            
            fused = fused.permute(0, 2, 1)
            if self.use_checkpoint:
                fused = torch.utils.checkpoint.checkpoint(self.decoder, fused)
                logits = torch.utils.checkpoint.checkpoint(self.classifier, fused)
            else:
                fused = self.decoder(fused)
                logits = self.classifier(fused)
            logits = logits.permute(0, 2, 1)
            logits_vox = logits[:, :, :(-n)].reshape(bs, self.classes, self.scale_w*self.bev_w, self.scale_h*self.bev_h, self.scale_z*self.bev_z)
            logits_pts = logits[:, :, (-n):].reshape(bs, self.classes, n, 1, 1)
            return logits_vox, logits_pts
            
        else:
            # bev_hw = bev_hw.unsqueeze(-1).permute(0, 1, 3, 2, 4).expand(-1, -1, -1, -1, self.scale_z*self.bev_z)
            # bev_zh = bev_zh.unsqueeze(-1).permute(0, 1, 4, 3, 2).expand(-1, -1, self.scale_w*self.bev_w, -1, -1)
            # bev_zw = bev_zw.unsqueeze(-1).permute(0, 1, 2, 4, 3).expand(-1, -1, -1, self.scale_h*self.bev_h, -1)
            bev_hw = bev_hw.unsqueeze(-1).expand(-1, -1, -1, -1, self.scale_z*self.bev_z)
            bev_zh = bev_zh.unsqueeze(-1).permute(0, 1, 3, 4, 2).expand(-1, -1, -1, self.scale_w*self.bev_w, -1)
            bev_zw = bev_zw.unsqueeze(-1).permute(0, 1, 4, 2, 3).expand(-1, -1, self.scale_h*self.bev_h, -1, -1)
        
            fused = bev_hw + bev_zh + bev_zw
            fused = fused.permute(0, 2, 3, 4, 1)
            if self.use_checkpoint:
                fused = torch.utils.checkpoint.checkpoint(self.decoder, fused)
                logits = torch.utils.checkpoint.checkpoint(self.classifier, fused)
            else:
                fused = self.decoder(fused)
                logits = self.classifier(fused)
            # fused = self.decoder(fused)
            # logits = self.classifier(fused)
            logits = logits.permute(0, 4, 1, 2, 3)
        
            return logits


@DETECTORS.register_module()
class TPVFormer(BaseModule):
    """BEVFormer.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False,
                 fusion_head=None
                 ):

        super().__init__()
        
        if pts_bbox_head:
            pts_train_cfg = train_cfg.pts if train_cfg else None
            pts_bbox_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg else None
            pts_bbox_head.update(test_cfg=pts_test_cfg)
            self.pts_bbox_head = builder.build_head(pts_bbox_head)

        if img_backbone is not None:
            self.img_backbone = builder.build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = builder.build_neck(img_neck)

        if fusion_head:
            self.fusion_head = builder.build_head(fusion_head)

        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            
            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)

        return img_feats

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          prev_bev=None):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """

        outs = self.pts_bbox_head(pts_feats, img_metas, prev_bev)
        return outs

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()
        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev, only_bev=True)[0]
            self.train()
            return prev_bev

    @auto_fp16(apply_to=('img', 'prev_bev', 'points'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        # import pdb; pdb.set_trace()
        if hasattr(self, 'img_backbone'):
            img_feats = self.extract_feat(img=img, img_metas=img_metas)
        else:
            img_feats = img
        outs = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                      gt_labels_3d, img_metas,
                                      gt_bboxes_ignore)
        outs = self.fusion_head(outs, points)
        return outs

    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0][0]['can_bus'][-1] = 0
            img_metas[0][0]['can_bus'][:3] = 0

        new_prev_bev, bbox_results = self.simple_test(
            img_metas[0], img[0], prev_bev=self.prev_frame_info['prev_bev'], **kwargs)
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev
        return bbox_results


    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts = self.simple_test_pts(
            img_feats, img_metas, prev_bev, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return new_prev_bev, bbox_list
