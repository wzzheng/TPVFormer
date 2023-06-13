# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.runner import BaseModule

from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmdet.models.utils import build_transformer
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet.models.dense_heads import DETRHead
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner import force_fp32, auto_fp16
import numpy as np
import mmcv
import cv2 as cv


@HEADS.register_module()
class TPVFormerHead(BaseModule):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 bev_h=30,
                 bev_w=30,
                 bev_z=30,
                 use_one_pe=True,
                 positional_encoding_zh=None,
                 positional_encoding_wz=None,
                 pc_range,
                 **kwargs):
        super().__init__()

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_z = bev_z
        self.fp16_enabled = False

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.pc_range = pc_range
        self.real_h = self.pc_range[3] - self.pc_range[0]
        self.real_w = self.pc_range[4] - self.pc_range[1]
        self.real_z = self.pc_range[5] - self.pc_range[2]
        # bev_mask_hw = torch.zeros(1, bev_h, bev_w)
        # self.register_buffer('bev_mask_hw', bev_mask_hw)
        # if positional_encoding_zh is not None:
        #     self.positional_encoding_zh = build_positional_encoding(positional_encoding_zh)
        #     self.positional_encoding_wz = build_positional_encoding(positional_encoding_wz)
        #     bev_mask_zh = torch.zeros(1, bev_z, bev_h)
        #     self.register_buffer('bev_mask_zh', bev_mask_zh)
        #     bev_mask_wz = torch.zeros(1, bev_w, bev_z)
        #     self.register_buffer('bev_mask_wz', bev_mask_wz)

        self.in_channels = kwargs['in_channels']

        self.act_cfg = transformer.get('act_cfg', dict(type='ReLU', inplace=True))
        if use_one_pe:
            positional_encoding = kwargs['positional_encoding']
            self.positional_encoding = build_positional_encoding(positional_encoding)
        self.use_one_pe = use_one_pe
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims
        # assert 'num_feats' in positional_encoding
        # num_feats = positional_encoding['num_feats']
        # assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
        #     f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
        #     f' and {num_feats}.'
        self._init_layers()


    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""

        if not self.as_two_stage:
            self.bev_embedding_hw = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)
            self.bev_embedding_zh = nn.Embedding(self.bev_z * self.bev_h, self.embed_dims)
            self.bev_embedding_wz = nn.Embedding(self.bev_w * self.bev_z, self.embed_dims)


    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()


    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, img_metas, prev_bev=None,  only_bev=True):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder. 
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        # import pdb; pdb.set_trace()
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        device = mlvl_feats[0].device
        # object_query_embeds = self.query_embedding.weight.to(dtype)
        bev_queries_hw = self.bev_embedding_hw.weight.to(dtype)
        bev_queries_zh = self.bev_embedding_zh.weight.to(dtype)
        bev_queries_wz = self.bev_embedding_wz.weight.to(dtype)

        # bev_mask_hw = self.bev_mask_hw.expand(bs, -1, -1)
        # bev_pos_hw = self.positional_encoding(bev_mask_hw).to(dtype)
        # if hasattr(self, 'positional_encoding_zh'):
        #     bev_mask_zh = self.bev_mask_zh.expand(bs, -1, -1)
        #     bev_mask_wz = self.bev_mask_wz.expand(bs, -1, -1)
        #     bev_pos_zh = self.positional_encoding_zh(bev_mask_zh).to(dtype)
        #     bev_pos_wz = self.positional_encoding_wz(bev_mask_wz).to(dtype)
        # else:
        #     bev_pos_zh = None
        #     bev_pos_wz = None
        if self.use_one_pe:
            bev_pos_hw = self.positional_encoding(bs, device, 'z')
            bev_pos_zh = self.positional_encoding(bs, device, 'w')
            bev_pos_wz = self.positional_encoding(bs, device, 'h')
            bev_pos = [bev_pos_hw, bev_pos_zh, bev_pos_wz]
        else:
            bev_pos = None

        return self.transformer.get_bev_features(
            mlvl_feats,
            [bev_queries_hw, bev_queries_zh, bev_queries_wz],
            self.bev_h,
            self.bev_w,
            self.bev_z,
            grid_length=(self.real_h / self.bev_h,
                            self.real_w / self.bev_w),
            bev_pos=bev_pos,
            img_metas=img_metas,
            prev_bev=prev_bev,
        )


from mmcv.cnn.bricks.transformer import POSITIONAL_ENCODING
@POSITIONAL_ENCODING.register_module()
class CustomPositionalEncoding(BaseModule):
    """Position embedding with learnable embedding weights.

    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. The final returned dimension for
            each position is 2 times of this value.
        row_num_embed (int, optional): The dictionary size of row embeddings.
            Default 50.
        col_num_embed (int, optional): The dictionary size of col embeddings.
            Default 50.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_feats,
                 h, w, z,
                 init_cfg=dict(type='Uniform', layer='Embedding')):
        super().__init__(init_cfg)
        if not isinstance(num_feats, list):
            num_feats = [num_feats] * 3
        self.h_embed = nn.Embedding(h, num_feats[0])
        self.w_embed = nn.Embedding(w, num_feats[1])
        self.z_embed = nn.Embedding(z, num_feats[2])
        self.num_feats = num_feats
        self.h, self.w, self.z = h, w, z

    def forward(self, bs, device, ignore_axis='z'):
        """Forward function for `LearnedPositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        if ignore_axis == 'h':
            h_embed = torch.zeros(1, 1, self.num_feats[0], device=device).repeat(self.w, self.z, 1) # w, z, d
            w_embed = self.w_embed(torch.arange(self.w, device=device))
            w_embed = w_embed.reshape(self.w, 1, -1).repeat(1, self.z, 1)
            z_embed = self.z_embed(torch.arange(self.z, device=device))
            z_embed = z_embed.reshape(1, self.z, -1).repeat(self.w, 1, 1)
        elif ignore_axis == 'w':
            h_embed = self.h_embed(torch.arange(self.h, device=device))
            h_embed = h_embed.reshape(1, self.h, -1).repeat(self.z, 1, 1)
            w_embed = torch.zeros(1, 1, self.num_feats[1], device=device).repeat(self.z, self.h, 1)
            z_embed = self.z_embed(torch.arange(self.z, device=device))
            z_embed = z_embed.reshape(self.z, 1, -1).repeat(1, self.h, 1)
        elif ignore_axis == 'z':
            h_embed = self.h_embed(torch.arange(self.h, device=device))
            h_embed = h_embed.reshape(self.h, 1, -1).repeat(1, self.w, 1)
            w_embed = self.w_embed(torch.arange(self.w, device=device))
            w_embed = w_embed.reshape(1, self.w, -1).repeat(self.h, 1, 1)
            z_embed = torch.zeros(1, 1, self.num_feats[2], device=device).repeat(self.h, self.w, 1)

        pos = torch.cat(
            (h_embed, w_embed, z_embed), dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(
                bs, 1, 1, 1)
        return pos

