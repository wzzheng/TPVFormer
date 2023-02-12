
import torch
import torch.nn as nn
from torch.nn.init import normal_

from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule
from mmseg.models import HEADS

from .temporal_self_attention import TPVTemporalSelfAttention
from .spatial_cross_attention import TPVMSDeformableAttention3D

@HEADS.register_module()
class TPVPerceptionTransformer(BaseModule):

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 encoder=None,
                 embed_dims=256,
                 use_cams_embeds=True,
                 **kwargs
        ):
        super().__init__(**kwargs)
        self.encoder = build_transformer_layer_sequence(encoder)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.use_cams_embeds = use_cams_embeds
        self.fp16_enabled = False

        self.init_layers()

    def init_layers(self):
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, TPVMSDeformableAttention3D) or isinstance(m, TPVTemporalSelfAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)

    def forward(
            self,
            mlvl_feats,
            bev_queries, # list
            bev_h,
            bev_w,
            bev_z,
            bev_pos=None, # list
            **kwargs):
        """
        obtain tpv features.
        """

        bs = mlvl_feats[0].size(0) # bs, num_cam, C, h, w
        bev_queries_hw = bev_queries[0].unsqueeze(1).repeat(1, bs, 1)
        bev_queries_zh = bev_queries[1].unsqueeze(1).repeat(1, bs, 1)
        bev_queries_wz = bev_queries[2].unsqueeze(1).repeat(1, bs, 1)
        bev_pos_hw = bev_pos[0].flatten(2).permute(2, 0, 1)

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2) # num_cam, bs, hw, c
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2) # num_cam, bs, hw++, c
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_pos_hw.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)
        bev_embed = self.encoder(
            [bev_queries_hw, bev_queries_zh, bev_queries_wz],
            feat_flatten,
            feat_flatten,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_z=bev_z,
            bev_pos=[bev_pos_hw, None, None],
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            **kwargs
        )

        return bev_embed
