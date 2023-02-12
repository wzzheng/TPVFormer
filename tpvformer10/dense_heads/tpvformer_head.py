
import torch
import torch.nn as nn

from mmcv.runner import BaseModule
from mmseg.models import builder, HEADS
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner import force_fp32, auto_fp16


@HEADS.register_module()
class TPVFormerHead(BaseModule):

    def __init__(self,
                 *args,
                 transformer=None,
                 positional_encoding=None,
                 bev_h=30,
                 bev_w=30,
                 bev_z=30,
                 pc_range,
                 **kwargs):
        
        super().__init__()

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_z = bev_z
        self.fp16_enabled = False

        self.pc_range = pc_range
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_z = self.pc_range[5] - self.pc_range[2]

        self.positional_encoding = build_positional_encoding(positional_encoding)
        self.transformer = builder.build_head(transformer)
        self.embed_dims = self.transformer.embed_dims
        self._init_layers()

    def _init_layers(self):
        self.bev_embedding_hw = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)
        self.bev_embedding_zh = nn.Embedding(self.bev_z * self.bev_h, self.embed_dims)
        self.bev_embedding_wz = nn.Embedding(self.bev_w * self.bev_z, self.embed_dims)

    def init_weights(self):
        self.transformer.init_weights()

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, img_metas):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        """
        bs = mlvl_feats[0].shape[0]
        dtype = mlvl_feats[0].dtype
        device = mlvl_feats[0].device
        bev_queries_hw = self.bev_embedding_hw.weight.to(dtype)
        bev_queries_zh = self.bev_embedding_zh.weight.to(dtype)
        bev_queries_wz = self.bev_embedding_wz.weight.to(dtype)

        bev_pos_hw = self.positional_encoding(bs, device, 'z')
        bev_pos_zh = self.positional_encoding(bs, device, 'w')
        bev_pos_wz = self.positional_encoding(bs, device, 'h')
        bev_pos = [bev_pos_hw, bev_pos_zh, bev_pos_wz]

        return self.transformer(
            mlvl_feats,
            [bev_queries_hw, bev_queries_zh, bev_queries_wz],
            self.bev_h,
            self.bev_w,
            self.bev_z,
            bev_pos=bev_pos,
            img_metas=img_metas,
        )


from mmcv.cnn.bricks.transformer import POSITIONAL_ENCODING
@POSITIONAL_ENCODING.register_module()
class CustomPositionalEncoding(BaseModule):

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

