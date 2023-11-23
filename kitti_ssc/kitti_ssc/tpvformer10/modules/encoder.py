from .custom_base_transformer_layer import TPVMyCustomBaseTransformerLayer
import copy
import warnings
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.runner import force_fp32, auto_fp16
import numpy as np
import torch
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.utils import ext_loader
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class TPVFormerEncoder(TransformerLayerSequence):

    """
    Attention with both self and cross
    Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, bev_h, bev_w, bev_z, pc_range=None, 
                 num_points_in_pillar=[4, 32, 32], num_points_in_pillar_cross_view=[32, 32, 32],
                 return_intermediate=False, dataset_type='nuscenes', **kwargs):

        super().__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

        self.bev_h, self.bev_w, self.bev_z = bev_h, bev_w, bev_z
        self.num_points_in_pillar = num_points_in_pillar
        assert num_points_in_pillar[1] == num_points_in_pillar[2] and num_points_in_pillar[1] % num_points_in_pillar[0] == 0
        self.pc_range = pc_range
        self.fp16_enabled = False

        ref_3d_hw = self.get_reference_points(bev_h, bev_w, pc_range[5]-pc_range[2], num_points_in_pillar[0], '3d', device='cpu')

        ref_3d_zh = self.get_reference_points(bev_z, bev_h, pc_range[4]-pc_range[1], num_points_in_pillar[1], '3d', device='cpu')
        ref_3d_zh = ref_3d_zh.permute(3, 0, 1, 2)[[1, 2, 0]]
        ref_3d_zh = ref_3d_zh.permute(1, 2, 3, 0)

        ref_3d_wz = self.get_reference_points(bev_w, bev_z, pc_range[3]-pc_range[0], num_points_in_pillar[2], '3d', device='cpu')
        ref_3d_wz = ref_3d_wz.permute(3, 0, 1, 2)[[2, 0, 1]]
        ref_3d_wz = ref_3d_wz.permute(1, 2, 3, 0)
        self.register_buffer('ref_3d_hw', ref_3d_hw)
        self.register_buffer('ref_3d_zh', ref_3d_zh)
        self.register_buffer('ref_3d_wz', ref_3d_wz)
        
        ref_2d_hw = self.get_reference_points(bev_h, bev_w, dim='2d', bs=1, device='cpu')
        ref_2d_zh = self.get_reference_points(bev_z, bev_h, dim='2d', bs=1, device='cpu')
        ref_2d_wz = self.get_reference_points(bev_w, bev_z, dim='2d', bs=1, device='cpu')
        self.register_buffer('ref_2d_hw', ref_2d_hw)
        self.register_buffer('ref_2d_zh', ref_2d_zh)
        self.register_buffer('ref_2d_wz', ref_2d_wz)

        cross_view_ref_points = self.get_cross_view_ref_points(bev_h, bev_w, bev_z, num_points_in_pillar_cross_view)
        self.register_buffer('cross_view_ref_points', cross_view_ref_points)
        self.num_points_cross_view = num_points_in_pillar_cross_view


    @staticmethod
    def get_cross_view_ref_points(bev_h, bev_w, bev_z, num_points_in_pillar):
        # ref points generating target: (#query)hw+zh+wz, (#level)3, #p, 2
        # generate points for hw and level 1
        h_ranges = torch.linspace(0.5, bev_h-0.5, bev_h) / bev_h
        w_ranges = torch.linspace(0.5, bev_w-0.5, bev_w) / bev_w
        h_ranges = h_ranges.unsqueeze(-1).expand(-1, bev_w).flatten()
        w_ranges = w_ranges.unsqueeze(0).expand(bev_h, -1).flatten()
        hw_hw = torch.stack([w_ranges, h_ranges], dim=-1) # hw, 2
        hw_hw = hw_hw.unsqueeze(1).expand(-1, num_points_in_pillar[2], -1) # hw, #p, 2
        # generate points for hw and level 2
        z_ranges = torch.linspace(0.5, bev_z-0.5, num_points_in_pillar[2]) / bev_z # #p
        z_ranges = z_ranges.unsqueeze(0).expand(bev_h*bev_w, -1) # hw, #p
        h_ranges = torch.linspace(0.5, bev_h-0.5, bev_h) / bev_h
        h_ranges = h_ranges.reshape(-1, 1, 1).expand(-1, bev_w, num_points_in_pillar[2]).flatten(0, 1)
        hw_zh = torch.stack([h_ranges, z_ranges], dim=-1) # hw, #p, 2
        # generate points for hw and level 3
        z_ranges = torch.linspace(0.5, bev_z-0.5, num_points_in_pillar[2]) / bev_z # #p
        z_ranges = z_ranges.unsqueeze(0).expand(bev_h*bev_w, -1) # hw, #p
        w_ranges = torch.linspace(0.5, bev_w-0.5, bev_w) / bev_w
        w_ranges = w_ranges.reshape(1, -1, 1).expand(bev_h, -1, num_points_in_pillar[2]).flatten(0, 1)
        hw_wz = torch.stack([z_ranges, w_ranges], dim=-1) # hw, #p, 2
        
        # generate points for zh and level 1
        w_ranges = torch.linspace(0.5, bev_w-0.5, num_points_in_pillar[1]) / bev_w
        w_ranges = w_ranges.unsqueeze(0).expand(bev_z*bev_h, -1)
        h_ranges = torch.linspace(0.5, bev_h-0.5, bev_h) / bev_h
        h_ranges = h_ranges.reshape(1, -1, 1).expand(bev_z, -1, num_points_in_pillar[1]).flatten(0, 1)
        zh_hw = torch.stack([w_ranges, h_ranges], dim=-1)
        # generate points for zh and level 2
        z_ranges = torch.linspace(0.5, bev_z-0.5, bev_z) / bev_z
        z_ranges = z_ranges.reshape(-1, 1, 1).expand(-1, bev_h, num_points_in_pillar[1]).flatten(0, 1)
        h_ranges = torch.linspace(0.5, bev_h-0.5, bev_h) / bev_h
        h_ranges = h_ranges.reshape(1, -1, 1).expand(bev_z, -1, num_points_in_pillar[1]).flatten(0, 1)
        zh_zh = torch.stack([h_ranges, z_ranges], dim=-1) # zh, #p, 2
        # generate points for zh and level 3
        w_ranges = torch.linspace(0.5, bev_w-0.5, num_points_in_pillar[1]) / bev_w
        w_ranges = w_ranges.unsqueeze(0).expand(bev_z*bev_h, -1)
        z_ranges = torch.linspace(0.5, bev_z-0.5, bev_z) / bev_z
        z_ranges = z_ranges.reshape(-1, 1, 1).expand(-1, bev_h, num_points_in_pillar[1]).flatten(0, 1)
        zh_wz = torch.stack([z_ranges, w_ranges], dim=-1)

        # generate points for wz and level 1
        h_ranges = torch.linspace(0.5, bev_h-0.5, num_points_in_pillar[0]) / bev_h
        h_ranges = h_ranges.unsqueeze(0).expand(bev_w*bev_z, -1)
        w_ranges = torch.linspace(0.5, bev_w-0.5, bev_w) / bev_w
        w_ranges = w_ranges.reshape(-1, 1, 1).expand(-1, bev_z, num_points_in_pillar[0]).flatten(0, 1)
        wz_hw = torch.stack([w_ranges, h_ranges], dim=-1)
        # generate points for wz and level 2
        h_ranges = torch.linspace(0.5, bev_h-0.5, num_points_in_pillar[0]) / bev_h
        h_ranges = h_ranges.unsqueeze(0).expand(bev_w*bev_z, -1)
        z_ranges = torch.linspace(0.5, bev_z-0.5, bev_z) / bev_z
        z_ranges = z_ranges.reshape(1, -1, 1).expand(bev_w, -1, num_points_in_pillar[0]).flatten(0, 1)
        wz_zh = torch.stack([h_ranges, z_ranges], dim=-1)
        # generate points for wz and level 3
        w_ranges = torch.linspace(0.5, bev_w-0.5, bev_w) / bev_w
        w_ranges = w_ranges.reshape(-1, 1, 1).expand(-1, bev_z, num_points_in_pillar[0]).flatten(0, 1)
        z_ranges = torch.linspace(0.5, bev_z-0.5, bev_z) / bev_z
        z_ranges = z_ranges.reshape(1, -1, 1).expand(bev_w, -1, num_points_in_pillar[0]).flatten(0, 1)
        wz_wz = torch.stack([z_ranges, w_ranges], dim=-1)

        reference_points = torch.cat([
            torch.stack([hw_hw, hw_zh, hw_wz], dim=1),
            torch.stack([zh_hw, zh_zh, zh_wz], dim=1),
            torch.stack([wz_hw, wz_zh, wz_wz], dim=1)
        ], dim=0) # hw+zh+wz, 3, #p, 2
        
        return reference_points

    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, -1, 1).expand(num_points_in_pillar, H, W) / H
            ys = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, -1).expand(num_points_in_pillar, H, W) / W
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    # This function must use fp32!!!
    @force_fp32(apply_to=('reference_points', 'img_metas'))
    def point_sampling(self, reference_points, pc_range,  img_metas):

        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
        lidar2img = np.asarray(lidar2img)
        lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
        reference_points = reference_points.clone()

        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (pc_range[5] - pc_range[2]) + pc_range[2]

        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1)

        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)

        reference_points = reference_points.view(
            D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)

        lidar2img = lidar2img.view(
            1, B, num_cam, 1, 3, 4).repeat(D, 1, 1, num_query, 1, 1)

        reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                            reference_points.to(torch.float32)).squeeze(-1)
        eps = 1e-5

        bev_mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)
        
        img_W = img_metas[0]['img_shape'][0][1]
        for batch in range(B):
            if img_metas[batch]['fliplr']:
                reference_points_cam[:, batch, :, :, 0] = img_W - 1 - reference_points_cam[:, batch, :, :, 0]

        reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
        reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]

        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            bev_mask = torch.nan_to_num(bev_mask)
        else:
            bev_mask = bev_mask.new_tensor(
                np.nan_to_num(bev_mask.cpu().numpy()))

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        return reference_points_cam, bev_mask

    @auto_fp16()
    def forward(self,
                bev_query, # list
                key,
                value,
                *args,
                bev_h=None,
                bev_w=None,
                bev_z=None,
                bev_pos=None, # list
                spatial_shapes=None,
                level_start_index=None,
                valid_ratios=None,
                prev_bev=None,
                shift=0.,
                use_checkpoint=False,
                **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            bev_query (Tensor): Input BEV query with shape
                `(num_query, bs, embed_dims)`.
            key & value (Tensor): Input multi-cameta features with shape
                (num_cam, num_value, bs, embed_dims)
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = bev_query
        intermediate = []

        bs = bev_query[0].shape[1]

        reference_points_cams, bev_masks = [], []
        ref_3ds = [self.ref_3d_hw, self.ref_3d_zh, self.ref_3d_wz]
        for ref_3d in ref_3ds:
            reference_points_cam, bev_mask = self.point_sampling(
                ref_3d, self.pc_range, kwargs['img_metas']) # num_cam, bs, hw++, #p, 2
            reference_points_cams.append(reference_points_cam)
            bev_masks.append(bev_mask)
        
        # bug: this code should be 'shift_ref_2d = ref_2d.clone()', we keep this bug for reproducing our results in paper.
        # ref_2d_hw = self.ref_2d_hw.clone().expand(bs, -1, -1, -1)
        # ref_2d_zh = self.ref_2d_zh.clone().expand(bs, -1, -1, -1)
        # ref_2d_wz = self.ref_2d_wz.clone().expand(bs, -1, -1, -1)

        ref_cross_view = self.cross_view_ref_points.clone().unsqueeze(0).expand(bs, -1, -1, -1, -1)

        # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
        # bev_query_hw = bev_query_hw.permute(1, 0, 2)
        bev_query = [bev.permute(1, 0, 2) for bev in bev_query]
        bev_pos = [pos.permute(1, 0, 2) for pos in bev_pos]

        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                key,
                value,
                *args,
                bev_pos=bev_pos,
                # ref_2d=[ref_2d_hw, ref_2d_zh, ref_2d_wz],
                ref_2d=ref_cross_view,
                ref_3d=ref_3ds,
                bev_h=bev_h,
                bev_w=bev_w,
                bev_z=bev_z,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cams=reference_points_cams,
                bev_masks=bev_masks,
                prev_bev=prev_bev,
                **kwargs)
            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


@TRANSFORMER_LAYER.register_module()
class TPVFormerLayer(TPVMyCustomBaseTransformerLayer):
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 use_checkpoint=False,
                 **kwargs):
        super().__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False
        self.use_checkpoint = use_checkpoint
        # assert len(operation_order) == 6
        # assert set(operation_order) == set(
        #     ['self_attn', 'norm', 'cross_attn', 'ffn'])

    def forward(self,
                query,
                key=None,
                value=None,
                bev_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_2d=None,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                bev_z=None,
                reference_points_cam=None,
                mask=None,
                spatial_shapes=None,
                level_start_index=None,
                prev_bev=None,
                reference_points_cams=None,
                bev_masks=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        if self.operation_order[0] == 'cross_attn':
            query = torch.cat(query, dim=1)
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                                                     f'operation_order {self.num_attn}'
        for layer in self.operation_order:

            # temporal self attention
            if layer == 'self_attn':
                # ss_hw = torch.tensor([[bev_h, bev_w]], device=query[0].device)
                # ss_zh = torch.tensor([[bev_z, bev_h]], device=query[0].device)
                # ss_wz = torch.tensor([[bev_w, bev_z]], device=query[0].device)
                ss = torch.tensor([
                    [bev_h, bev_w],
                    [bev_z, bev_h],
                    [bev_w, bev_z]
                ], device=query[0].device)
                lsi = torch.tensor([
                    0, bev_h*bev_w, bev_h*bev_w+bev_z*bev_h
                ], device=query[0].device)

                if not isinstance(query, (list, tuple)):
                    query = torch.split(
                        query, [bev_h*bev_w, bev_z*bev_h, bev_w*bev_z], dim=1)

                query = self.attentions[attn_index](
                    query,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    reference_points=ref_2d,
                    # spatial_shapes=[ss_hw, ss_zh, ss_wz],
                    spatial_shapes=ss,
                    # level_start_index=torch.tensor([0], device=query[0].device),
                    level_start_index=lsi,
                    **kwargs)
                attn_index += 1
                query = torch.cat(query, dim=1)
                identity = query

            elif layer == 'norm':
                #### applied separately
                query = self.norms[norm_index](query)
                norm_index += 1

            # spaital cross attention
            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    reference_points_cams=reference_points_cams,
                    bev_masks=bev_masks,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                ffn = self.ffns[ffn_index]
                if self.use_checkpoint:
                    query = torch.utils.checkpoint.checkpoint(
                        ffn,
                        query, identity if self.pre_norm else None
                    )
                else:
                    query = ffn(
                        query, identity if self.pre_norm else None)
                ffn_index += 1
        query = torch.split(query, [bev_h*bev_w, bev_z*bev_h, bev_w*bev_z], dim=1)
        return query

