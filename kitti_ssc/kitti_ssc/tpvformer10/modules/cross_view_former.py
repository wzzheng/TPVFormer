import torch, numpy as np
import copy, warnings
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import TransformerLayerSequence, build_positional_encoding
from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import TORCH_VERSION, digit_version

from .custom_base_transformer_layer import MyCustomBaseTransformerLayer


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class CrossViewEncoder(TransformerLayerSequence):

    """
    Attention with both self and cross
    Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args,
                 bev_h=None,
                 bev_w=None,
                 bev_z=None,
                 num_points_in_pillar=4,
                 positional_encoding_hw=None,
                 positional_encoding_zh=None,
                 positional_encoding_wz=None,
                 return_intermediate=False, 
                #  dataset_type='nuscenes',
                 **kwargs):

        super().__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False
        
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_z = bev_z
        self.num_points_in_pillar = num_points_in_pillar

        # level 1: hw plane
        # level 2: zh plane
        # level 3: wz plane
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

        # reference_points = torch.stack([reference_points[..., 1], reference_points[..., 0]], dim=-1)
        
        self.register_buffer('reference_points', reference_points)

        self.positional_encoding_hw = build_positional_encoding(positional_encoding_hw)
        self.positional_encoding_zh = build_positional_encoding(positional_encoding_zh)
        self.positional_encoding_wz = build_positional_encoding(positional_encoding_wz)

        spatial_shapes = torch.tensor([[bev_h, bev_w], [bev_z, bev_h], [bev_w, bev_z]], dtype=torch.long)
        self.register_buffer('spatial_shapes', spatial_shapes)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        self.register_buffer('level_start_index', level_start_index)

    @auto_fp16()
    def forward(self, bevs, *args, **kwargs):

        # import pdb; pdb.set_trace()
        bev_hw, bev_zh, bev_wz = bevs[0], bevs[1], bevs[2]
        bs, _, c = bev_hw.shape
        dtype, device = bev_hw.dtype, bev_hw.device
        bev_mask_hw = torch.zeros((bs, self.bev_h, self.bev_w), device=device).to(dtype)
        bev_mask_zh = torch.zeros((bs, self.bev_z, self.bev_h), device=device).to(dtype)
        bev_mask_wz = torch.zeros((bs, self.bev_w, self.bev_z), device=device).to(dtype)
        bev_pos_hw = self.positional_encoding_hw(bev_mask_hw).to(dtype).flatten(2).permute(0, 2, 1) # bs, c, h, w
        bev_pos_zh = self.positional_encoding_zh(bev_mask_zh).to(dtype).flatten(2).permute(0, 2, 1)
        bev_pos_wz = self.positional_encoding_wz(bev_mask_wz).to(dtype).flatten(2).permute(0, 2, 1)
        
        # output = bev_query
        intermediate = []
        
        # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
        bev_hw_q = bev_hw + bev_pos_hw
        bev_zh_q = bev_zh + bev_pos_zh
        bev_wz_q = bev_wz + bev_pos_wz

        bev_q = torch.cat([bev_hw_q, bev_zh_q, bev_wz_q], dim=1) # bs, hw+zh+wz, c
        bev_kv = torch.cat([bev_hw, bev_zh, bev_wz], dim=1).unsqueeze(0).permute(0, 2, 1, 3) # 1, hw+zh+wz, bs, c

        for lid, layer in enumerate(self.layers):
            # import pdb; pdb.set_trace()
            output = layer(
                bev_q,
                bev_kv,
                bev_kv,
                *args,
                spatial_shapes=self.spatial_shapes,
                level_start_index=self.level_start_index,
                reference_points_cam=self.reference_points,
                **kwargs)

            bev_q = output
            bev_kv = output.permute(1, 0, 2).unsqueeze(0)
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)
        
        return output


from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.transformer import build_attention
from mmcv.cnn import xavier_init, constant_init
import torch.nn as nn, math
from mmcv.utils import ext_loader
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32, \
    MultiScaleDeformableAttnFunction_fp16
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch


@ATTENTION.register_module()
class CrossViewAttention(BaseModule):
    """An attention module used in BEVFormer.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_cams (int): The number of cameras
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        deformable_attention: (dict): The config for the deformable attention used in SCA.
    """

    def __init__(self,
                 embed_dims=256,
                 num_cams=6,
                 pc_range=None,
                 dropout=0.1,
                 init_cfg=None,
                 batch_first=False,
                 deformable_attention=dict(
                     type='MSDeformableAttention3D',
                     embed_dims=256,
                     num_levels=4),
                 **kwargs
                 ):
        super().__init__(init_cfg)

        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.deformable_attention = build_attention(deformable_attention)
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.batch_first = batch_first

        
        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
    
    @force_fp32(apply_to=('query', 'key', 'value', 'query_pos', 'reference_points_cam'))
    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                reference_points_cam=None,
                bev_mask=None,
                level_start_index=None,
                flag='encoder',
                **kwargs):

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
            # slots = torch.zeros_like(query)
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.size()

        max_len = num_query
        num_cams, l, bs, embed_dims = key.shape

        key = key.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)
        value = value.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)
        
        reference_points_cam = reference_points_cam.unsqueeze(0).expand(bs*num_cams, -1, -1, -1, -1)

        queries = self.deformable_attention(query=query, key=key, value=value,
                                            reference_points=reference_points_cam, spatial_shapes=spatial_shapes,
                                            level_start_index=level_start_index).view(bs, max_len, self.embed_dims)

        queries = self.output_proj(queries)

        return self.dropout(queries) + inp_residual


@ATTENTION.register_module()
class CrossViewDeformableAttention3D(BaseModule):

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=8,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.output_proj = None
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)

        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.
        Args:
            query (Tensor): Query of Transformer with shape
                ( bs, num_query, embed_dims).
            key (Tensor): The key tensor with shape
                `(bs, num_key,  embed_dims)`.
            value (Tensor): The value tensor with shape
                `(bs, num_key,  embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        # import pdb; pdb.set_trace()
        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)

        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)

        if reference_points.shape[-1] == 2:
            """
            For each BEV query, it owns `num_Z_anchors` in 3D space that having different heights.
            After proejcting, each BEV query has `num_Z_anchors` reference points in each 2D image.
            For each referent point, we sample `num_points` sampling points.
            For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.
            """
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)

            bs, num_query, _, num_Z_anchors, xy = reference_points.shape
            reference_points = reference_points[:, :, None, :, None, :, :]
            sampling_offsets = sampling_offsets / \
                offset_normalizer[None, None, None, :, None, :]
            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
            # import pdb; pdb.set_trace()
            sampling_offsets = sampling_offsets.view(
                bs, num_query, num_heads, num_levels, num_all_points // num_Z_anchors, num_Z_anchors, xy)
            sampling_locations = reference_points + sampling_offsets
            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
            assert num_all_points == num_points * num_Z_anchors

            sampling_locations = sampling_locations.view(
                bs, num_query, num_heads, num_levels, num_all_points, xy)

        elif reference_points.shape[-1] == 4:
            assert False
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        #  sampling_locations.shape: bs, num_query, num_heads, num_levels, num_all_points, 2
        #  attention_weights.shape: bs, num_query, num_heads, num_levels, num_all_points
        #

        if torch.cuda.is_available() and value.is_cuda:
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)
        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return output
