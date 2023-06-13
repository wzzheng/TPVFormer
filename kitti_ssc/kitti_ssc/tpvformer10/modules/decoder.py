# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import mmcv
import cv2 as cv
import copy
import warnings
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
import math
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmcv.utils import (ConfigDict, build_from_cfg, deprecated_api_warning,
                        to_2tuple)

from mmcv.utils import ext_loader
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32, \
    MultiScaleDeformableAttnFunction_fp16

ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.
    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class TPVDetectionTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False

    def forward(self,
                query,
                *args,
                reference_points=None,
                reg_branches=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `Detr3DTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):

            # reference_points_input = reference_points[..., :2].unsqueeze(2)  # BS NUM_QUERY NUM_LEVEL 2
            reference_points_input = reference_points.unsqueeze(2)
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                **kwargs)
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)

                assert reference_points.shape[-1] == 3

                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[
                    ..., :2] + inverse_sigmoid(reference_points[..., :2])
                new_reference_points[..., 2:3] = tmp[
                    ..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])

                new_reference_points = new_reference_points.sigmoid()

                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points


@ATTENTION.register_module()
class TPVCustomMSDeformableAttention(BaseModule):
    """An attention module used in Deformable-Detr.

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=False,
                 norm_cfg=None,
                 init_cfg=None,
                 floor_sampling_offset=True,
                ):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
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
        self.sampling_offsets = nn.Linear(embed_dims, 
            num_heads * num_levels * num_points * 3)
        self.floor_sampling_offset = floor_sampling_offset
        self.attention_weights = nn.Linear(embed_dims,
            num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        # self.decoder = nn.Sequential(
        #     nn.Linear(dim_per_head, 2*dim_per_head),
        #     nn.Softplus(),
        #     nn.Linear(2*dim_per_head, dim_per_head)
        # )
        self.decoder = None
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        # thetas = torch.arange(
        #     self.num_heads,
        #     dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        # grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        # grid_init = (grid_init /
        #              grid_init.abs().max(-1, keepdim=True)[0]).view(
        #     self.num_heads, 1, 1,
        #     2).repeat(1, self.num_levels, self.num_points, 1)
        # for i in range(self.num_points):
        #     grid_init[:, :, i, :] *= i + 1
        # grid_init = torch.cat([grid_init, torch.zeros(self.num_heads, self.num_levels, self.num_points, 1)], dim=-1)

        grid_init = torch.randn(self.num_heads, self.num_levels, self.num_points, 3)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        if hasattr(self, 'value_proj'):
            xavier_init(self.value_proj, distribution='uniform', bias=0.)
        # xavier_init(self.decoder, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiScaleDeformableAttention')
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
                flag='decoder',
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
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
        assert value is not None
        value = torch.cat(value, dim=0)
        # if value is None:
            # value = query

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
        # assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        assert key_padding_mask is None
        # if key_padding_mask is not None:
            # value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        h, w, z = spatial_shapes[0]
        value = torch.split(value, [h*w, z*h, w*z], dim=1)
        # value = [v.view(bs, v.shape[1], self.num_heads, -1) for v in value]

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 3)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        if reference_points.shape[-1] == 3:
            # offset_normalizer = torch.stack(
            #     [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            offset_normalizer = spatial_shapes
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
            
            if self.floor_sampling_offset:
                sampling_locations = sampling_locations - torch.floor(sampling_locations)
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        if False: # torch.cuda.is_available() and value.is_cuda:

            # using fp16 deformable attention is unstable because it performs many sum operations
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            # output = multi_scale_deformable_attn_pytorch(
            #     value, spatial_shapes, sampling_locations, attention_weights)
            output = custom_multi_scale_deformable_attn_pytorch(
                value, spatial_shapes[0], sampling_locations, attention_weights, self.decoder)

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity


def custom_multi_scale_deformable_attn_pytorch(value, value_spatial_shapes,
                                               sampling_locations, attention_weights, decoder):
    """CPU version of multi-scale deformable attention.

    Args:
        value (Tensor): The value has shape
            (bs, num_keys, mum_heads, embed_dims//num_heads)
        value_spatial_shapes (Tensor): Spatial shape of
            3d space, has shape (3)
        sampling_locations (Tensor): The location of sampling points,
            has shape
            (bs ,num_queries, num_heads, num_levels, num_points, 3),
            the last dimension 2 represent (x, y).
        attention_weights (Tensor): The weight of sampling points used
            when calculate the attention, has shape
            (bs ,num_queries, num_heads, num_levels, num_points),

    Returns:
        Tensor: has shape (bs, num_queries, embed_dims)
    """

    bs, _, num_heads, embed_dims = value[0].shape
    _, num_queries, num_heads, num_levels, num_points, _ =\
        sampling_locations.shape
    # value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes],
                            #  dim=1)
    assert num_levels == 1
    sampling_grids = 2 * sampling_locations.squeeze(3) - 1
    sampling_value_list = []
    # ss = value_spatial_shapes
    # order = [0, 2, 1]
    # for i, o in enumerate(order):
    #     # bs, H_*W_, num_heads, embed_dims ->
    #     # bs, H_*W_, num_heads*embed_dims ->
    #     # bs, num_heads*embed_dims, H_*W_ ->
    #     # bs*num_heads, embed_dims, H_, W_
    #     # value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(
    #     #     bs * num_heads, embed_dims, H_, W_)
    #     value_l_ = value[i].reshape(bs, ss[o], ss[(o+1)%3], num_heads, -1).permute(0, 3, 4, 1, 2).flatten(0, 1)
    #     # bs, num_queries, num_heads, num_points, 2 ->
    #     # bs, num_heads, num_queries, num_points, 2 ->
    #     # bs*num_heads, num_queries, num_points, 2
    #     sampling_grid_l_ = sampling_grids[...,[o, (o+1)%3]].transpose(1, 2).flatten(0, 1)
    #     # bs*num_heads, embed_dims, num_queries, num_points
    #     sampling_value_l_ = F.grid_sample(
    #         value_l_,
    #         sampling_grid_l_,
    #         mode='bilinear',
    #         padding_mode='zeros',
    #         align_corners=False)
    #     sampling_value_list.append(sampling_value_l_)
    
    w, h, z = value_spatial_shapes
    for i in range(3):
        if i == 0:
            value_l_ = value[0].reshape(bs, h, w, num_heads, -1).permute(0, 3, 4, 1, 2).flatten(0, 1)
            sampling_grid_l_ = sampling_grids[...,[0, 1]].transpose(1, 2).flatten(0, 1)
        elif i == 1:
            value_l_ = value[1].reshape(bs, z, h, num_heads, -1).permute(0, 3, 4, 1, 2).flatten(0, 1)
            sampling_grid_l_ = sampling_grids[...,[1, 2]].transpose(1, 2).flatten(0, 1)
        else:
            value_l_ = value[i].reshape(bs, w, z, num_heads, -1).permute(0, 3, 4, 1, 2).flatten(0, 1)
            sampling_grid_l_ = sampling_grids[...,[2, 0]].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    
    # bs*h, c, query, points
    space_feat = sampling_value_list[0] + sampling_value_list[1] + sampling_value_list[2]
    # space_feat = decoder(space_feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs*num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points)
    output = (space_feat * attention_weights).sum(-1).view(bs, num_heads * embed_dims, num_queries)
    return output.transpose(1, 2).contiguous()
