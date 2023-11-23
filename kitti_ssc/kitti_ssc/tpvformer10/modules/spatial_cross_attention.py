
# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import warnings
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.cnn.bricks.transformer import build_attention
import math
from mmcv.runner import force_fp32
from mmcv.runner.base_module import BaseModule

from mmcv.utils import ext_loader
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@ATTENTION.register_module()
class TPVSpatialCrossAttention(BaseModule):
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
                 bev_h=None,
                 bev_w=None,
                 bev_z=None,
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
        self.bev_h, self.bev_w, self.bev_z = bev_h, bev_w, bev_z
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
                reference_points_cams=None,
                bev_masks=None,
                level_start_index=None,
                flag='encoder',
                mapp=None,
                multiplier=1,
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for  `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        # import pdb; pdb.set_trace()
        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.size()

        queries = torch.split(query, [self.bev_h*self.bev_w, self.bev_z*self.bev_h, self.bev_w*self.bev_z], dim=1)
        if residual is None:
            slots = [torch.zeros_like(q) for q in queries]
        indexeses = []
        max_lens = []
        queries_rebatches = []
        reference_points_rebatches = []
        for bev_idx, bev_mask in enumerate(bev_masks):
            indexes = []
            for _, mask_per_img in enumerate(bev_mask):
                index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
                indexes.append(index_query_per_img)
            max_len = max([len(each) for each in indexes])
            max_lens.append(max_len)
            indexeses.append(indexes)

            reference_points_cam = reference_points_cams[bev_idx]
            D = reference_points_cam.size(3)

            queries_rebatch = queries[bev_idx].new_zeros(
                [bs * self.num_cams, max_len, self.embed_dims])
            reference_points_rebatch = reference_points_cam.new_zeros(
                [bs * self.num_cams, max_len, D, 2])

            for i, reference_points_per_img in enumerate(reference_points_cam):
                for j in range(bs):
                    index_query_per_img = indexes[i]
                    queries_rebatch[j * self.num_cams + i, :len(index_query_per_img)] = queries[bev_idx][j, index_query_per_img]
                    reference_points_rebatch[j * self.num_cams + i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]
            
            queries_rebatches.append(queries_rebatch)
            reference_points_rebatches.append(reference_points_rebatch)

        num_cams, l, bs, embed_dims = key.shape

        key = key.permute(0, 2, 1, 3).view(
            self.num_cams * bs, l, self.embed_dims)
        value = value.permute(0, 2, 1, 3).view(
            self.num_cams * bs, l, self.embed_dims)

        queries = self.deformable_attention(query=queries_rebatches, key=key, value=value, ref_3d=None,
                                            reference_points=reference_points_rebatches, spatial_shapes=spatial_shapes,
                                            level_start_index=level_start_index, indexeses=indexeses)
        
        for bev_idx, indexes in enumerate(indexeses):
            for i, index_query_per_img in enumerate(indexes):
                for j in range(bs):
                    slots[bev_idx][j, index_query_per_img] += queries[bev_idx][j * self.num_cams + i, :len(index_query_per_img)]

            count = bev_masks[bev_idx].sum(-1) > 0
            count = count.permute(1, 2, 0).sum(-1)
            count = torch.clamp(count, min=1.0)
            slots[bev_idx] = slots[bev_idx] / count[..., None]
        slots = torch.cat(slots, dim=1)
        slots = self.output_proj(slots)

        return self.dropout(slots) + inp_residual


@ATTENTION.register_module()
class TPVMSDeformableAttention3D(BaseModule):
    """An attention module used in BEVFormer based on Deformable-Detr.
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
                 num_points=[8, 64, 64],
                 num_z_anchors=[4, 32, 32],
                 pc_range=None,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None,
                 floor_sampling_offset=True,
                 bev_h=None,
                 bev_w=None,
                 bev_z=None,
                ):
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
        self.num_z_anchors = num_z_anchors
        self.base_num_points = num_points[0]
        self.base_z_anchors = num_z_anchors[0]
        self.points_multiplier = [points // self.base_z_anchors for points in num_z_anchors]
        self.pc_range = pc_range
        self.bev_h, self.bev_w, self.bev_z = bev_h, bev_w, bev_z
        # self.sampling_offsets = nn.Linear(embed_dims, 
        #     num_heads * num_levels * num_points * 2)
        self.sampling_offsets = nn.ModuleList([
            nn.Linear(embed_dims, num_heads * num_levels * num_points[i] * 2) for i in range(3)
        ])
        self.floor_sampling_offset = floor_sampling_offset
        # self.attention_weights = nn.Linear(embed_dims,
        #     num_heads * num_levels * num_points)
        self.attention_weights = nn.ModuleList([
            nn.Linear(embed_dims, num_heads * num_levels * num_points[i]) for i in range(3)
        ])
        self.value_proj = nn.Linear(embed_dims, embed_dims)

        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        for i in range(3):
            constant_init(self.sampling_offsets[i], 0.)
            thetas = torch.arange(
                self.num_heads,
                dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            grid_init = (grid_init /
                        grid_init.abs().max(-1, keepdim=True)[0]).view(
                self.num_heads, 1, 1,
                2).repeat(1, self.num_levels, self.num_points[i], 1)
            grid_init = grid_init.reshape(self.num_heads, self.num_levels, self.num_z_anchors[i], -1, 2)
            for j in range(self.num_points[i] // self.num_z_anchors[i]):
                grid_init[:, :, :, j, :] *= j + 1
        
            self.sampling_offsets[i].bias.data = grid_init.view(-1)
            constant_init(self.attention_weights[i], val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def get_sampling_offsets_and_attention(self, queries):
        offsets = []
        attns = []
        for i, (query, fc, attn) in enumerate(zip(queries, self.sampling_offsets, self.attention_weights)):
            bs, l, d = query.shape

            offset = fc(query).reshape(bs, l, self.num_heads, self.num_levels, self.points_multiplier[i], -1, 2)
            offset = offset.permute(0, 1, 4, 2, 3, 5, 6).flatten(1, 2)
            offsets.append(offset)

            attention = attn(query).reshape(bs, l, self.num_heads, -1)
            attention = attention.softmax(-1)
            attention = attention.view(bs, l, self.num_heads, self.num_levels, self.points_multiplier[i], -1)
            attention = attention.permute(0, 1, 4, 2, 3, 5).flatten(1, 2)
            attns.append(attention)
        
        offsets = torch.cat(offsets, dim=1)
        attns = torch.cat(attns, dim=1)
        return offsets, attns

    def reshape_reference_points(self, reference_points):
        reference_point_list = []
        for i, reference_point in enumerate(reference_points):
            bs, l, z_anchors, _  = reference_point.shape
            reference_point = reference_point.reshape(bs, l, self.points_multiplier[i], -1, 2)
            reference_point = reference_point.flatten(1, 2)
            reference_point_list.append(reference_point)
        return torch.cat(reference_point_list, dim=1)
    
    def reshape_output(self, output, lens):
        bs, _, d = output.shape
        outputs = torch.split(output, [lens[0]*self.points_multiplier[0], lens[1]*self.points_multiplier[1], lens[2]*self.points_multiplier[2]], dim=1)
        
        outputs = [o.reshape(bs, -1, self.points_multiplier[i], d).sum(dim=2) for i, o in enumerate(outputs)]
        return outputs

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                ref_3d=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                indexeses=None,
                indexes_more=None,
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

        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = [q.permute(1, 0, 2) for q in query]
            value = value.permute(1, 0, 2)

        # bs, num_query, _ = query.shape
        query_lens = [q.shape[1] for q in query]
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)

        # sampling_offsets = self.sampling_offsets(query).view(
        #     bs, num_query, self.num_heads, self.num_levels, self.num_points, -1)
        sampling_offsets, attention_weights = self.get_sampling_offsets_and_attention(query)

        reference_points = self.reshape_reference_points(reference_points)

        # attention_weights = self.attention_weights(query).view(
            # bs, num_query, self.num_heads, self.num_levels * self.num_points)

        # attention_weights = attention_weights.softmax(-1)

        # attention_weights = attention_weights.view(bs, num_query,
        #                                            self.num_heads,
        #                                            self.num_levels,
        #                                            self.num_points)

        # # bs, l, D, 3
        # ref_3d[..., 0:1] = ref_3d[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        # ref_3d[..., 1:2] = ref_3d[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        # ref_3d[..., 2:3] = ref_3d[..., 2:3] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
        # ref_3d = ref_3d[:, :, None, None, None, :, :] # bs, l, 1, 1, 1, D, 3
        # bs, num_query, num_heads, num_levels, num_all_points, _ = sampling_offsets.shape
        # sampling_offsets = sampling_offsets.view(
        #     bs, num_query, num_heads, num_levels, num_all_points // self.num_z_anchors, self.num_z_anchors, -1)
        # sampling_locations = ref_3d + sampling_offsets

        # sampling_locations = sampling_locations.view(bs, num_query, -1, 3).permute(0, 2, 1, 3)

        # sampling_locations = sampling_locations.view(
        #     bs, num_query, num_heads, num_levels, num_all_points, -1)
        

        
        if reference_points.shape[-1] == 2:
            """
            For each BEV query, it owns `num_Z_anchors` in 3D space that having different heights.
            After proejcting, each BEV query has `num_Z_anchors` reference points in each 2D image.
            For each referent point, we sample `num_points` sampling points.
            For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.
            """
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)

            bs, num_query, num_Z_anchors, xy = reference_points.shape
            reference_points = reference_points[:, :, None, None, :, None, :]
            sampling_offsets = sampling_offsets / \
                offset_normalizer[None, None, None, :, None, :]
            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
            sampling_offsets = sampling_offsets.view(
                bs, num_query, num_heads, num_levels, num_Z_anchors, num_all_points // num_Z_anchors, xy)
            sampling_locations = reference_points + sampling_offsets
            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
            assert num_all_points == num_points * num_Z_anchors

            sampling_locations = sampling_locations.view(
                bs, num_query, num_heads, num_levels, num_all_points, xy)
            
            if self.floor_sampling_offset:
                sampling_locations = sampling_locations - torch.floor(sampling_locations)

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

        output = self.reshape_output(output, query_lens)
        if not self.batch_first:
            output = [o.permute(1, 0, 2) for o in output]

        return output


    # This function must use fp32!!!
    # @force_fp32(apply_to=('reference_points', 'img_metas'))
    # def point_sampling(self, reference_points, pc_range, img_metas):

    #     lidar2img = []
    #     for img_meta in img_metas:
    #         lidar2img.append(img_meta['lidar2img'])
    #     lidar2img = np.asarray(lidar2img)
    #     lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
    #     reference_points = reference_points.clone()

    #     # reference_points[..., 0:1] = reference_points[..., 0:1] * \
    #     #     (pc_range[3] - pc_range[0]) + pc_range[0]
    #     # reference_points[..., 1:2] = reference_points[..., 1:2] * \
    #     #     (pc_range[4] - pc_range[1]) + pc_range[1]
    #     # reference_points[..., 2:3] = reference_points[..., 2:3] * \
    #     #     (pc_range[5] - pc_range[2]) + pc_range[2]

    #     reference_points = torch.cat(
    #         (reference_points, torch.ones_like(reference_points[..., :1])), -1)

    #     reference_points = reference_points.permute(1, 0, 2, 3)
    #     D, B, num_query = reference_points.size()[:3]
    #     num_cam = lidar2img.size(1)

    #     reference_points = reference_points.view(
    #         D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)

    #     lidar2img = lidar2img.view(
    #         1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

    #     reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
    #                                         reference_points.to(torch.float32)).squeeze(-1)
    #     eps = 1e-5

    #     bev_mask = (reference_points_cam[..., 2:3] > eps)
    #     reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
    #         reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)

    #     reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
    #     reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]

    #     bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
    #                 & (reference_points_cam[..., 1:2] < 1.0)
    #                 & (reference_points_cam[..., 0:1] < 1.0)
    #                 & (reference_points_cam[..., 0:1] > 0.0))
    #     if digit_version(TORCH_VERSION) >= digit_version('1.8'):
    #         bev_mask = torch.nan_to_num(bev_mask)
    #     else:
    #         bev_mask = bev_mask.new_tensor(
    #             np.nan_to_num(bev_mask.cpu().numpy()))

    #     reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
    #     bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

    #     return reference_points_cam, bev_mask


if __name__ == "__main__":
    value = torch.randint(10, (1, 16, 1, 2)).float().cuda()
    spatial_shapes = torch.tensor([[4, 4]]).cuda()
    level_start_index = torch.tensor([0]).cuda()
    sampling_locations = torch.tensor([1.0, 0.0]).reshape(1, 1, 1, 1, 1, -1).cuda()
    attention_weights = torch.tensor([1.0]).reshape(1, 1, 1, 1, 1).cuda()
    gpuFunc = MultiScaleDeformableAttnFunction_fp32.apply(
        value, spatial_shapes, level_start_index, sampling_locations, attention_weights, 64
    )

    cpuFun = multi_scale_deformable_attn_pytorch(
        value, spatial_shapes, sampling_locations, attention_weights
    )
    import pdb; pdb.set_trace()
    pass