from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
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
    """

    def __init__(self, *args, tpv_h, tpv_w, tpv_z, pc_range=None, 
                 num_points_in_pillar=[4, 32, 32], return_intermediate=False, 
                 **kwargs):

        super().__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

        self.tpv_h, self.tpv_w, self.tpv_z = tpv_h, tpv_w, tpv_z
        self.num_points_in_pillar = num_points_in_pillar
        assert num_points_in_pillar[1] == num_points_in_pillar[2] and num_points_in_pillar[1] % num_points_in_pillar[0] == 0
        self.pc_range = pc_range
        self.fp16_enabled = False
        ref_3d_hw = self.get_reference_points(tpv_h, tpv_w, pc_range[5]-pc_range[2], num_points_in_pillar[0], '3d', device='cpu')

        ref_3d_zh = self.get_reference_points(tpv_z, tpv_h, pc_range[3]-pc_range[0], num_points_in_pillar[1], '3d', device='cpu')
        ref_3d_zh = ref_3d_zh.permute(3, 0, 1, 2)[[2, 0, 1]]
        ref_3d_zh = ref_3d_zh.permute(1, 2, 3, 0)

        ref_3d_wz = self.get_reference_points(tpv_w, tpv_z, pc_range[4]-pc_range[1], num_points_in_pillar[2], '3d', device='cpu')
        ref_3d_wz = ref_3d_wz.permute(3, 0, 1, 2)[[1, 2, 0]]
        ref_3d_wz = ref_3d_wz.permute(1, 2, 3, 0)
        self.register_buffer('ref_3d_hw', ref_3d_hw)
        self.register_buffer('ref_3d_zh', ref_3d_zh)
        self.register_buffer('ref_3d_wz', ref_3d_wz)
        
        ref_2d_hw = self.get_reference_points(tpv_h, tpv_w, dim='2d', bs=1, device='cpu')
        ref_2d_zh = self.get_reference_points(tpv_z, tpv_h, dim='2d', bs=1, device='cpu')
        ref_2d_wz = self.get_reference_points(tpv_w, tpv_z, dim='2d', bs=1, device='cpu')
        self.register_buffer('ref_2d_hw', ref_2d_hw)
        self.register_buffer('ref_2d_zh', ref_2d_zh)
        self.register_buffer('ref_2d_wz', ref_2d_wz)

    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in spatial cross-attn and self-attn.
        Args:
            H, W: spatial shape of tpv plane.
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
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, -1).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, -1, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D plane, used in temporal self-attention (TSA).
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
    def point_sampling(self, reference_points, pc_range, img_metas):

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
            1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

        reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                            reference_points.to(torch.float32)).squeeze(-1)
        eps = 1e-5

        tpv_mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)

        reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
        reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]

        tpv_mask = (tpv_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            tpv_mask = torch.nan_to_num(tpv_mask)
        else:
            tpv_mask = tpv_mask.new_tensor(
                np.nan_to_num(tpv_mask.cpu().numpy()))

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        tpv_mask = tpv_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        return reference_points_cam, tpv_mask

    @auto_fp16()
    def forward(self,
                tpv_query, # list
                key,
                value,
                *args,
                tpv_h=None,
                tpv_w=None,
                tpv_z=None,
                tpv_pos=None, # list
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            tpv_query (List[Tensor]): Input tpv query with shape
                `[(bs, num_query, embed_dims)` * 3].
            key & value (Tensor): Input multi-cameta features with shape
                (num_cam, num_value, bs, embed_dims)
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
        """
        output = tpv_query
        intermediate = []

        bs = tpv_query[0].shape[0]

        reference_points_cams, tpv_masks = [], []
        ref_3ds = [self.ref_3d_hw, self.ref_3d_zh, self.ref_3d_wz]
        for ref_3d in ref_3ds:
            reference_points_cam, tpv_mask = self.point_sampling(
                ref_3d, self.pc_range, kwargs['img_metas']) # num_cam, bs, hw++, #p, 2
            reference_points_cams.append(reference_points_cam)
            tpv_masks.append(tpv_mask)
        
        ref_2d_hw = self.ref_2d_hw.clone().expand(bs, -1, -1, -1)
        hybird_ref_2d = torch.cat([ref_2d_hw, ref_2d_hw], 0)

        for lid, layer in enumerate(self.layers):
            output = layer(
                tpv_query,
                key,
                value,
                *args,
                tpv_pos=tpv_pos,
                ref_2d=hybird_ref_2d,
                tpv_h=tpv_h,
                tpv_w=tpv_w,
                tpv_z=tpv_z,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cams=reference_points_cams,
                tpv_masks=tpv_masks,
                **kwargs)
            tpv_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output