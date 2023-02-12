_base_ = [
    './_base_/dataset.py',
    './_base_/optimizer.py',
    './_base_/schedule.py',
]

occupancy = False

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

_dim_ = 128
num_heads = 8
_pos_dim_ = [48, 48, 32]
_ffn_dim_ = _dim_*2
_num_levels_ = 4
_num_cams_ = 6

bev_h_ = 200
bev_w_ = 200
bev_z_ = 16
scale_h = 1
scale_w = 1
scale_z = 1
bev_encoder_layers = 5
num_points_in_pillar = [4, 32, 32]
num_points = [8, 64, 64]
hybrid_attn_anchors = 16
hybrid_attn_points = 32
hybrid_attn_init = 0

grid_size = [bev_h_*scale_h, bev_w_*scale_w, bev_z_*scale_z]
nbr_class = 17

self_cross_layer = dict(
    type='TPVFormerLayer',
    attn_cfgs=[
        dict(
            type='TPVTemporalSelfAttention',
            bev_h=bev_h_,
            bev_w=bev_w_,
            bev_z=bev_z_,
            num_anchors=hybrid_attn_anchors,
            embed_dims=_dim_,
            num_heads=num_heads,
            num_points=hybrid_attn_points,
            init_mode=hybrid_attn_init,
        ),
        dict(
            type='TPVSpatialCrossAttention',
            pc_range=point_cloud_range,
            num_cams=_num_cams_,
            deformable_attention=dict(
                type='TPVMSDeformableAttention3D',
                embed_dims=_dim_,
                num_heads=num_heads,
                num_points=num_points,
                num_z_anchors=num_points_in_pillar,
                num_levels=_num_levels_,
                floor_sampling_offset=False,
                bev_h=bev_h_,
                bev_w=bev_w_,
                bev_z=bev_z_,
            ),
            embed_dims=_dim_,
            bev_h=bev_h_,
            bev_w=bev_w_,
            bev_z=bev_z_,
        )
    ],
    feedforward_channels=_ffn_dim_,
    ffn_dropout=0.1,
    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
)

self_layer = dict(
    type='TPVFormerLayer',
    attn_cfgs=[
        dict(
            type='TPVTemporalSelfAttention',
            bev_h=bev_h_,
            bev_w=bev_w_,
            bev_z=bev_z_,
            num_anchors=hybrid_attn_anchors,
            embed_dims=_dim_,
            num_heads=num_heads,
            num_points=hybrid_attn_points,
            init_mode=hybrid_attn_init,
        )
    ],
    feedforward_channels=_ffn_dim_,
    ffn_dropout=0.1,
    operation_order=('self_attn', 'norm', 'ffn', 'norm')
)


model = dict(
    type='TPVFormer',
    use_grid_mask=True,
    fusion_head = dict(
        type='TPVFuser',
        bev_h=bev_h_,
        bev_w=bev_w_,
        bev_z=bev_z_,
        nbr_classes=nbr_class,
        in_dims=_dim_,
        hidden_dims=2*_dim_,
        out_dims=_dim_,
        scale_h=scale_h,
        scale_w=scale_w,
        scale_z=scale_z
    ),
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), # original DCNv2 will print log when perform load_state_dict
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='TPVFormerHead',
        bev_h=bev_h_,
        bev_w=bev_w_,
        bev_z=bev_z_,
        pc_range=point_cloud_range,
        transformer=dict(
            type='TPVPerceptionTransformer',
            embed_dims=_dim_,
            num_cams=_num_cams_,
            encoder=dict(
                type='TPVFormerEncoder',
                bev_h=bev_h_,
                bev_w=bev_w_,
                bev_z=bev_z_,
                num_layers=bev_encoder_layers,
                pc_range=point_cloud_range,
                num_points_in_pillar=num_points_in_pillar,
                num_points_in_pillar_cross_view=[16, 16, 16],
                return_intermediate=False,
                transformerlayers=[
                    self_cross_layer,
                    self_cross_layer,
                    self_cross_layer,
                    self_layer,
                    self_layer,
                ])),
        positional_encoding=dict(
            type='CustomPositionalEncoding',
            num_feats=_pos_dim_,
            h=bev_h_,
            w=bev_w_,
            z=bev_z_
        )))