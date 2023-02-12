_base_ = [
    './_base_/dataset.py',
    './_base_/optimizer.py',
    './_base_/schedule.py',
]

dataset_params = dict(
    version = "v1.0-trainval",
    ignore_label = 0,
    fill_label = 17,
    fixed_volume_space = True,
    label_mapping = "./config/label_mapping/nuscenes-noIgnore.yaml",
    max_volume_space = [51.2, 51.2, 3],
    min_volume_space = [-51.2, -51.2, -5],
)

occupancy = True

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 4
bev_h_ = 100
bev_w_ = 100
bev_z_ = 8
scale_h = 1
scale_w = 1
scale_z = 1
grid_size = [bev_h_*scale_h, bev_w_*scale_w, bev_z_*scale_z]
num_points_in_pillar = [4, 32, 32]
num_points = [8, 64, 64]
nbr_class = 18

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
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), 
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
            encoder=dict(
                type='TPVFormerEncoder',
                bev_h=bev_h_,
                bev_w=bev_w_,
                bev_z=bev_z_,
                num_layers=3,
                pc_range=point_cloud_range,
                num_points_in_pillar=num_points_in_pillar,
                return_intermediate=False,
                transformerlayers=dict(
                    type='TPVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TPVTemporalSelfAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                        dict(
                            type='TPVSpatialCrossAttention',
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='TPVMSDeformableAttention3D',
                                embed_dims=_dim_,
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
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
            )))
