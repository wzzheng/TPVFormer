
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [0, -25.6, -2, 51.2, 25.6, 4.4]
voxel_size = [0.2, 0.2, 8]

_dim_ = 96
num_heads = 6
_pos_dim_ = _dim_ // 3
_ffn_dim_ = _dim_ * 2
_num_levels_ = 4
_num_cams_ = 1

bev_h_ = 128
bev_w_ = 128
bev_z_ = 16
scale_h = 2
scale_w = 2
scale_z = 2
bev_encoder_layers = 5
num_points_in_pillar = [4, 32, 32]
num_points = [8, 64, 64]
nbr_class = 20

feature = _dim_
project_scale = 2

lr = 2e-4
weight_decay = 0.01
cos_lr = True
# decoder_checkpoint = True
# grid_size = [bev_h_*scale_h, bev_w_*scale_w, bev_z_*scale_z]

self_cross_layer = dict(
    type='TPVFormerLayer',
    # use_checkpoint=True,
    attn_cfgs=[
        dict(
            type='TPVTemporalSelfAttention',
            bev_h=bev_h_,
            bev_w=bev_w_,
            bev_z=bev_z_,
            num_anchors=16,
            embed_dims=_dim_,
            num_heads=num_heads,
            num_points=32,
            init_mode=1,
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
    # use_checkpoint=True,
    attn_cfgs=[
        dict(
            type='TPVTemporalSelfAttention',
            bev_h=bev_h_,
            bev_w=bev_w_,
            bev_z=bev_z_,
            num_anchors=16,
            embed_dims=_dim_,
            num_heads=num_heads,
            num_points=32,
            init_mode=1,
        )
    ],
    feedforward_channels=_ffn_dim_,
    ffn_dropout=0.1,
    operation_order=('self_attn', 'norm', 'ffn', 'norm')
)

model = dict(
    type='TPVFormer',
    use_grid_mask=True,
    video_test_mode=True,
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
        scale_z=scale_z,
        # use_checkpoint=True
    ),
    # img_backbone=dict(
    #     type='ResNet',
    #     depth=101,
    #     num_stages=4,
    #     out_indices=(1, 2, 3),
    #     frozen_stages=1,
    #     norm_cfg=dict(type='BN2d', requires_grad=False),
    #     norm_eval=True,
    #     style='caffe',
    #     dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), # original DCNv2 will print log when perform load_state_dict
    #     stage_with_dcn=(False, False, True, True)),
    # img_neck=dict(
    #     type='FPN',
    #     in_channels=[512, 1024, 2048],
    #     out_channels=_dim_,
    #     start_level=0,
    #     add_extra_convs='on_output',
    #     num_outs=4,
    #     relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='TPVFormerHead',
        bev_h=bev_h_,
        bev_w=bev_w_,
        bev_z=bev_z_,
        pc_range=point_cloud_range,
        num_query=900,
        num_classes=10,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type='TPVPerceptionTransformer',
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=False,
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
                ]),
            decoder=dict(
                type='TPVDetectionTransformerDecoder',
                num_layers=3,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                         dict(
                            type='TPVCustomMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1,
                            floor_sampling_offset=False,
                         ),
                    ],

                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10),
        positional_encoding=dict(
            type='CustomPositionalEncoding',
            num_feats=_pos_dim_,
            h=bev_h_,
            w=bev_w_,
            z=bev_z_
            ),
        # positional_encoding_zh=dict(
        #     type='LearnedPositionalEncoding',
        #     num_feats=_pos_dim_,
        #     row_num_embed=bev_z_,
        #     col_num_embed=bev_h_,
        #     ),
        # positional_encoding_wz=dict(
        #     type='LearnedPositionalEncoding',
        #     num_feats=_pos_dim_,
        #     row_num_embed=bev_w_,
        #     col_num_embed=bev_z_,
        #     ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head.
            pc_range=point_cloud_range))))


# optimizer = dict(
#     type='AdamW',
#     lr=2e-4,
#     paramwise_cfg=dict(
#         custom_keys={
#             'img_backbone': dict(lr_mult=0.1),
#         }),
#     weight_decay=0.01)

# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# # learning policy
# lr_config = dict(
#     policy='CosineAnnealing',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=1.0 / 3,
#     min_lr_ratio=1e-3)

# print_freq = 50
# max_epochs = 24
# grad_max_norm = 35

# load_from = './ckpts/r101_dcn_fcos3d_pretrain.pth'
