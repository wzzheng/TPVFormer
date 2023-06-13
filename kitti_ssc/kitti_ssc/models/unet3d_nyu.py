# encoding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from kitti_ssc.models.CRP3D import CPMegaVoxels
from kitti_ssc.models.modules import (
    Process,
    Upsample,
    Downsample,
    SegmentationHead,
    ASPP,
)


class UNet3D(nn.Module):
    def __init__(
        self,
        class_num,
        norm_layer,
        feature,
        full_scene_size,
        n_relations=4,
        project_res=[],
        context_prior=True,
        bn_momentum=0.1,
    ):
        super(UNet3D, self).__init__()
        self.business_layer = []
        self.project_res = project_res

        self.feature_1_4 = feature
        self.feature_1_8 = feature * 2
        self.feature_1_16 = feature * 4

        self.feature_1_16_dec = self.feature_1_16
        self.feature_1_8_dec = self.feature_1_8
        self.feature_1_4_dec = self.feature_1_4

        self.process_1_4 = nn.Sequential(
            Process(self.feature_1_4, norm_layer, bn_momentum, dilations=[1, 2, 3]),
            Downsample(self.feature_1_4, norm_layer, bn_momentum),
        )
        self.process_1_8 = nn.Sequential(
            Process(self.feature_1_8, norm_layer, bn_momentum, dilations=[1, 2, 3]),
            Downsample(self.feature_1_8, norm_layer, bn_momentum),
        )
        self.up_1_16_1_8 = Upsample(
            self.feature_1_16_dec, self.feature_1_8_dec, norm_layer, bn_momentum
        )
        self.up_1_8_1_4 = Upsample(
            self.feature_1_8_dec, self.feature_1_4_dec, norm_layer, bn_momentum
        )
        self.ssc_head_1_4 = SegmentationHead(
            self.feature_1_4_dec, self.feature_1_4_dec, class_num, [1, 2, 3]
        )

        self.context_prior = context_prior
        size_1_16 = tuple(np.ceil(i / 4).astype(int) for i in full_scene_size)

        if context_prior:
            self.CP_mega_voxels = CPMegaVoxels(
                self.feature_1_16,                
                size_1_16,
                n_relations=n_relations,
                bn_momentum=bn_momentum,
            )

    #
    def forward(self, input_dict):
        res = {}

        x3d_1_4 = input_dict["x3d"]
        x3d_1_8 = self.process_1_4(x3d_1_4)
        x3d_1_16 = self.process_1_8(x3d_1_8)

        if self.context_prior:
            ret = self.CP_mega_voxels(x3d_1_16)
            x3d_1_16 = ret["x"]
            for k in ret.keys():
                res[k] = ret[k]

        x3d_up_1_8 = self.up_1_16_1_8(x3d_1_16) + x3d_1_8
        x3d_up_1_4 = self.up_1_8_1_4(x3d_up_1_8) + x3d_1_4

        ssc_logit_1_4 = self.ssc_head_1_4(x3d_up_1_4)

        res["ssc_logit"] = ssc_logit_1_4

        return res
