import torch, torch.nn as nn, torch.nn.functional as F
from mmcv.runner import BaseModule
from mmseg.models import HEADS


@HEADS.register_module()
class TPVFuser(BaseModule):
    def __init__(
        self, bev_h, bev_w, bev_z, nbr_classes=20, 
        in_dims=64, hidden_dims=128, out_dims=None,
        scale_h=2, scale_w=2, scale_z=2
    ):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_z = bev_z
        self.scale_h = scale_h
        self.scale_w = scale_w
        self.scale_z = scale_z

        out_dims = in_dims if out_dims is None else out_dims

        self.decoder = nn.Sequential(
            nn.Linear(in_dims, hidden_dims),
            nn.Softplus(),
            nn.Linear(hidden_dims, out_dims)
        )

        self.classifier = nn.Linear(out_dims, nbr_classes)
        self.classes = nbr_classes
    
    def forward(self, bev_list, points=None, use_checkpoint=True):
        """
        bev_list[0]: bs, h*w, c
        bev_list[1]: bs, z*h, c
        bev_list[2]: bs, w*z, c
        """
        bev_hw, bev_zh, bev_zw = bev_list[0], bev_list[1], bev_list[2]
        bs, _, c = bev_hw.shape
        bev_hw = bev_hw.permute(0, 2, 1).reshape(bs, c, self.bev_h, self.bev_w)
        bev_zh = bev_zh.permute(0, 2, 1).reshape(bs, c, self.bev_z, self.bev_h)
        bev_zw = bev_zw.permute(0, 2, 1).reshape(bs, c, self.bev_w, self.bev_z)

        # NVIDIA STYLE
        if self.scale_h != 1 or self.scale_w != 1:
            bev_hw = F.interpolate(
                bev_hw, 
                size=(self.bev_h*self.scale_h, self.bev_w*self.scale_w),
                mode='bilinear'
            )
        if self.scale_z != 1 or self.scale_h != 1:
            bev_zh = F.interpolate(
                bev_zh, 
                size=(self.bev_z*self.scale_z, self.bev_h*self.scale_h),
                mode='bilinear'
            )
        if self.scale_w != 1 or self.scale_z != 1:
            bev_zw = F.interpolate(
                bev_zw, 
                size=(self.bev_w*self.scale_w, self.bev_z*self.scale_z),
                mode='bilinear'
            )
        
        if points is not None:
            # points: bs, n, 3
            _, n, _ = points.shape
            points = points.reshape(bs, 1, n, 3)
            points[..., 0] = points[..., 0] / (self.bev_w*self.scale_w) * 2 - 1
            points[..., 1] = points[..., 1] / (self.bev_h*self.scale_h) * 2 - 1
            points[..., 2] = points[..., 2] / (self.bev_z*self.scale_z) * 2 - 1
            sample_loc = points[:, :, :, [0, 1]]
            bev_hw_pts = F.grid_sample(bev_hw, sample_loc).squeeze(2) # bs, c, n
            sample_loc = points[:, :, :, [1, 2]]
            bev_zh_pts = F.grid_sample(bev_zh, sample_loc).squeeze(2)
            sample_loc = points[:, :, :, [2, 0]]
            bev_zw_pts = F.grid_sample(bev_zw, sample_loc).squeeze(2)

            bev_hw_vox = bev_hw.unsqueeze(-1).permute(0, 1, 3, 2, 4).expand(-1, -1, -1, -1, self.scale_z*self.bev_z)
            bev_zh_vox = bev_zh.unsqueeze(-1).permute(0, 1, 4, 3, 2).expand(-1, -1, self.scale_w*self.bev_w, -1, -1)
            bev_zw_vox = bev_zw.unsqueeze(-1).permute(0, 1, 2, 4, 3).expand(-1, -1, -1, self.scale_h*self.bev_h, -1)
        
            fused_vox = (bev_hw_vox + bev_zh_vox + bev_zw_vox).flatten(2)
            fused_pts = bev_hw_pts + bev_zh_pts + bev_zw_pts
            fused = torch.cat([fused_vox, fused_pts], dim=-1) # bs, c, whz+n
            
            fused = fused.permute(0, 2, 1)
            if use_checkpoint:
                fused = torch.utils.checkpoint.checkpoint(self.decoder, fused)
                logits = torch.utils.checkpoint.checkpoint(self.classifier, fused)
            else:
                fused = self.decoder(fused)
                logits = self.classifier(fused)
            logits = logits.permute(0, 2, 1)
            logits_vox = logits[:, :, :(-n)].reshape(bs, self.classes, self.scale_w*self.bev_w, self.scale_h*self.bev_h, self.scale_z*self.bev_z)
            logits_pts = logits[:, :, (-n):].reshape(bs, self.classes, n, 1, 1)
            return logits_vox, logits_pts
            
        else:
            bev_hw = bev_hw.unsqueeze(-1).permute(0, 1, 3, 2, 4).expand(-1, -1, -1, -1, self.scale_z*self.bev_z)
            bev_zh = bev_zh.unsqueeze(-1).permute(0, 1, 4, 3, 2).expand(-1, -1, self.scale_w*self.bev_w, -1, -1)
            bev_zw = bev_zw.unsqueeze(-1).permute(0, 1, 2, 4, 3).expand(-1, -1, -1, self.scale_h*self.bev_h, -1)
        
            fused = bev_hw + bev_zh + bev_zw
            fused = fused.permute(0, 2, 3, 4, 1)
            if use_checkpoint:
                fused = torch.utils.checkpoint.checkpoint(self.decoder, fused)
                logits = torch.utils.checkpoint.checkpoint(self.classifier, fused)
            else:
                fused = self.decoder(fused)
                logits = self.classifier(fused)
            logits = logits.permute(0, 4, 1, 2, 3)
        
            return logits
