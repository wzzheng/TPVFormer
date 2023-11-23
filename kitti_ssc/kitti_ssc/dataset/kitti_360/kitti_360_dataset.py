import torch
import os
import glob
from torch.utils.data import Dataset
import numpy as np
from kitti_ssc.dataset.utils.helpers import vox2pix
from PIL import Image
from torchvision import transforms


class Kitti360Dataset(Dataset):
    def __init__(self, root, sequences, n_scans):
        """
        Paramters
        --------
        root: str
            Path to KITTI-360 dataset i.e. contain sequences such as 2013_05_28_drive_0009_sync
        sequence: str
            KITTI-360 sequence e.g. 2013_05_28_drive_0009_sync
        n_scans: int
            Only use the first n_scans since KITTI-360 sequence is very long
        """
        self.root = root
        self.img_H = 376
        self.img_W = 1408
        self.project_scale = 2
        self.output_scale = 1
        self.voxel_size = 0.2
        self.vox_origin = np.array([0, -25.6, -2])
        self.scene_size = (51.2, 51.2, 6.4)
        self.T_velo_2_cam = self.get_velo2cam()
        self.cam_k = self.get_cam_k()
        self.scans = []
        for sequence in sequences:
            glob_path = os.path.join(
                self.root, "data_2d_raw", sequence, "image_00/data_rect", "*.png"
            )
            for img_path in glob.glob(glob_path):
                self.scans.append({"img_path": img_path, "sequence": sequence})
        self.scans = self.scans[:n_scans]
        self.normalize_rgb = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.scans)

    def get_cam_k(self):
        cam_k = np.array(
            [
                552.554261,
                0.000000,
                682.049453,
                0.000000,
                0.000000,
                552.554261,
                238.769549,
                0.000000,
                0.000000,
                0.000000,
                1.000000,
                0.000000,
            ]
        ).reshape(3, 4)
        return cam_k[:3, :3]

    def get_velo2cam(self):
        cam2velo = np.array(
            [
                0.04307104361,
                -0.08829286498,
                0.995162929,
                0.8043914418,
                -0.999004371,
                0.007784614041,
                0.04392796942,
                0.2993489574,
                -0.01162548558,
                -0.9960641394,
                -0.08786966659,
                -0.1770225824,
            ]
        ).reshape(3, 4)
        cam2velo = np.concatenate(
            [cam2velo, np.array([0, 0, 0, 1]).reshape(1, 4)], axis=0
        )
        return np.linalg.inv(cam2velo)

    def __getitem__(self, index):
        data = {"T_velo_2_cam": self.T_velo_2_cam, "cam_k": self.cam_k}
        scan = self.scans[index]
        img_path = scan["img_path"]
        sequence = scan["sequence"]
        filename = os.path.basename(img_path)
        frame_id = os.path.splitext(filename)[0]
        data["frame_id"] = frame_id
        data["img_path"] = img_path
        data["sequence"] = sequence

        img = Image.open(img_path).convert("RGB")
        img = np.array(img, dtype=np.float32, copy=False) / 255.0
        img = self.normalize_rgb(img)
        data["img"] = img

        scale_3ds = [self.project_scale, self.output_scale]
        data["scale_3ds"] = scale_3ds
        
        for scale_3d in scale_3ds:
            projected_pix, fov_mask, _ = vox2pix(
                self.T_velo_2_cam,
                self.cam_k,
                self.vox_origin,
                self.voxel_size * scale_3d,
                self.img_W,
                self.img_H,
                self.scene_size,
            )
            data["projected_pix_{}".format(scale_3d)] = projected_pix
            data["fov_mask_{}".format(scale_3d)] = fov_mask
        return data
