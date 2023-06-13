import torch
import os
import glob
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import transforms
from kitti_ssc.dataset.utils.helpers import (
    vox2pix,
    compute_local_frustums,
    compute_CP_mega_matrix,
)


class KittiDataset(Dataset):
    def __init__(
        self,
        split,
        root,
        preprocess_root,
        project_scale=2,
        frustum_size=4,
        color_jitter=None,
        fliplr=0.0,
    ):
        super().__init__()
        self.root = root
        self.label_root = os.path.join(preprocess_root, "labels")
        self.n_classes = 20
        splits = {
            "train": ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],
            "val": ["08"],
            "test": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"],
            "trainval": ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"],
        }
        self.split = split
        self.sequences = splits[split]
        self.frustum_size = frustum_size
        self.project_scale = project_scale
        # self.output_scale = int(self.project_scale / 2)
        self.output_scale = 1
        self.scene_size = (51.2, 51.2, 6.4)
        self.vox_origin = np.array([0, -25.6, -2])
        self.fliplr = fliplr

        self.voxel_size = 0.2  # 0.2m
        self.img_W = 1220
        self.img_H = 370

        self.color_jitter = (
            transforms.ColorJitter(*color_jitter) if color_jitter else None
        )
        self.scans = []
        for sequence in self.sequences:
            calib = self.read_calib(
                os.path.join(self.root, "dataset", "sequences", sequence, "calib.txt")
            )
            P = calib["P2"]
            T_velo_2_cam = calib["Tr"]
            proj_matrix = P @ T_velo_2_cam

            cam_k = np.copy(P)
            cam_k[:, 3] = 0.
            my_proj_mat = cam_k @ T_velo_2_cam

            glob_path = os.path.join(
                self.root, "dataset", "sequences", sequence, "voxels", "*.bin"
            )
            for voxel_path in glob.glob(glob_path):
                self.scans.append(
                    {
                        "sequence": sequence,
                        "P": P,
                        "T_velo_2_cam": T_velo_2_cam,
                        "proj_matrix": proj_matrix,
                        "voxel_path": voxel_path,
                        "img_metas": {
                            'img_shape': [[self.img_H, self.img_W]],
                            'lidar2img': np.expand_dims(my_proj_mat, axis=0).tolist(),
                            'fliplr': False,
                        }
                    }
                )

        self.normalize_rgb = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __getitem__(self, index):
        scan = self.scans[index]
        voxel_path = scan["voxel_path"]
        sequence = scan["sequence"]
        P = scan["P"]
        T_velo_2_cam = scan["T_velo_2_cam"]
        proj_matrix = scan["proj_matrix"]
        img_metas = scan["img_metas"]

        filename = os.path.basename(voxel_path)
        frame_id = os.path.splitext(filename)[0]

        rgb_path = os.path.join(
            self.root, "dataset", "sequences", sequence, "image_2", frame_id + ".png"
        )

        data = {
            "frame_id": frame_id,
            "sequence": sequence,
            "P": P,
            "T_velo_2_cam": T_velo_2_cam,
            "proj_matrix": proj_matrix,
            "img_metas": img_metas
        }
        scale_3ds = [self.output_scale, self.project_scale]
        data["scale_3ds"] = scale_3ds
        cam_k = P[0:3, 0:3]
        data["cam_k"] = cam_k
        for scale_3d in scale_3ds:

            # compute the 3D-2D mapping
            projected_pix, fov_mask, pix_z = vox2pix(
                T_velo_2_cam,
                cam_k,
                self.vox_origin,
                self.voxel_size,
                scale_3d,
                self.img_W,
                self.img_H,
                self.scene_size,
            )            

            data["projected_pix_{}".format(scale_3d)] = projected_pix
            data["pix_z_{}".format(scale_3d)] = pix_z
            data["fov_mask_{}".format(scale_3d)] = fov_mask

        if self.split != 'test':
            target_1_path = os.path.join(self.label_root, sequence, frame_id + "_1_1.npy")
            target = np.load(target_1_path)
            data["target"] = target
            target_8_path = os.path.join(self.label_root, sequence, frame_id + "_1_8.npy")
            target_1_8 = np.load(target_8_path)
            CP_mega_matrix = compute_CP_mega_matrix(target_1_8)
            data["CP_mega_matrix"] = CP_mega_matrix

        # Compute the masks, each indicate the voxels of a local frustum
        if self.split != "test":
            projected_pix_output = data["projected_pix_{}".format(self.output_scale)]
            pix_z_output = data[
                "pix_z_{}".format(self.output_scale)
            ]
            frustums_masks, frustums_class_dists = compute_local_frustums(
                projected_pix_output,
                pix_z_output,
                target,
                self.img_W,
                self.img_H,
                dataset="kitti",
                n_classes=20,
                size=self.frustum_size,
            )
        else:
            frustums_masks = None
            frustums_class_dists = None
        data["frustums_masks"] = frustums_masks
        data["frustums_class_dists"] = frustums_class_dists

        img = Image.open(rgb_path).convert("RGB")

        # Image augmentation
        if self.color_jitter is not None:
            img = self.color_jitter(img)

        # PIL to numpy
        img = np.array(img, dtype=np.float32, copy=False) / 255.0
        img = img[:370, :1220, :]  # crop image

        # Fliplr the image
        if np.random.rand() < self.fliplr:
            data['img_metas']['fliplr'] = True
            img = np.ascontiguousarray(np.fliplr(img))
            for scale in scale_3ds:
                key = "projected_pix_" + str(scale)
                data[key][:, 0] = img.shape[1] - 1 - data[key][:, 0]

        data["img"] = self.normalize_rgb(img)
        return data

    def __len__(self):
        return len(self.scans)

    @staticmethod
    def read_calib(calib_path):
        """
        Modify from https://github.com/utiasSTARS/pykitti/blob/d3e1bb81676e831886726cc5ed79ce1f049aef2c/pykitti/utils.py#L68
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        calib_all = {}
        with open(calib_path, "r") as f:
            for line in f.readlines():
                if line == "\n":
                    break
                key, value = line.split(":", 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        # reshape matrices
        calib_out = {}
        # 3x4 projection matrix for left camera
        calib_out["P2"] = calib_all["P2"].reshape(3, 4)
        calib_out["Tr"] = np.identity(4)  # 4x4 matrix
        calib_out["Tr"][:3, :4] = calib_all["Tr"].reshape(3, 4)
        return calib_out
