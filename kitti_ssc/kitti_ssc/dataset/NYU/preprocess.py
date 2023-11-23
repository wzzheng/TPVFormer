import numpy as np
from tqdm import tqdm
import numpy.matlib
import os
import glob
import pickle
import hydra
from omegaconf import DictConfig


seg_class_map = [
    0,
    1,
    2,
    3,
    4,
    11,
    5,
    6,
    7,
    8,
    8,
    10,
    10,
    10,
    11,
    11,
    9,
    8,
    11,
    11,
    11,
    11,
    11,
    11,
    11,
    11,
    11,
    10,
    10,
    11,
    8,
    10,
    11,
    9,
    11,
    11,
    11,
]


def _rle2voxel(rle, voxel_size=(240, 144, 240), rle_filename=""):
    r"""Read voxel label data from file (RLE compression), and convert it to fully occupancy labeled voxels.
    code taken from https://github.com/waterljwant/SSC/blob/master/dataloaders/dataloader.py#L172
    In the data loader of pytorch, only single thread is allowed.
    For multi-threads version and more details, see 'readRLE.py'.
    output: seg_label: 3D numpy array, size 240 x 144 x 240
    """
    seg_label = np.zeros(
        int(voxel_size[0] * voxel_size[1] * voxel_size[2]), dtype=np.uint8
    )  # segmentation label
    vox_idx = 0
    for idx in range(int(rle.shape[0] / 2)):
        check_val = rle[idx * 2]
        check_iter = rle[idx * 2 + 1]
        if check_val >= 37 and check_val != 255:  # 37 classes to 12 classes
            print("RLE {} check_val: {}".format(rle_filename, check_val))
        seg_label_val = (
            seg_class_map[check_val] if check_val != 255 else 255
        )  # 37 classes to 12 classes
        seg_label[vox_idx : vox_idx + check_iter] = np.matlib.repmat(
            seg_label_val, 1, check_iter
        )
        vox_idx = vox_idx + check_iter
    seg_label = seg_label.reshape(voxel_size)  # 3D array, size 240 x 144 x 240
    return seg_label


def _read_rle(rle_filename):  # 0.0005s
    """Read RLE compression data
    code taken from https://github.com/waterljwant/SSC/blob/master/dataloaders/dataloader.py#L153
    Return:
        vox_origin,
        cam_pose,
        vox_rle, voxel label data from file
    Shape:
        vox_rle, (240, 144, 240)
    """
    fid = open(rle_filename, "rb")
    vox_origin = np.fromfile(
        fid, np.float32, 3
    ).T  # Read voxel origin in world coordinates
    cam_pose = np.fromfile(fid, np.float32, 16).reshape((4, 4))  # Read camera pose
    vox_rle = (
        np.fromfile(fid, np.uint32).reshape((-1, 1)).T
    )  # Read voxel label data from file
    vox_rle = np.squeeze(vox_rle)  # 2d array: (1 x N), to 1d array: (N , )
    fid.close()
    return vox_origin, cam_pose, vox_rle


def _downsample_label(label, voxel_size=(240, 144, 240), downscale=4):
    r"""downsample the labeled data,
    code taken from https://github.com/waterljwant/SSC/blob/master/dataloaders/dataloader.py#L262
    Shape:
        label, (240, 144, 240)
        label_downscale, if downsample==4, then (60, 36, 60)
    """
    if downscale == 1:
        return label
    ds = downscale
    small_size = (
        voxel_size[0] // ds,
        voxel_size[1] // ds,
        voxel_size[2] // ds,
    )  # small size
    label_downscale = np.zeros(small_size, dtype=np.uint8)
    empty_t = 0.95 * ds * ds * ds  # threshold
    s01 = small_size[0] * small_size[1]
    label_i = np.zeros((ds, ds, ds), dtype=np.int32)

    for i in range(small_size[0] * small_size[1] * small_size[2]):
        z = int(i / s01)
        y = int((i - z * s01) / small_size[0])
        x = int(i - z * s01 - y * small_size[0])

        label_i[:, :, :] = label[
            x * ds : (x + 1) * ds, y * ds : (y + 1) * ds, z * ds : (z + 1) * ds
        ]
        label_bin = label_i.flatten()

        zero_count_0 = np.array(np.where(label_bin == 0)).size
        zero_count_255 = np.array(np.where(label_bin == 255)).size

        zero_count = zero_count_0 + zero_count_255
        if zero_count > empty_t:
            label_downscale[x, y, z] = 0 if zero_count_0 > zero_count_255 else 255
        else:
            label_i_s = label_bin[
                np.where(np.logical_and(label_bin > 0, label_bin < 255))
            ]
            label_downscale[x, y, z] = np.argmax(np.bincount(label_i_s))
    return label_downscale


@hydra.main(config_name="../../config/kitti_ssc.yaml")
def main(config: DictConfig):
    scene_size = (240, 144, 240)
    for split in ["train", "test"]:
        root = os.path.join(config.NYU_root, "NYU" + split)
        base_dir = os.path.join(config.NYU_preprocess_root, "base", "NYU" + split)
        os.makedirs(base_dir, exist_ok=True)

        scans = glob.glob(os.path.join(root, "*.bin"))
        for scan in tqdm(scans):
            filename = os.path.basename(scan)
            name = filename[:-4]
            filepath = os.path.join(base_dir, name + ".pkl")
            if os.path.exists(filepath):
                continue

            vox_origin, cam_pose, rle = _read_rle(scan)

            target_1_1 = _rle2voxel(rle, scene_size, scan)
            target_1_4 = _downsample_label(target_1_1, scene_size, 4)
            target_1_16 = _downsample_label(target_1_1, scene_size, 16)

            data = {
                "cam_pose": cam_pose,
                "voxel_origin": vox_origin,
                "name": name,
                "target_1_4": target_1_4,
                "target_1_16": target_1_16,
            }

            with open(filepath, "wb") as handle:
                pickle.dump(data, handle)
                print("wrote to", filepath)


if __name__ == "__main__":
    main()
