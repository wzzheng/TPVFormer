"""
Code partly taken from https://github.com/cv-rits/LMSCNet/blob/main/LMSCNet/data/labels_downscale.py
"""
import numpy as np
from tqdm import tqdm
import numpy.matlib
import os
import glob
import hydra
from omegaconf import DictConfig
import kitti_ssc.dataset.semantic_kitti.io_data as SemanticKittiIO
from hydra.utils import get_original_cwd
from kitti_ssc.dataset.NYU.preprocess import _downsample_label


def majority_pooling(grid, k_size=2):
    result = np.zeros(
        (grid.shape[0] // k_size, grid.shape[1] // k_size, grid.shape[2] // k_size)
    )
    for xx in range(0, int(np.floor(grid.shape[0] / k_size))):
        for yy in range(0, int(np.floor(grid.shape[1] / k_size))):
            for zz in range(0, int(np.floor(grid.shape[2] / k_size))):

                sub_m = grid[
                    (xx * k_size) : (xx * k_size) + k_size,
                    (yy * k_size) : (yy * k_size) + k_size,
                    (zz * k_size) : (zz * k_size) + k_size,
                ]
                unique, counts = np.unique(sub_m, return_counts=True)
                if True in ((unique != 0) & (unique != 255)):
                    # Remove counts with 0 and 255
                    counts = counts[((unique != 0) & (unique != 255))]
                    unique = unique[((unique != 0) & (unique != 255))]
                else:
                    if True in (unique == 0):
                        counts = counts[(unique != 255)]
                        unique = unique[(unique != 255)]
                value = unique[np.argmax(counts)]
                result[xx, yy, zz] = value
    return result


@hydra.main(config_name="../../config/kitti_ssc.yaml")
def main(config: DictConfig):
    scene_size = (256, 256, 32)
    sequences = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
    remap_lut = SemanticKittiIO.get_remap_lut(
        os.path.join(
            get_original_cwd(),
            "kitti_ssc",
            "dataset",
            "semantic_kitti",
            "semantic-kitti.yaml",
        )
    )

    for sequence in sequences:
        sequence_path = os.path.join(
            config.kitti_root, "dataset", "sequences", sequence
        )
        label_paths = sorted(
            glob.glob(os.path.join(sequence_path, "voxels", "*.label"))
        )
        invalid_paths = sorted(
            glob.glob(os.path.join(sequence_path, "voxels", "*.invalid"))
        )
        out_dir = os.path.join(config.kitti_preprocess_root, "labels", sequence)
        os.makedirs(out_dir, exist_ok=True)

        downscaling = {"1_1": 1, "1_8": 8}

        for i in tqdm(range(len(label_paths))):

            frame_id, extension = os.path.splitext(os.path.basename(label_paths[i]))

            LABEL = SemanticKittiIO._read_label_SemKITTI(label_paths[i])
            INVALID = SemanticKittiIO._read_invalid_SemKITTI(invalid_paths[i])
            LABEL = remap_lut[LABEL.astype(np.uint16)].astype(
                np.float32
            )  # Remap 20 classes semanticKITTI SSC
            LABEL[
                np.isclose(INVALID, 1)
            ] = 255  # Setting to unknown all voxels marked on invalid mask...
            LABEL = LABEL.reshape([256, 256, 32])

            for scale in downscaling:
                filename = frame_id + "_" + scale + ".npy"
                label_filename = os.path.join(out_dir, filename)
                # If files have not been created...
                if not os.path.exists(label_filename):
                    if scale == "1_8":
                        LABEL_ds = _downsample_label(
                            LABEL, (256, 256, 32), downscaling[scale]
                        )
                    else:
                        LABEL_ds = LABEL
                    np.save(label_filename, LABEL_ds)
                    print("wrote to", label_filename)


if __name__ == "__main__":
    main()
