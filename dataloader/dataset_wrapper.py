
import numpy as np
import torch
import numba as nb
from torch.utils import data
from dataloader.transform_3d import PadMultiViewImage, NormalizeMultiviewImage, \
    PhotoMetricDistortionMultiViewImage, RandomScaleImageMultiViewImage


img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)


class DatasetWrapper_NuScenes(data.Dataset):
    def __init__(self, in_dataset, grid_size, fill_label=0,
                 fixed_volume_space=False, max_volume_space=[51.2, 51.2, 3], 
                 min_volume_space=[-51.2, -51.2, -5], phase='train', scale_rate=1):
        'Initialization'
        self.imagepoint_dataset = in_dataset
        self.grid_size = np.asarray(grid_size)
        self.fill_label = fill_label
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space

        if scale_rate != 1:
            if phase == 'train':
                transforms = [
                    PhotoMetricDistortionMultiViewImage(),
                    NormalizeMultiviewImage(**img_norm_cfg),
                    RandomScaleImageMultiViewImage([scale_rate]),
                    PadMultiViewImage(size_divisor=32)
                ]
            else:
                transforms = [
                    NormalizeMultiviewImage(**img_norm_cfg),
                    RandomScaleImageMultiViewImage([scale_rate]),
                    PadMultiViewImage(size_divisor=32)
                ]
        else:
            if phase == 'train':
                transforms = [
                    PhotoMetricDistortionMultiViewImage(),
                    NormalizeMultiviewImage(**img_norm_cfg),
                    PadMultiViewImage(size_divisor=32)
                ]
            else:
                transforms = [
                    NormalizeMultiviewImage(**img_norm_cfg),
                    PadMultiViewImage(size_divisor=32)
                ]
        self.transforms = transforms

    def __len__(self):
        return len(self.imagepoint_dataset)

    def __getitem__(self, index):
        data = self.imagepoint_dataset[index]
        imgs, img_metas, xyz, labels = data

        # deal with img augmentations
        imgs_dict = {'img': imgs, 'lidar2img': img_metas['lidar2img']}
        for t in self.transforms:
            imgs_dict = t(imgs_dict)
        imgs = imgs_dict['img']
        imgs = [img.transpose(2, 0, 1) for img in imgs]

        img_metas['img_shape'] = imgs_dict['img_shape']
        img_metas['lidar2img'] = imgs_dict['lidar2img']

        assert self.fixed_volume_space
        max_bound = np.asarray(self.max_volume_space)  # 51.2 51.2 3
        min_bound = np.asarray(self.min_volume_space)  # -51.2 -51.2 -5
        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size                 # 200, 200, 16
        # TODO: intervals should not minus one.
        intervals = crop_range / (cur_grid_size - 1)   

        if (intervals == 0).any(): 
            print("Zero interval!")
        # TODO: grid_ind_float should actually be returned.
        # grid_ind_float = (np.clip(xyz, min_bound, max_bound - 1e-3) - min_bound) / intervals
        grid_ind_float = (np.clip(xyz, min_bound, max_bound) - min_bound) / intervals
        grid_ind = np.floor(grid_ind_float).astype(np.int)

        # process labels
        processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.fill_label
        label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)
        data_tuple = (imgs, img_metas, processed_label)

        data_tuple += (grid_ind, labels)

        return data_tuple


@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    return processed_label


def custom_collate_fn(data):
    img2stack = np.stack([d[0] for d in data]).astype(np.float32)
    meta2stack = [d[1] for d in data]
    label2stack = np.stack([d[2] for d in data]).astype(np.int)
    # because we use a batch size of 1, so we can stack these tensor together.
    grid_ind_stack = np.stack([d[3] for d in data]).astype(np.float)
    point_label = np.stack([d[4] for d in data]).astype(np.int)
    return torch.from_numpy(img2stack), \
        meta2stack, \
        torch.from_numpy(label2stack), \
        torch.from_numpy(grid_ind_stack), \
        torch.from_numpy(point_label)
