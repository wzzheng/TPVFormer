
import os
import numpy as np
from torch.utils import data
import yaml
import pickle
from mmcv.image.io import imread
from copy import deepcopy

import torch
import numba as nb
from torch.utils import data
from dataloader.transform_3d import PadMultiViewImage, \
    NormalizeMultiviewImage, \
    PhotoMetricDistortionMultiViewImage


img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

train_pipeline = [
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
]

test_pipeline = [
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
]


class ImagePoint_NuScenes_vis(data.Dataset):
    def __init__(self, data_path, imageset='train', 
                 scene_idx=None, scene_name=None,
                 label_mapping="nuscenes.yaml", nusc=None):
        self.return_ref = False

        with open(imageset, 'rb') as f:
            data = pickle.load(f)

        with open(label_mapping, 'r') as stream:
            nuscenesyaml = yaml.safe_load(stream)
        self.learning_map = nuscenesyaml['learning_map']

        nusc_infos = data['infos']

        # insert sweep frames between keyframes
        if scene_idx is not None or scene_name is not None:
            scene_name = list(nusc_infos.keys())[scene_idx] if scene_name is None else scene_name
            print(f'visualizing {scene_name}')
            self.nusc_infos = nusc_infos[scene_name]
            nusc_infos = deepcopy(self.nusc_infos)

            sweep_cams = []
            sweep_tss = []
            reverse_tab = {
                'CAM_FRONT':0, 
                'CAM_FRONT_RIGHT':1, 
                'CAM_FRONT_LEFT':2, 
                'CAM_BACK':3, 
                'CAM_BACK_LEFT':4, 
                'CAM_BACK_RIGHT':5
            }
            for cam_type in ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:
                dir = os.path.join(data_path, 'sweeps', cam_type)
                filenames = os.listdir(dir)
                files = [os.path.join(dir, fn) for fn in filenames]
                ts = [int(fn.split('__')[-1].split('.')[0]) for fn in filenames]
                idx = np.argsort(ts)
                sweep_cams.append(np.array(files)[idx])
                sweep_tss.append(np.array(ts)[idx])
            sweep_cams = np.array(sweep_cams)
            sweep_tss = np.array(sweep_tss)

            for i in range(len(self.nusc_infos)-1):
                insert_items = []
                start_ts = self.nusc_infos[i]['timestamp']
                end_ts = self.nusc_infos[i+1]['timestamp']
                temp_cams = []
                for sweep_cam, sweep_ts in zip(sweep_cams, sweep_tss):
                    temp_cam = sweep_cam[[(ts < end_ts and ts > start_ts) for ts in sweep_ts]]
                    temp_cams.append(temp_cam.tolist())
                min_len = min([len(temp_cam) for temp_cam in temp_cams])
                temp_cams = [temp_cam[:min_len] for temp_cam in temp_cams]
                for j in range(min_len):
                    temp_dict = deepcopy(self.nusc_infos[i])
                    for cam_type, cam_info in temp_dict['cams'].items():
                        cam_info['data_path'] = temp_cams[reverse_tab[cam_type]][j]
                    temp_dict['timestamp'] = temp_cams[0][j].split('__')[-1].split('.')[0]
                    insert_items.append(temp_dict)
                nusc_infos.extend(insert_items)
        
        self.nusc_infos = nusc_infos
        
        self.data_path = data_path
        self.lidarseg_path = data_path
        self.nusc = nusc
        self.cam_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.nusc_infos)

    def __getitem__(self, index):
        info = self.nusc_infos[index]
        imgs_info = self.get_data_info(info)
        img_metas = {
            'lidar2img': imgs_info['lidar2img'],
            'cam_positions': imgs_info['cam_positions'],
            'focal_positions': imgs_info['focal_positions']
        }
        # read 6 cams
        imgs = []
        for filename in imgs_info['img_filename']:
            imgs.append(
                imread(filename, 'unchanged').astype(np.float32)
            )
        
        lidar_sd_token = self.nusc.get('sample', info['token'])['data']['LIDAR_TOP']
        lidarseg_labels_filename = os.path.join(self.lidarseg_path, self.nusc.get('lidarseg', lidar_sd_token)['filename'])
        points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
        points_label = np.vectorize(self.learning_map.__getitem__)(points_label)
        
        lidar_path = info['lidar_path']
        points = np.fromfile(lidar_path, dtype=np.float32, count=-1).reshape([-1, 5])

        data_tuple = (imgs, img_metas, points[:, :3], points_label.astype(np.uint8))

        # deal with scene
        scene_token = self.nusc.get('sample', info['token'])['scene_token']
        scene_meta = self.nusc.get('scene', scene_token)
        timestamp = info['timestamp']
        return data_tuple, imgs_info['img_filename'], scene_meta, timestamp
    

    def get_data_info(self, info):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        # standard protocal modified from SECOND.Pytorch
        f = 0.0055
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
        )

        image_paths = []
        lidar2img_rts = []
        lidar2cam_rts = []
        cam_intrinsics = []
        cam_positions = []
        focal_positions = []
        for cam_type, cam_info in info['cams'].items():
            image_paths.append(cam_info['data_path'])
            # obtain lidar to image transformation matrix
            lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
            lidar2cam_t = cam_info[
                'sensor2lidar_translation'] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            intrinsic = cam_info['cam_intrinsic']
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = (viewpad @ lidar2cam_rt.T)
            lidar2img_rts.append(lidar2img_rt)

            cam_intrinsics.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)

            cam_position = np.linalg.inv(lidar2cam_rt.T) @ np.array([0., 0., 0., 1.]).reshape([4, 1])
            cam_positions.append(cam_position.flatten()[:3])
            focal_position = np.linalg.inv(lidar2cam_rt.T) @ np.array([0., 0., f, 1.]).reshape([4, 1])
            focal_positions.append(focal_position.flatten()[:3])

        input_dict.update(
            dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsics,
                lidar2cam=lidar2cam_rts,
                cam_positions=cam_positions, # w, h, z, meters,
                focal_positions=focal_positions
            ))

        return input_dict
    

class DatasetWrapper_NuScenes_vis(data.Dataset):
    def __init__(self, in_dataset, grid_size, ignore_label=0, fixed_volume_space=False, 
                 max_volume_space=[50, np.pi, 3], min_volume_space=[0, -np.pi, -5], phase='train'):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.grid_size = np.asarray(grid_size).astype(np.int32)
        self.ignore_label = ignore_label
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space
        self.polar = False

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
        return len(self.point_cloud_dataset)

    def __getitem__(self, index):
        data, filelist, scene_meta, timestamp = self.point_cloud_dataset[index]
        imgs, img_metas, xyz, labels = data
        
        # deal with img augmentations
        imgs_dict = {'img': imgs}
        for t in self.transforms:
            imgs_dict = t(imgs_dict)
        imgs = imgs_dict['img']
        imgs = [img.transpose(2, 0, 1) for img in imgs]
        img_metas['img_shape'] = imgs_dict['img_shape']

        xyz_pol = xyz
        
        assert self.fixed_volume_space
        max_bound = np.asarray(self.max_volume_space)  # 51.2 51.2 3
        min_bound = np.asarray(self.min_volume_space)  # -51.2 -51.2 -5
        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size                 # 200, 200, 16
        intervals = crop_range / (cur_grid_size - 1)

        if (intervals == 0).any(): print("Zero interval!")
        # TODO: grid_ind of float dtype may be better.
        grid_ind = (np.floor((np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)

        # process labels
        processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)
        data_tuple = (imgs, img_metas, processed_label, grid_ind, labels)

        return data_tuple, filelist, scene_meta, timestamp


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
