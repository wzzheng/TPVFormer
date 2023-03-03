
import argparse, torch, os, json
import shutil
import numpy as np
import mmcv
from mmcv import Config
from collections import OrderedDict

from pyvirtualdisplay import Display
display = Display(visible=False, size=(2560, 1440))
display.start()

from mayavi import mlab
import mayavi
mlab.options.offscreen = True
print("Set mlab.options.offscreen={}".format(mlab.options.offscreen))


def revise_ckpt(state_dict):
    tmp_k = list(state_dict.keys())[0]
    if tmp_k.startswith('module.'):
        state_dict = OrderedDict(
            {k[7:]: v for k, v in state_dict.items()})
    return state_dict


def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """

    g_xx = np.arange(0, dims[0]) # [0, 1, ..., 256]
    # g_xx = g_xx[::-1]
    g_yy = np.arange(0, dims[1]) # [0, 1, ..., 256]
    # g_yy = g_yy[::-1]
    g_zz = np.arange(0, dims[2]) # [0, 1, ..., 32]

    # Obtaining the grid with coords...
    xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz)
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float32)
    resolution = np.array(resolution, dtype=np.float32).reshape([1, 3])

    coords_grid = (coords_grid * resolution) + resolution / 2

    return coords_grid


def draw(
    voxels,          # semantic occupancy predictions
    vox_origin,
    voxel_size=0.2,  # voxel size in the real world
    grid=None,       # voxel coordinates of point cloud
    pt_label=None,   # label of point cloud
    save_dirs=None,
    cam_positions=None,
    focal_positions=None,
    timestamp=None,
):
    w, h, z = voxels.shape
    grid = grid.astype(np.int)

    # Compute the voxels coordinates
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
    ) + np.array(vox_origin, dtype=np.float32).reshape([1, 3])
    grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T
    grid_coords[grid_coords[:, 3] == 17, 3] = 20
    
    # draw a simple car at the middle

    # car_vox_range = np.array([
    #     [w//2 - 2 - 4, w//2 - 2 + 4],
    #     [h//2 - 2 - 4, h//2 - 2 + 4],
    #     [z//2 - 2 - 3, z//2 - 2 + 3]
    # ], dtype=np.int)
    # car_x = np.arange(car_vox_range[0, 0], car_vox_range[0, 1])
    # car_y = np.arange(car_vox_range[1, 0], car_vox_range[1, 1])
    # car_z = np.arange(car_vox_range[2, 0], car_vox_range[2, 1])
    # car_xx, car_yy, car_zz = np.meshgrid(car_x, car_y, car_z)
    # car_label = np.zeros([8, 8, 6], dtype=np.int)
    # car_label[:3, :, :2] = 17
    # car_label[3:6, :, :2] = 18
    # car_label[6:, :, :2] = 19
    # car_label[:3, :, 2:4] = 18
    # car_label[3:6, :, 2:4] = 19
    # car_label[6:, :, 2:4] = 17
    # car_label[:3, :, 4:] = 19
    # car_label[3:6, :, 4:] = 17
    # car_label[6:, :, 4:] = 18
    # car_grid = np.array([car_xx.flatten(), car_yy.flatten(), car_zz.flatten()]).T
    # car_indexes = car_grid[:, 0] * h * z + car_grid[:, 1] * z + car_grid[:, 2]
    # grid_coords[car_indexes, 3] = car_label.flatten()

    # Get the voxels inside FOV
    fov_grid_coords = grid_coords

    # Remove empty and unknown voxels
    fov_voxels = fov_grid_coords[
        (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 20)
    ]
    print(len(fov_voxels))
    
    figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
    # Draw occupied inside FOV voxels
    voxel_size = sum(voxel_size) / 3
    plt_plot_fov = mlab.points3d(
        fov_voxels[:, 1],
        fov_voxels[:, 0],
        fov_voxels[:, 2],
        fov_voxels[:, 3],
        colormap="viridis",
        scale_factor=0.95 * voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=1,
        vmax=19, # 16
    )

    colors = np.array(
        [
            [255, 120,  50, 255],       # barrier              orange
            [255, 192, 203, 255],       # bicycle              pink
            [255, 255,   0, 255],       # bus                  yellow
            [  0, 150, 245, 255],       # car                  blue
            [  0, 255, 255, 255],       # construction_vehicle cyan
            [255, 127,   0, 255],       # motorcycle           dark orange
            [255,   0,   0, 255],       # pedestrian           red
            [255, 240, 150, 255],       # traffic_cone         light yellow
            [135,  60,   0, 255],       # trailer              brown
            [160,  32, 240, 255],       # truck                purple                
            [255,   0, 255, 255],       # driveable_surface    dark pink
            # [175,   0,  75, 255],       # other_flat           dark red
            [139, 137, 137, 255],
            [ 75,   0,  75, 255],       # sidewalk             dard purple
            [150, 240,  80, 255],       # terrain              light green          
            [230, 230, 250, 255],       # manmade              white
            [  0, 175,   0, 255],       # vegetation           green
            [  0, 255, 127, 255],       # ego car              dark cyan
            [255,  99,  71, 255],       # ego car
            [  0, 191, 255, 255]        # ego car
        ]
    ).astype(np.uint8)
    
    plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors

    scene = figure.scene

    for i, save_dir in enumerate(save_dirs):
        if i < 6:
            scene.camera.position = cam_positions[i] - np.array([0.7, 1.3, 0.])
            scene.camera.focal_point = focal_positions[i] - np.array([0.7, 1.3, 0.])
            scene.camera.view_angle = 35 if i != 3 else 60
            scene.camera.view_up = [0.0, 0.0, 1.0]
            scene.camera.clipping_range = [0.01, 300.]
            scene.camera.compute_view_plane_normal()
            scene.render()
        elif i == 6:
            # scene.camera.position = [-4.69302904, -52.74874688, 19.16181492]
            # scene.camera.focal_point = [-4.52985313, -51.8233303, 18.81979477]
            # scene.camera.view_angle = 40.0
            # scene.camera.view_up = [0.0, 0.0, 1.0]
            # scene.camera.clipping_range = [0.01, 300.]
            # scene.camera.compute_view_plane_normal()
            # scene.render()
            scene.camera.position = [  0.75131739, -35.08337438,  16.71378558]
            scene.camera.focal_point = [  0.75131739, -34.21734897,  16.21378558]
            scene.camera.view_angle = 40.0
            scene.camera.view_up = [0.0, 0.0, 1.0]
            scene.camera.clipping_range = [0.01, 300.]
            scene.camera.compute_view_plane_normal()
            scene.render()

        else:
            # scene.camera.position = [91.84365261779985, 87.2356528161641, 86.90232146965226]
            # scene.camera.focal_point = [4.607997894287109, -1.9073486328125e-06, -0.33333325386047363]
            # scene.camera.view_angle = 30.0
            # scene.camera.view_up = [0.0, 0.0, 1.0]
            # scene.camera.clipping_range = [33.458354318473965, 299.5433372220855]
            # scene.camera.compute_view_plane_normal()
            # scene.render()
            scene.camera.position = [ 0.75131739,  0.78265103, 93.21378558]
            scene.camera.focal_point = [ 0.75131739,  0.78265103, 92.21378558]
            scene.camera.view_angle = 40.0
            scene.camera.view_up = [0., 1., 0.]
            scene.camera.clipping_range = [0.01, 400.]
            scene.camera.compute_view_plane_normal()
            scene.render()

        mlab.savefig(os.path.join(save_dir, f'vis_{timestamp}.png'))
    mlab.close()


if __name__ == "__main__":
    import sys; sys.path.insert(0, os.path.abspath('.'))

    device = torch.device('cuda:0')
    # device = torch.device('cpu')
    ## prepare config
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/tpv04_occupancy.py')
    parser.add_argument('--work-dir', type=str, default='out/tpv_occupancy')
    parser.add_argument('--ckpt-path', type=str, default='out/tpv_occupancy/latest.pth')
    parser.add_argument('--vis-train', action='store_true', default=False)
    parser.add_argument('--save-path', type=str, default='out/tpv_occupancy/videos')
    parser.add_argument('--scene-name', type=str, default='', nargs='+', help='names of scenes to visualize')
    parser.add_argument('--scene-idx', type=int, default=0, nargs='+', help='idx of scenes to visualize')

    args = parser.parse_args()
    print(args)

    cfg = Config.fromfile(args.py_config)
    dataset_config = cfg.dataset_params

    # prepare model
    logger = mmcv.utils.get_logger('mmcv')
    logger.setLevel("WARNING")
    if cfg.get('occupancy', False):
        from builder import tpv_occupancy_builder as model_builder
    else:
        from builder import tpv_lidarseg_builder as model_builder
    my_model = model_builder.build(cfg.model).to(device)
    if args.ckpt_path:
        ckpt = torch.load(args.ckpt_path, map_location='cpu')
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        print(my_model.load_state_dict(revise_ckpt(ckpt)))
    my_model.eval()

    # prepare data
    from nuscenes import NuScenes
    from visualization.dataset import ImagePoint_NuScenes_vis, DatasetWrapper_NuScenes_vis

    if args.vis_train:
        pkl_path = 'data/nuscenes_infos_train_scene.pkl'
    else:
        pkl_path = 'data/nuscenes_infos_val_scene.pkl'
    
    data_path = 'data/nuscenes'
    label_mapping = dataset_config['label_mapping']

    nusc = NuScenes(version='v1.0-trainval', dataroot=data_path, verbose=True)
    
    if args.scene_name != '':
        scene_names = [args.scene_name] if not isinstance(args.scene_name, list) else args.scene_name
        num_scenes = len(scene_names)
        scene_idxs = [None] * num_scenes
    else:
        scene_idxs = [args.scene_idx] if not isinstance(args.scene_idx, list) else args.scene_idx
        num_scenes = len(scene_idxs)
        scene_names = [None] * num_scenes
    
    # render scenes one by one
    for idx in range(num_scenes):
        pt_dataset = ImagePoint_NuScenes_vis(
            data_path, imageset=pkl_path, scene_idx=scene_idxs[idx], scene_name=scene_names[idx],
            label_mapping=label_mapping, nusc=nusc)

        dataset = DatasetWrapper_NuScenes_vis(
            pt_dataset,
            grid_size=cfg.grid_size,
            fixed_volume_space=dataset_config['fixed_volume_space'],
            max_volume_space=dataset_config['max_volume_space'],
            min_volume_space=dataset_config['min_volume_space'],
            ignore_label=dataset_config["fill_label"],
            phase='val'
        )
        print(len(dataset))

        for index in range(len(dataset)):
            print(f'processing frame {index} of scene {idx}')
            batch_data, filelist, scene_meta, timestamp = dataset[index]
            imgs, img_metas, vox_label, grid, pt_label = batch_data
            imgs = torch.from_numpy(np.stack([imgs]).astype(np.float32)).to(device)
            grid = torch.from_numpy(np.stack([grid]).astype(np.float32))#.to(device)
            with torch.no_grad():
                outputs_vox = my_model(img=imgs, img_metas=[img_metas])
            
                predict_vox = torch.argmax(outputs_vox, dim=1) # bs, w, h, z
                predict_vox = predict_vox.squeeze(0).cpu().numpy() # w, h, z

            voxel_origin = dataset_config['min_volume_space']
            voxel_max = dataset_config['max_volume_space']
            grid_size = cfg.grid_size
            resolution = [(e - s) / l for e, s, l in zip(voxel_max, voxel_origin, grid_size)]

            scene_name = scene_meta['name']
            scene_dir = os.path.join(args.save_path, scene_name)
            os.makedirs(scene_dir, exist_ok=True)
            if not os.path.exists(os.path.join(scene_dir, 'meta.json')):
                with open(os.path.join(scene_dir, 'meta.json'), 'w') as fp:
                    json.dump(scene_meta, fp)
            
            clip_dirs = []
            for video_clip in range(8):
                clip_dir = os.path.join(scene_dir, str(video_clip))
                clip_dirs.append(clip_dir)
                os.makedirs(clip_dir, exist_ok=True)
                
                if video_clip < 6:
                    filename = filelist[video_clip]
                    shutil.copy(filename, os.path.join(clip_dir, 'img_'+str(timestamp)+'.jpg'))

            draw(predict_vox, 
                 voxel_origin, 
                 resolution, 
                 grid.squeeze(0).cpu().numpy(), 
                 pt_label.squeeze(-1),
                 clip_dirs,
                 img_metas['cam_positions'],
                 img_metas['focal_positions'],
                 timestamp=timestamp,)
