## How to generate the video demo?

Note that we also use the camera sweep data without annotation from nuScenes dataset in order to produce videos with higher fps.

### 1. prepare scene-level pkl files

Since every video clip corresponds to a scene in nuScenes dataset, we rearrange the pkl files according to scene information.

```
python visualize/dump_pkl.py --src-path data/nuscenes_infos_val.pkl --dst-path data/nuscenes_infos_val_scene.pkl --data-path data/nuscenes
```

### 2. generate individual video frames

```
python visualize/vis_scene.py --py-config config/tpv04_occupancy.py --work-dir out/tpv_occupancy --ckpt-path out/tpv_occupancy/latest.pth --save-path out/tpv_occupancy/videos --scene-name scene-0916 scene-0015
```

### 3. generate video flow from frames

```
python generate_videos.py --scene-dir out/tpv_occupancy/videos --scene-name scene-0916 scene-0015
```