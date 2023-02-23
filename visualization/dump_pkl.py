import pickle
from nuscenes import NuScenes
import argparse


def arange_according_to_scene(infos, nusc):
    scenes = dict()

    for i, info in enumerate(infos):
        scene_token = nusc.get('sample', info['token'])['scene_token']
        scene_meta = nusc.get('scene', scene_token)
        scene_name = scene_meta['name']
        if not scene_name in scenes:
            scenes[scene_name] = [info]
        else:
            scenes[scene_name].append(info)
    
    return scenes


if __name__ == "__main__":

    parse = argparse.ArgumentParser('')
    parse.add_argument('--src-path', type=str, default='', help='path of the original pkl file')
    parse.add_argument('--dst-path', type=str, default='', help='path of the output pkl file')
    parse.add_argument('--data-path', type=str, default='', help='path of the nuScenes dataset')
    args = parse.parse_args()

    with open(args.src_path, 'rb') as f:
        data = pickle.load(f)
    nusc = NuScenes('v1.0-trainval', args.data_path)
    data['infos'] = arange_according_to_scene(data['infos'], nusc)
    
    with open(args.dst_path, 'wb') as f:
        pickle.dump(data, f)
    