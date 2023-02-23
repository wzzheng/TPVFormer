import os, os.path as osp
import PIL.Image as Image
import cv2
import argparse

def cat_images(dir, cam_img_size, pred_img_size, pred_img_size2, spacing):
    numFrames = len(os.listdir(osp.join(dir, '0'))) // 2
    cam_imgs = []
    pred_imgs = []
    for i in range(8):
        clip_dir = osp.join(dir, str(i))
        if i < 6:
            cam_files = [osp.join(clip_dir, fn) for fn in os.listdir(clip_dir) if fn.startswith('img')]
            cam_files = sorted(cam_files)
            cam_imgs.append([
                Image.open(fn).resize(cam_img_size, Image.BILINEAR) for fn in cam_files
            ])
            pred_files = [osp.join(clip_dir, fn) for fn in os.listdir(clip_dir) if fn.startswith('vis')]
            pred_files = sorted(pred_files)
            pred_imgs.append([
                Image.open(fn).resize(cam_img_size, Image.BILINEAR) for fn in pred_files 
            ])
        else:
            pred_files = [osp.join(clip_dir, fn) for fn in os.listdir(clip_dir) if fn.startswith('vis')]
            pred_files = sorted(pred_files)
            if i == 6:
                pred_imgs.append([
                    Image.open(fn).resize(pred_img_size, Image.BILINEAR) for fn in pred_files 
                ])
            else:
                pred_imgs.append([
                    Image.open(fn).resize(pred_img_size, Image.BILINEAR).crop([460, 0, 1460, 1080]) for fn in pred_files 
                ])
    
    cam_w, cam_h = cam_img_size
    pred_w, pred_h = pred_img_size
    result_w = cam_w * 6 + 5 * spacing
    result_h = cam_h * 2 + pred_h + 2 * spacing
    
    results = []
    for i in range(numFrames):
        result = Image.new(pred_imgs[0][0].mode, (result_w, result_h), (0, 0, 0))
        result.paste(cam_imgs[0][i], box=(1*cam_w+1*spacing, 0))
        result.paste(cam_imgs[1][i], box=(2*cam_w+2*spacing, 0))
        result.paste(cam_imgs[2][i], box=(0, 0))
        result.paste(cam_imgs[3][i], box=(1*cam_w+1*spacing, 1*cam_h+1*spacing))
        result.paste(cam_imgs[4][i], box=(0, 1*cam_h+1*spacing))
        result.paste(cam_imgs[5][i], box=(2*cam_w+2*spacing, 1*cam_h+1*spacing))

        result.paste(pred_imgs[0][i], box=(4*cam_w+4*spacing, 0))
        result.paste(pred_imgs[1][i], box=(5*cam_w+5*spacing, 0))
        result.paste(pred_imgs[2][i], box=(3*cam_w+3*spacing, 0))
        result.paste(pred_imgs[3][i], box=(4*cam_w+4*spacing, 1*cam_h+1*spacing))
        result.paste(pred_imgs[4][i], box=(3*cam_w+3*spacing, 1*cam_h+1*spacing))
        result.paste(pred_imgs[5][i], box=(5*cam_w+5*spacing, 1*cam_h+1*spacing))

        result.paste(pred_imgs[6][i], box=(0, 2*cam_h+2*spacing))
        result.paste(pred_imgs[7][i], box=(1*pred_w+1*spacing, 2*cam_h+2*spacing))
        
        results.append(result)
    
    result_path = osp.join(dir, 'cat')
    os.makedirs(result_path, exist_ok=True)
    for i, result in enumerate(results):
        result.save(osp.join(result_path, f'{i}.png'))
    return results


def get_video(img_path, video_path, fps, size):
    video = cv2.VideoWriter(
        video_path, 
        cv2.VideoWriter_fourcc(*"MJPG"), 
        fps, 
        size
    )
    
    num_imgs = len(os.listdir(img_path))
    # img_files = [osp.join(img_path, fn) for fn in img_files]
    # img_files = sorted(img_files)
    for i in range(num_imgs):
        fn = osp.join(img_path, f'{i}.png')      
        img = cv2.imread(fn)
        video.write(img)
    
    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parse = argparse.ArgumentParser('')
    parse.add_argument('--scene-dir', type=str, default='out/tpv_occupancy/videos', 
                       help='directory of the scene outputs')
    parse.add_argument('--scene-name', type=str, default='scene-0916', nargs='+')
    args = parse.parse_args()

    scene_dir = args.scene_dir
    dirs = args.scene_name
    dirs = [osp.join(scene_dir, d) for d in dirs]
    cam_img_size = [480, 270]
    pred_img_size = [1920, 1080]
    pred_img_size2 = [1000, 1080]
    spacing = 10

    for dir in dirs:

        print(f'processing {os.path.basename(dir)}')

        cat_images(dir, cam_img_size, pred_img_size, pred_img_size2, spacing)

        get_video(
            osp.join(dir, 'cat'),
            osp.join(dir, 'video.avi'),
            12,
            [2930, 1640]
        )
    
    