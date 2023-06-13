from pytorch_lightning import Trainer
# from kitti_ssc.models.monoscene import MonoScene
from kitti_ssc.dataset.NYU.nyu_dm import NYUDataModule
from kitti_ssc.dataset.semantic_kitti.kitti_dm import KittiDataModule
from kitti_ssc.dataset.kitti_360.kitti_360_dm import Kitti360DataModule
import hydra
from omegaconf import DictConfig
import torch
import numpy as np
import os
from hydra.utils import get_original_cwd
from tqdm import tqdm
import pickle
from mmcv import Config
import yaml


@hydra.main(config_name="../config/kitti_ssc.yaml")
def main(config: DictConfig):
    torch.set_grad_enabled(False)

    model_cfg = Config.fromfile(config.model_cfg)

    # Setup dataloader
    if config.dataset == "kitti" or config.dataset == "kitti_360":
        project_scale = model_cfg.project_scale
        feature = model_cfg.feature
        full_scene_size = (256, 256, 32)

        if config.dataset == "kitti":
            data_module = KittiDataModule(
                root=config.kitti_root,
                preprocess_root=config.kitti_preprocess_root,
                frustum_size=config.frustum_size,
                batch_size=int(config.batch_size / config.n_gpus),
                num_workers=int(config.num_workers_per_gpu * config.n_gpus),
            )
            data_module.setup()
            # data_loader = data_module.val_dataloader()
            data_loader = data_module.test_dataloader() # use this if you want to infer on test set
        else:
            data_module = Kitti360DataModule(
                root=config.kitti_360_root,
                sequences=[config.kitti_360_sequence],
                n_scans=2000,
                batch_size=1,
                num_workers=3,
            )
            data_module.setup()
            data_loader = data_module.dataloader()

    elif config.dataset == "NYU":
        project_scale = 1
        feature = 200
        full_scene_size = (60, 36, 60)
        data_module = NYUDataModule(
            root=config.NYU_root,
            preprocess_root=config.NYU_preprocess_root,
            n_relations=config.n_relations,
            frustum_size=config.frustum_size,
            batch_size=int(config.batch_size / config.n_gpus),
            num_workers=int(config.num_workers_per_gpu * config.n_gpus),
        )
        data_module.setup()
        data_loader = data_module.val_dataloader()
        # data_loader = data_module.test_dataloader() # use this if you want to infer on test set
    else:
        print("dataset not support")

    # Load pretrained models
    if config.dataset == "NYU":
        model_path = os.path.join(
            get_original_cwd(), "trained_models", "monoscene_nyu.ckpt"
        )
    else:
        # model_path = os.path.join(
        #     get_original_cwd(), "trained_models", "monoscene_kitti.ckpt"
        # )
        model_path = config.model_path
    
    print('importing tpv10')
    from kitti_ssc.tpvformer10.kitti_ssc_tpv import TPV
   
    model = TPV.load_from_checkpoint(
        model_path,
        feature=feature,
        project_scale=project_scale,
        fp_loss=config.fp_loss,
        full_scene_size=full_scene_size,
    )
    model.cuda()
    model.eval()

    with open('kitti_ssc/data/semantic_kitti/semantic-kitti.yaml', 'r') as f:
        inv_learning_map = yaml.load(f.read())['learning_map_inv']

    # Save prediction and additional data 
    # to draw the viewing frustum and remove scene outside the room for NYUv2
    output_path = os.path.join(config.output_path, config.dataset)
    with torch.no_grad():
        for batch in tqdm(data_loader):
            batch["img"] = batch["img"].cuda()
            pred = model(batch)
            y_pred = torch.softmax(pred["ssc_logit"], dim=1).detach().cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1)
            y_pred = np.vectorize(inv_learning_map.__getitem__)(y_pred)
            for i in range(config.batch_size):
                out_dict = {"y_pred": y_pred[i].astype(np.uint16)}
                out_array = y_pred[i].astype(np.uint16)
                if "target" in batch:
                    out_dict["target"] = (
                        batch["target"][i].detach().cpu().numpy().astype(np.uint16)
                    )

                if config.dataset == "NYU":
                    write_path = output_path
                    filepath = os.path.join(write_path, batch["name"][i] + ".pkl")
                    out_dict["cam_pose"] = batch["cam_pose"][i].detach().cpu().numpy()
                    out_dict["vox_origin"] = (
                        batch["vox_origin"][i].detach().cpu().numpy()
                    )
                else:
                    # write_path = os.path.join(output_path, batch["sequence"][i])
                    # filepath = os.path.join(write_path, batch["frame_id"][i] + ".pkl")
                    # out_dict["fov_mask_1"] = (
                    #     batch["fov_mask_1"][i].detach().cpu().numpy()
                    # )
                    # out_dict["cam_k"] = batch["cam_k"][i].detach().cpu().numpy()
                    # out_dict["T_velo_2_cam"] = (
                    #     batch["T_velo_2_cam"][i].detach().cpu().numpy()
                    # )
                    write_path = os.path.join(output_path, 'sequences', batch["sequence"][i], 'predictions')
                    filepath = os.path.join(write_path, batch["frame_id"][i] + ".label")

                os.makedirs(write_path, exist_ok=True)
                # np.save(filepath, out_array)
                # print("wrote to", filepath)
                with open(filepath, "wb") as handle:
                    # pickle.dump(out_dict, handle)
                    handle.write(out_array)
                    print("wrote to", filepath)


if __name__ == "__main__":
    main()
