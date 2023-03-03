import torch
from dataloader.dataset import ImagePoint_NuScenes
from dataloader.dataset_wrapper import custom_collate_fn, DatasetWrapper_NuScenes
from nuscenes import NuScenes


def build(dataset_config,
          train_dataloader_config,
          val_dataloader_config,
          grid_size=[200, 200, 16],
          version='v1.0-trainval',
          dist=False,
          scale_rate=1,
    ):
    data_path = train_dataloader_config["data_path"]
    train_imageset = train_dataloader_config["imageset"]
    val_imageset = val_dataloader_config["imageset"]
    label_mapping = dataset_config["label_mapping"]

    nusc = NuScenes(version=version, dataroot=data_path, verbose=True)
    train_dataset = ImagePoint_NuScenes(data_path, imageset=train_imageset,
                                     label_mapping=label_mapping, nusc=nusc)
    val_dataset = ImagePoint_NuScenes(data_path, imageset=val_imageset,
                                   label_mapping=label_mapping, nusc=nusc)

    train_dataset = DatasetWrapper_NuScenes(
        train_dataset,
        grid_size=grid_size,
        fixed_volume_space=dataset_config['fixed_volume_space'],
        max_volume_space=dataset_config['max_volume_space'],
        min_volume_space=dataset_config['min_volume_space'],
        fill_label=dataset_config["fill_label"],
        phase='train',
        scale_rate=scale_rate,
    )

    val_dataset = DatasetWrapper_NuScenes(
        val_dataset,
        grid_size=grid_size,
        fixed_volume_space=dataset_config['fixed_volume_space'],
        max_volume_space=dataset_config['max_volume_space'],
        min_volume_space=dataset_config['min_volume_space'],
        fill_label=dataset_config["fill_label"],
        phase='val',
        scale_rate=scale_rate,
    )

    if dist:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, drop_last=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=False)
    else:
        sampler = None
        val_sampler = None

    train_dataset_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       batch_size=train_dataloader_config["batch_size"],
                                                       collate_fn=custom_collate_fn,
                                                       shuffle=False if dist else train_dataloader_config["shuffle"],
                                                       sampler=sampler,
                                                       num_workers=train_dataloader_config["num_workers"])
    val_dataset_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                     batch_size=val_dataloader_config["batch_size"],
                                                     collate_fn=custom_collate_fn,
                                                     shuffle=False if dist else val_dataloader_config["shuffle"],
                                                     sampler=val_sampler,
                                                     num_workers=val_dataloader_config["num_workers"])

    return train_dataset_loader, val_dataset_loader
