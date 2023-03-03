
dataset_params = dict(
    version = "v1.0-trainval",
    ignore_label = 0,
    fill_label = 0,
    fixed_volume_space = True,
    label_mapping = "./config/label_mapping/nuscenes.yaml",
    max_volume_space = [51.2, 51.2, 3],
    min_volume_space = [-51.2, -51.2, -5],
)

train_data_loader = dict(
    data_path = "data/nuscenes/",
    imageset = "./data/nuscenes_infos_train.pkl",
    batch_size = 1,
    shuffle = True,
    num_workers = 1,
)

val_data_loader = dict(
    data_path = "data/nuscenes/",
    imageset = "./data/nuscenes_infos_val.pkl",
    batch_size = 1,
    shuffle = False,
    num_workers = 1,
)

unique_label = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]