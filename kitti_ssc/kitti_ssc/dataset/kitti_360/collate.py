import torch


def collate_fn(batch):
    data = {}
    imgs = []
    frame_ids = []
    img_paths = []
    sequences = []

    cam_ks = []
    T_velo_2_cams = []

    scale_3ds = batch[0]["scale_3ds"]
    for scale_3d in scale_3ds:
        data["projected_pix_{}".format(scale_3d)] = []
        data["fov_mask_{}".format(scale_3d)] = []

    for _, input_dict in enumerate(batch):
        if "img_path" in input_dict:
            img_paths.append(input_dict["img_path"])

        for key in data:
            data[key].append(torch.from_numpy(input_dict[key]))

        cam_ks.append(torch.from_numpy(input_dict["cam_k"]).float())
        T_velo_2_cams.append(torch.from_numpy(input_dict["T_velo_2_cam"]).float())

        sequences.append(input_dict["sequence"])

        img = input_dict["img"]
        imgs.append(img)

        frame_ids.append(input_dict["frame_id"])

    ret_data = {
        "sequence": sequences,
        "frame_id": frame_ids,
        "cam_k": cam_ks,
        "T_velo_2_cam": T_velo_2_cams,
        "img": torch.stack(imgs),
        "img_path": img_paths,
    }
    for key in data:
        ret_data[key] = data[key]

    return ret_data
