# from operator import gt
import pickle
import numpy as np
from omegaconf import DictConfig
import hydra
from mayavi import mlab


def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """

    g_xx = np.arange(0, dims[0] + 1)
    g_yy = np.arange(0, dims[1] + 1)
    sensor_pose = 10
    g_zz = np.arange(0, dims[2] + 1)

    # Obtaining the grid with coords...
    xx, yy, zz = np.meshgrid(g_xx[:-1], g_yy[:-1], g_zz[:-1])
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float)

    coords_grid = (coords_grid * resolution) + resolution / 2

    temp = np.copy(coords_grid)
    temp[:, 0] = coords_grid[:, 1]
    temp[:, 1] = coords_grid[:, 0]
    coords_grid = np.copy(temp)

    return coords_grid


def draw(
    voxels,
    T_velo_2_cam,
    vox_origin,
    fov_mask,
    img_size,
    f,
    voxel_size=0.2,
    d=7,  # 7m - determine the size of the mesh representing the camera
):
    # Compute the coordinates of the mesh representing camera
    x = d * img_size[0] / (2 * f)
    y = d * img_size[1] / (2 * f)
    tri_points = np.array(
        [
            [0, 0, 0],
            [x, y, d],
            [-x, y, d],
            [-x, -y, d],
            [x, -y, d],
        ]
    )
    tri_points = np.hstack([tri_points, np.ones((5, 1))])
    tri_points = (np.linalg.inv(T_velo_2_cam) @ tri_points.T).T
    x = tri_points[:, 0] - vox_origin[0]
    y = tri_points[:, 1] - vox_origin[1]
    z = tri_points[:, 2] - vox_origin[2]
    triangles = [
        (0, 1, 2),
        (0, 1, 4),
        (0, 3, 4),
        (0, 2, 3),
    ]

    # Compute the voxels coordinates
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
    )

    # Attach the predicted class to every voxel
    grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T

    # Get the voxels inside FOV
    fov_grid_coords = grid_coords[fov_mask, :]

    # Get the voxels outside FOV
    outfov_grid_coords = grid_coords[~fov_mask, :]

    # Remove empty and unknown voxels
    fov_voxels = fov_grid_coords[
        (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 255)
    ]
    outfov_voxels = outfov_grid_coords[
        (outfov_grid_coords[:, 3] > 0) & (outfov_grid_coords[:, 3] < 255)
    ]

    figure = mlab.figure(size=(1400, 1400), bgcolor=(1, 1, 1))

    # Draw the camera
    mlab.triangular_mesh(
        x, y, z, triangles, representation="wireframe", color=(0, 0, 0), line_width=5
    )

    # Draw occupied inside FOV voxels
    plt_plot_fov = mlab.points3d(
        fov_voxels[:, 0],
        fov_voxels[:, 1],
        fov_voxels[:, 2],
        fov_voxels[:, 3],
        colormap="viridis",
        scale_factor=voxel_size - 0.05 * voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=1,
        vmax=19,
    )

    # Draw occupied outside FOV voxels
    plt_plot_outfov = mlab.points3d(
        outfov_voxels[:, 0],
        outfov_voxels[:, 1],
        outfov_voxels[:, 2],
        outfov_voxels[:, 3],
        colormap="viridis",
        scale_factor=voxel_size - 0.05 * voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=1,
        vmax=19,
    )

    colors = np.array(
        [
            [100, 150, 245, 255],
            [100, 230, 245, 255],
            [30, 60, 150, 255],
            [80, 30, 180, 255],
            [100, 80, 250, 255],
            [255, 30, 30, 255],
            [255, 40, 200, 255],
            [150, 30, 90, 255],
            [255, 0, 255, 255],
            [255, 150, 255, 255],
            [75, 0, 75, 255],
            [175, 0, 75, 255],
            [255, 200, 0, 255],
            [255, 120, 50, 255],
            [0, 175, 0, 255],
            [135, 60, 0, 255],
            [150, 240, 80, 255],
            [255, 240, 150, 255],
            [255, 0, 0, 255],
        ]
    ).astype(np.uint8)

    plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    plt_plot_outfov.glyph.scale_mode = "scale_by_vector"

    plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors

    outfov_colors = colors
    outfov_colors[:, :3] = outfov_colors[:, :3] // 3 * 2
    plt_plot_outfov.module_manager.scalar_lut_manager.lut.table = outfov_colors

    mlab.show()


@hydra.main(config_path=None)
def main(config: DictConfig):
    scan = config.file
    with open(scan, "rb") as handle:
        b = pickle.load(handle)

    fov_mask_1 = b["fov_mask_1"]
    T_velo_2_cam = b["T_velo_2_cam"]
    vox_origin = np.array([0, -25.6, -2])

    y_pred = b["y_pred"]

    if config.dataset == "kitti_360":
        # Visualize KITTI-360
        draw(
            y_pred,
            T_velo_2_cam,
            vox_origin,
            fov_mask_1,
            voxel_size=0.2,
            f=552.55426,
            img_size=(1408, 376),
            d=7,
        )
    else:
        # Visualize Semantic KITTI
        draw(
            y_pred,
            T_velo_2_cam,
            vox_origin,
            fov_mask_1,
            img_size=(1220, 370),
            f=707.0912,
            voxel_size=0.2,
            d=7,
        )


if __name__ == "__main__":
    main()
