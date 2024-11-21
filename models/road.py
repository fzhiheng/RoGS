import random

import torch
import numpy as np

try:
    import mayavi.mlab as mlab
    from utils.vis import plot_gaussion_3d
except ImportError:
    pass
import cv2
from pytorch3d.ops.knn import knn_points, knn_gather
from pytorch3d.transforms import matrix_to_quaternion
from diff_gaussian_rasterization.scene.cameras import OrthographicCamera


def create_hive_vertices(min_coords, max_coords, resolution=0.05):
    """
    Create a flat hive mesh.

    Args:
        x_length (float): Length along x of the mesh.
        y_length (float): Length along y of the mesh.
        poses (torch.Tensor): A tensor of shape (N, 3) containing the poses of the mesh.
        resolution (float): Resolution of the mesh. default: 1
    Returns:
        torch.Tensor: A tensor of shape ((num_vertices_x * num_vertices_y), 3) containing the vertices of the mesh.
        torch.Tensor: A tensor of shape ((num_vertices_x-1) * (num_vertices_y-1), 3) containing the faces of the mesh.
    """

    box = max_coords - min_coords
    x_length = box[0]
    y_length = box[1]

    x_resolution = resolution
    y_resolution = x_resolution * 2 / 1.7320508075688772
    num_vertices_x = int(x_length / x_resolution) + 1
    num_vertices_y = int(y_length / y_resolution) + 1
    assert num_vertices_x > 0 and num_vertices_y > 0, "Mesh resolution too high."
    vertices = torch.zeros((num_vertices_x, num_vertices_y, 3), dtype=torch.float32)
    vertices[:, :, 0] = torch.unsqueeze(torch.linspace(0, x_length, num_vertices_x), dim=0).T
    vertices[::2, :, 1] = torch.unsqueeze(torch.linspace(0, y_length + y_resolution / 2, num_vertices_y), dim=0)
    vertices[1::2, :, 1] = torch.unsqueeze(torch.linspace(-y_resolution / 2, y_length, num_vertices_y), dim=0)
    vertices = vertices.reshape(-1, 3)
    vertices += torch.tensor(min_coords, dtype=torch.float32)
    return vertices, (num_vertices_x, num_vertices_y), (x_resolution, y_resolution)


def create_rect_vertices(min_coords, max_coords, resolution=0.05):
    box = max_coords - min_coords
    x_length = box[0]
    y_length = box[1]

    num_vertices_x = int(x_length / resolution) + 1
    num_vertices_y = int(y_length / resolution) + 1
    assert num_vertices_x > 0 and num_vertices_y > 0, "Mesh resolution too high."
    vertices = torch.zeros((num_vertices_x, num_vertices_y, 3), dtype=torch.float32)
    vertices[:, :, 0] = torch.unsqueeze(torch.linspace(min_coords[0], max_coords[0], num_vertices_x), dim=0).T
    vertices[:, :, 1] = torch.unsqueeze(torch.linspace(min_coords[1], max_coords[1], num_vertices_y), dim=0)
    vertices = vertices.reshape(-1, 3)

    return vertices, (num_vertices_x, num_vertices_y), (resolution, resolution)

def cut_point_by_pose(vertices, vertices_shape, xy_resolution, min_coords, poses_xy, resolution=1.0, cut_range=3.0):
    """
    Cut point_plane using poses

    Args:
        vertices (torch.Tensor): A tensor of shape (N, 3) containing the vertices of the mesh.
        point_shape:
        min_coords:
        max_coords:
        poses_xy (torch.Tensor): A tensor of shape (N, 2) containing the poses in camera2world transform.
    """
    # pose_xy to pixel_xy
    x_resolution, y_resolution = xy_resolution
    pixel_xy = np.zeros_like(poses_xy)
    pixel_xy[:, 0] = (poses_xy[:, 0] - min_coords[0]) / x_resolution
    pixel_xy[:, 1] = (poses_xy[:, 1] - min_coords[1]) / y_resolution
    pixel_xy = np.unique(pixel_xy.round(), axis=0)

    # construct the mask
    pixel_xy[:, 0] = np.clip(pixel_xy[:, 0], 0, vertices_shape[0] - 1)
    pixel_xy[:, 1] = np.clip(pixel_xy[:, 1], 0, vertices_shape[1] - 1)
    pixel_xy = pixel_xy.astype(np.int32)

    mask = np.zeros(vertices_shape, dtype=np.uint8)
    mask[pixel_xy[:, 0], pixel_xy[:, 1]] = 1

    # dilate the mask
    kernel_size = int(cut_range / resolution)  # around cut_range meters
    kernel = np.ones((kernel_size, kernel_size), dtype=np.int32)
    mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2)  # (num_x, num_y)

    all_indices = np.arange(0, vertices_shape[0] * vertices_shape[1], 1, dtype=np.int64).reshape(vertices_shape[:2])  # (num_x, num_y)
    mask_index = all_indices[mask == 1]
    cut_vertices = vertices[mask_index]

    # ======>Get the four nearest neighbor points of each pixel
    maksed_indices = np.cumsum(mask.astype(np.int64)).reshape(vertices_shape[:2]) - 1  # (num_x, num_y)
    maksed_indices[mask == 0] = -1  # (num_x, num_y)
    maksed_indices = maksed_indices.flatten()  # (num_x * num_y,)

    left_indice = all_indices.copy() - 1  # (num_x, num_y)
    right_indice = all_indices.copy() + 1  # (num_x, num_y)
    up_indice = all_indices.copy() - vertices_shape[1]  # (num_x, num_y)
    down_indice = all_indices.copy() + vertices_shape[1]  # (num_x, num_y)

    left_indice[:, 0] += 1
    right_indice[:, -1] -= 1
    up_indice[0, :] += vertices_shape[1]
    down_indice[-1, :] -= vertices_shape[1]

    four_indices = np.stack([left_indice, right_indice, up_indice, down_indice], axis=-1).reshape(-1, 4)  # (num_x * num_y, 4)
    four_indices = maksed_indices[four_indices]  # (num_x * num_y, 4)
    mask_four_indices = four_indices[mask_index]  # (M, 4)
    mask_four_indices = np.where(mask_four_indices == -1, np.arange(0, mask_four_indices.shape[0], 1, dtype=np.int64).reshape(-1, 1), mask_four_indices)
    mask_four_indices = torch.from_numpy(mask_four_indices)

    return cut_vertices, mask_four_indices


def clean_nan(grad):
    grad = torch.nan_to_num_(grad)
    return grad


def inter_pose(pose, inter_length=1.0):
    xyzs = []
    rotations = []
    for i in range(pose.shape[0] - 1):
        start = pose[i][:3, 3]  # (3,)
        end = pose[i + 1][:3, 3]  # (3,)
        length = torch.norm(end - start)
        norm = (end - start) / (length + 1e-10)  # (3,)
        num = int(length / inter_length)
        if num == 0:
            xyzs.append(start[None])
            rotations.append(pose[i:i + 1, :3, :3])
            continue
        ll = torch.linspace(0, length, num, device=pose.device)
        delta = ll[:, None] * norm[None, :]  # (num, 3)
        pp = start[None, :] + delta  # (num, 3)
        rr = torch.stack([pose[i, :3, :3]] * pp.shape[0], dim=0)
        xyzs.append(pp)
        rotations.append(rr)
    xyz = torch.cat(xyzs, dim=0)  # (M, 3)
    rotation = torch.cat(rotations, dim=0)  # (M, 3, 3)
    return xyz, rotation


class Road(object):
    def __init__(self, config, dataset, device='cuda:0', vis=False):
        self.device = device
        self.resolution = config["bev_resolution"]
        self.cut_range = config["cut_range"]
        all_poses, num_classes = dataset.chassis2world_unique, dataset.num_class
        all_pose_xyz = all_poses[:, :3, 3]
        self.ref_pose = torch.from_numpy(dataset.ref_pose).float().to(device)
        min_coords = np.min(all_pose_xyz, axis=0) - self.cut_range
        max_coords = np.max(all_pose_xyz, axis=0) + self.cut_range
        self.min_z = np.min(all_pose_xyz[:, -1])
        self.max_z = np.max(all_pose_xyz[:, -1])
        self.min_xy = min_coords[:2]
        self.max_xy = max_coords[:2]

        box = max_coords - min_coords
        self.bev_x_length = box[0]
        self.bev_y_length = box[1]

        # (N, 3) (num_x, num_y)
        # vertices, self.bev_size_pixel, xy_resolution = create_hive_vertices(min_coords, max_coords, self.resolution)
        vertices, self.bev_size_pixel, xy_resolution = create_rect_vertices(min_coords, max_coords, self.resolution)
        print(f"Before cutting,  {vertices.shape[0]} vertices")

        # (M, 3)
        vertices, four_indices = cut_point_by_pose(vertices, self.bev_size_pixel, xy_resolution, min_coords, all_pose_xyz, self.resolution, self.cut_range)
        print(f"After cutting,  {vertices.shape[0]} vertices")
        self.four_indices = four_indices.to(device)

        vertices = vertices.to(device)

        # ====> int z and rotation
        traj_point = torch.from_numpy(all_pose_xyz).float().to(device)
        traj_rotation = torch.from_numpy(all_poses[:, :3, :3]).float().to(device)

        nearst_result = knn_points(vertices[None, :, :2], traj_point[None, :, :2], K=1)  # (1, M, K)
        near_points = knn_gather(traj_point[None], nearst_result.idx)[0]  # (M, K, 3)

        nearest_idx = nearst_result.idx[0, :, 0]
        init_z = torch.mean(near_points, dim=1)[:, 2]  # (M,)
        vertices[:, 2] = init_z
        rotation = traj_rotation[nearest_idx]  # (M, 3, 3)

        self.vertices = vertices  # (M, 3)
        self.rotation = matrix_to_quaternion(rotation)  # (M, 4)
        self.rgb = torch.zeros_like(self.vertices, device=device)
        self.label = torch.zeros((self.vertices.shape[0], num_classes), dtype=torch.float32, device=device)

        # creat one Orthographic Camera
        SLACK_Z = 1
        mid_xy = (min_coords[:2] + max_coords[:2]) / 2
        bevcam2world = np.array([[1, 0, 0, mid_xy[0]],
                                 [0, -1, 0, mid_xy[1]],
                                 [0, 0, -1, self.max_z + SLACK_Z],
                                 [0, 0, 0, 1]])
        bevcam2world = torch.from_numpy(bevcam2world).float()

        render_resolution = self.resolution
        width = int(box[0] / render_resolution) + 1
        height = int(box[1] / render_resolution) + 1
        self.bev_camera = OrthographicCamera(R=bevcam2world[:3, :3], T=bevcam2world[:3, 3], W=width, H=height, znear=0,
                                             zfar=bevcam2world[2, 3] - self.min_z + SLACK_Z,
                                             top=-self.bev_y_length * 0.5, bottom=self.bev_y_length * 0.5, right=self.bev_x_length * 0.5,
                                             left=-self.bev_x_length * 0.5, device=device)

        if vis:
            points = vertices.cpu().numpy()
            sample_idx = random.sample(range(points.shape[0]), 150)
            mask1 = np.logical_and(points[:, 0] > min_coords[0] + 10, points[:, 0] < min_coords[0] + 11)
            mask2 = np.logical_and(points[:, 1] > min_coords[1] + 10, points[:, 1] < min_coords[1] + 11)
            mask = np.logical_and(mask1, mask2)
            sample_idx = np.where(mask)[0]

            sample_points = points[sample_idx]  # (1000, 3)
            sample_rotation = rotation.cpu().numpy()[sample_idx]  # (1000, 3, 3)

            fig = mlab.figure(bgcolor=(1, 1, 1), size=(800, 800))
            for r, c in zip(sample_rotation, sample_points):
                S = np.array([[self.resolution * 0.56, 0, 0],
                              [0, self.resolution * 0.6, 0],
                              [0, 0, 0]])
                plot_gaussion_3d(figure=fig, R=r, center=c, S=S, num=50, color=(0.5, 0.5, 0.5, 1), plot_axis=False)

            # # ====> origin and xyz-axis
            # mlab.points3d(min_coords[0], min_coords[1], min_coords[2], scale_factor=1, color=(0, 0, 0), figure=fig)
            # mlab.quiver3d(min_coords[0], min_coords[1], min_coords[2], 10, 0, 0, scale_factor=1, color=(1, 0, 0), figure=fig)
            # mlab.quiver3d(min_coords[0], min_coords[1], min_coords[2], 0, 10, 0, scale_factor=1, color=(0, 1, 0), figure=fig)
            # mlab.quiver3d(min_coords[0], min_coords[1], min_coords[2], 0, 0, 10, scale_factor=1, color=(0, 0, 1), figure=fig)
            #
            # # ===> trajecotry
            # pose_points = all_pose_xyz
            # start = pose_points[:-1].cpu().numpy()
            # end = pose_points[1:].cpu().numpy()
            # arrow = end - start
            # mlab.quiver3d(start[:, 0], start[:, 1], start[:, 2], arrow[:, 0], arrow[:, 1], arrow[:, 2], scale_factor=1, color=(0, 0, 1), line_width=5,
            #               figure=fig)
            #
            # front_pose = dataset.camera2world_all[np.array(dataset.cameras_idx_all) == 0]
            # flow = front_pose[1:, :3, 3] - front_pose[:-1, :3, 3]
            # mlab.quiver3d(front_pose[:-1, 0, 3], front_pose[:-1, 1, 3], front_pose[:-1, 2, 3], flow[:, 0], flow[:, 1], flow[:, 2], scale_factor=1,
            #               color=(0, 0, 1), line_width=5, figure=fig)
            # mlab.quiver3d(front_pose[:, 0, 3], front_pose[:, 1, 3], front_pose[:, 2, 3], front_pose[:, 0, 0], front_pose[:, 1, 0], front_pose[:, 2, 0],
            #               scale_factor=1, color=(1, 0, 0), line_width=5, figure=fig)
            # mlab.quiver3d(front_pose[:, 0, 3], front_pose[:, 1, 3], front_pose[:, 2, 3], front_pose[:, 0, 1], front_pose[:, 1, 1], front_pose[:, 2, 1],
            #               scale_factor=1, color=(0, 1, 0), line_width=5, figure=fig)
            # mlab.quiver3d(front_pose[:, 0, 3], front_pose[:, 1, 3], front_pose[:, 2, 3], front_pose[:, 0, 2], front_pose[:, 1, 2], front_pose[:, 2, 2],
            #               scale_factor=1, color=(0, 0, 1), line_width=5, figure=fig)
            #
            # # ===> bev camera
            # mlab.points3d(mid_xy[0], mid_xy[1], 15, scale_factor=1, color=(0, 0, 0), figure=fig)
            # mlab.quiver3d(mid_xy[0], mid_xy[1], 15, 1, 0, 0, scale_factor=3, color=(1, 0, 0), line_width=5, figure=fig)
            # mlab.quiver3d(mid_xy[0], mid_xy[1], 15, 0, -1, 0, scale_factor=3, color=(0, 1, 0), line_width=5, figure=fig)
            # mlab.quiver3d(mid_xy[0], mid_xy[1], 15, 0, 0, -1, scale_factor=3, color=(0, 0, 1), line_width=5, figure=fig)

            mlab.show()
