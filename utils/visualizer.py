from typing import Tuple

import cv2
import torch
import numpy as np
from torch import nn
from pytorch3d.renderer.cameras import OrthographicCameras
from pytorch3d.renderer import PointsRasterizationSettings, PointsRasterizer

def mesh2height(mesh, bev_size_pixel):
    z_tensor = mesh._verts_list[0][:, 2]
    z_tensor = z_tensor.reshape(bev_size_pixel)
    return z_tensor


def loss2color(loss):
    min, max = loss.min(), loss.max()
    if (max - min) < 1e-7:
        loss = np.zeros_like(loss)
    else:
        # normalize depth by min max
        loss = (loss - min) / (max - min)
        loss.clip(0, 1)
    # convert to rgb
    loss = (loss * 255).astype(np.uint8)
    loss_rgb = cv2.applyColorMap(loss, cv2.COLORMAP_HOT)
    # BGR to RGB
    loss_rgb = cv2.cvtColor(loss_rgb, cv2.COLOR_BGR2RGB)
    return loss_rgb


def depth2color(depth, mask=None, min=None, max=None, rescale=False):
    # normalize depth by min max
    if mask is not None and mask.sum() == 0:
        return np.zeros_like(depth)
    mask_ = mask if mask is not None else np.ones_like(depth, dtype=bool)
    min_ = min if min else depth[mask_].min()
    max_ = max if max else depth[mask_].max()
    if min_ == max_:
        return np.zeros_like(depth)
    depth = (depth - min_) / (max_ - min_)
    depth = depth.clip(0, 1)
    if rescale:
        depth = np.sqrt(depth)
    # convert to rgb
    depth = (depth * 255).astype(np.uint8)
    # depth_rgb = cv2.applyColorMap(depth, cv2.COLORMAP_HOT)
    depth_rgb = cv2.applyColorMap(depth, cv2.COLORMAP_JET)  # BGR
    # BGR to RGB
    depth_rgb = cv2.cvtColor(depth_rgb, cv2.COLOR_BGR2RGB)
    if mask is not None:
        depth_rgb[~mask] = 0
    return depth_rgb


class CustomPointsRenderer(nn.Module):
    """
    1. Render without actiavte idx, all the points will be used to calculate loss
    2. all the points will be fed into rasterizer, the rendering speed is very slow
    """

    def __init__(self, rasterizer):
        super().__init__()
        self.rasterizer = rasterizer

    def forward(self, point_clouds, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        fragments = self.rasterizer(point_clouds, **kwargs)

        idx = fragments.idx.long()  # (N, H, W, K)
        features_packed = point_clouds.features_packed()

        image_shape = idx.shape[:3] + (features_packed.shape[-1],)
        images = torch.zeros(image_shape, dtype=features_packed.dtype, device=idx.device)
        forground_mask = idx[..., 0] >= 0  # (N, H, W)
        forground_idx = idx[forground_mask]  # (forground_num, K)
        images[forground_mask] = features_packed[forground_idx[:, 0], :]
        alpha = forground_mask.type_as(images)[..., None]  # (N, H, W, 1)

        depth = fragments.zbuf
        depth[~forground_mask] = 0

        feature = torch.cat([images, alpha], dim=-1)  # (N, H, W, C+1)

        return feature, depth


class CustomPointVisualizer(nn.Module):
    def __init__(self, device, min_xy, max_xy, cam_height, wh=None, resolution=None):
        super().__init__()
        self.device = device
        length_x = max_xy[0] - min_xy[0]
        length_y = max_xy[1] - min_xy[1]
        if wh is not None:
            width, height = wh
        elif resolution is not None:
            width = int(length_x / resolution) + 1
            height = int(length_y / resolution) + 1
        else:
            raise ValueError("wh or resolution must be provided")
        image_size = (height, width)
        rotation = torch.from_numpy(np.asarray([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, -1],
        ], dtype=np.float32))
        cx = length_x / 2
        cy = length_y / 2
        focal_length = 1 / min(cx, cy)
        bev_cam_xy = (min_xy + max_xy) / 2
        bev_cam_xyz = torch.tensor([bev_cam_xy[0], bev_cam_xy[1], cam_height], dtype=torch.float32)
        translation = -rotation @ bev_cam_xyz[:, None]
        translation = translation.reshape(1, 3)
        camera = OrthographicCameras(
            focal_length=focal_length,
            R=rotation[None],
            T=translation,
            device=device
        )
        raster_settings = PointsRasterizationSettings(
            image_size=image_size,
            radius=0.003,  # 0.003 0.03 0.01
            points_per_pixel=1,
        )
        self.point_renderer = CustomPointsRenderer(
            rasterizer=PointsRasterizer(
                cameras=camera,
                raster_settings=raster_settings
            )
        )

    def forward(self, point_clouds):
        feature, depth = self.point_renderer(point_clouds)
        return feature, depth
