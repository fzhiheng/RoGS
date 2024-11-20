import math

import torch
import numpy as np
from pytorch3d.transforms import quaternion_apply, quaternion_multiply
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from diff_gaussian_rasterization.scene.cameras import PerspectiveCamera, OrthographicCamera

from diff_gs_label import GaussianRasterizationSettings as GaussianRasterizationSettingsLabel
from diff_gs_label import GaussianRasterizer as GaussianRasterizerLabel

from models.gaussian_model import GaussianModel2D
from utils.sh_utils import eval_sh


def get_perspective_rect(cam2world, x_length=40, y_length=50):
    cam_xy1 = cam2world[:2, 3]
    cam_x_axis = cam2world[:2, 0]
    cam_x_norm = cam_x_axis / cam_x_axis.norm()
    point1 = cam_xy1 - cam_x_norm * x_length / 2
    point2 = cam_xy1 + cam_x_norm * x_length / 2

    cam_z_axis = cam2world[:2, 2]
    cam_z_norm = cam_z_axis / cam_z_axis.norm()
    point3 = point1 + cam_z_norm * y_length
    point4 = point2 + cam_z_norm * y_length

    points = torch.stack([point1, point2, point3, point4])  # 4x2
    min_xy = torch.min(points, dim=0).values
    max_xy = torch.max(points, dim=0).values
    return min_xy, max_xy


def get_orthographic_rect(camera: OrthographicCamera, dilate=1):
    points = torch.zeros(8, 3)
    points[0] = torch.tensor([camera.left, camera.bottom, camera.znear])
    points[1] = torch.tensor([camera.right, camera.bottom, camera.znear])
    points[2] = torch.tensor([camera.right, camera.top, camera.znear])
    points[3] = torch.tensor([camera.left, camera.top, camera.znear])
    points[4] = torch.tensor([camera.left, camera.bottom, camera.zfar])
    points[5] = torch.tensor([camera.right, camera.bottom, camera.zfar])
    points[6] = torch.tensor([camera.right, camera.top, camera.zfar])
    points[7] = torch.tensor([camera.left, camera.top, camera.zfar])

    points = points.to(camera.device) @ camera.cam2world[:3, :3].T + camera.cam2world[:3, 3:4].T

    min_xy = torch.min(points[:, :2], dim=0).values - dilate
    max_xy = torch.max(points[:, :2], dim=0).values + dilate
    return min_xy, max_xy


def render_blocks(viewpoint_camera, gaussians: GaussianModel2D, pipe, bg_color: torch.Tensor, delta_pose=None, render_type="rgb"):
    assert isinstance(viewpoint_camera, OrthographicCamera)
    bev_cams = viewpoint_camera.split(edge_pixel=2000)
    # print("bev cams nums", len(bev_cams) * len(bev_cams[0]))

    col_bev_images = []
    col_bev_depths = []
    col_bev_masks = []
    for w_cams in bev_cams:
        row_bev_images = []
        row_bev_depths = []
        row_bev_masks = []
        for cam in w_cams:
            bev_pkg = render(cam, gaussians, pipe, bg_color, delta_pose=delta_pose, render_type=render_type)
            row_bev_images.append(bev_pkg["render"].detach().cpu().numpy())
            row_bev_depths.append(bev_pkg["depth"].detach().cpu().numpy())
            row_bev_masks.append(bev_pkg["mask"].detach().cpu().numpy())
        col_bev_images.append(np.concatenate(row_bev_images, axis=-1))
        col_bev_depths.append(np.concatenate(row_bev_depths, axis=-1))
        col_bev_masks.append(np.concatenate(row_bev_masks, axis=-1))
    bev_image = np.concatenate(col_bev_images, axis=-2)
    bev_depth = np.concatenate(col_bev_depths, axis=-2)
    bev_mask = np.concatenate(col_bev_masks, axis=-2)

    return {"render": bev_image, "depth": bev_depth, "mask": bev_mask}


def render(viewpoint_camera, pc, pipe, bg_color, delta_pose=None, render_type="rgb"):
    """_summary_

    Args:
        render_type (str, optional): Defaults to "rgb". rgb or label

    """

    scaling_modifier = 1.0
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means

    if isinstance(viewpoint_camera, PerspectiveCamera):
        camera_type = 0
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    if isinstance(viewpoint_camera, OrthographicCamera):
        camera_type = 1
        tanfovx = 0
        tanfovy = 0

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        camera_type=camera_type,
        debug=pipe.debug,
    )

    means3D = pc.get_xyz

    if camera_type == 0:
        min_xy, max_xy = get_perspective_rect(viewpoint_camera.cam2world, x_length=40, y_length=50)
    else:
        min_xy, max_xy = get_orthographic_rect(viewpoint_camera)
    activate_mask1 = torch.bitwise_and(means3D[:, 0] >= min_xy[0], means3D[:, 0] <= max_xy[0])
    activate_mask2 = torch.bitwise_and(means3D[:, 1] >= min_xy[1], means3D[:, 1] <= max_xy[1])
    activate_mask = torch.bitwise_and(activate_mask1, activate_mask2)  # (N,)

    means3D = means3D[activate_mask]
    opacity = pc.get_opacity_mask(activate_mask)
    screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    means2D = screenspace_points

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance_mask(activate_mask, scaling_modifier)
    else:
        scales = pc.get_scaling_mask(activate_mask)
        rotations = pc.get_rotation_mask(activate_mask)

    if delta_pose is not None:
        delta_quat, delta_t = delta_pose
        means3D = quaternion_apply(delta_quat, means3D) + delta_t
        if rotations is not None:
            rotations = quaternion_multiply(delta_quat, rotations)

    if render_type == "rgb":
        if pc.use_rgb:
            colors_precomp = pc.get_rgb_mask(activate_mask)
        else:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            rgb = torch.clamp_min(sh2rgb + 0.5, 0.0)
            colors_precomp = rgb[activate_mask]

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        rendered_image, radii, depth, alpha = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=None,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )

    elif render_type == "label":
        colors_precomp = pc.get_label_mask(activate_mask)

        rasterizer = GaussianRasterizerLabel(raster_settings=raster_settings)
        if colors_precomp.shape[-1] == 5:
            from diff_gs_label2 import GaussianRasterizer as GaussianRasterizerLabel2

            rasterizer = GaussianRasterizerLabel2(raster_settings=raster_settings)
        rendered_image, radii, depth, alpha = rasterizer(
            means3D=means3D.detach(),
            means2D=means2D.detach(),
            shs=None,
            colors_precomp=colors_precomp,
            opacities=opacity.detach(),
            scales=scales.detach() if scales is not None else None,
            rotations=rotations.detach() if rotations is not None else None,
            cov3D_precomp=cov3D_precomp.detach() if cov3D_precomp is not None else None,
        )
    else:
        raise ValueError("Unknown render type")

    visibility_filter = torch.zeros(pc.get_xyz.shape[0], dtype=torch.bool, device="cuda")
    visibility_filter[activate_mask] = radii > 0
    valid_mask = depth[0] > 0

    return {
        "render": rendered_image,  # (C, H, W)
        "visibility_filter": visibility_filter,  # (N,)
        "depth": depth,  # (1, H, W)
        "mask": valid_mask,
    }  # (1, H, W)
