import math

import torch
from pytorch3d.transforms import quaternion_apply, quaternion_multiply
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from diff_gs_label import GaussianRasterizationSettings as GaussianRasterizationSettingsLabel
from diff_gs_label import GaussianRasterizer as GaussianRasterizerLabel
from diff_gaussian_rasterization.scene.cameras import PerspectiveCamera, OrthographicCamera

from models.gaussian_model import GaussianModel2D
from utils.sh_utils import eval_sh


def get_rect(cam2world, x_length=40, y_length=50):
    cam_xy1 = cam2world[:2, 3]
    cam_x_axis = cam2world[:2, 0]
    cam_x_norm = cam_x_axis / cam_x_axis.norm()
    point1 = cam_xy1 - cam_x_norm * x_length / 2 
    point2 = cam_xy1 + cam_x_norm * x_length / 2

    # cam z
    cam_z_axis = cam2world[:2, 2]
    cam_z_norm = cam_z_axis / cam_z_axis.norm()
    point3 = point1 + cam_z_norm * y_length
    point4 = point2 + cam_z_norm * y_length

    rect = torch.stack([point1, point2, point3, point4])  # 4x2
    min_xy = torch.min(rect, dim=0).values
    max_xy = torch.max(rect, dim=0).values
    return min_xy, max_xy


def render(viewpoint_camera, pc: GaussianModel2D, pipe, bg_color: torch.Tensor, delta_pose=None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    scaling_modifier = 1.0
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    
    fast_flag = True
    if isinstance(viewpoint_camera, PerspectiveCamera):
        camera_type = 0
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    if isinstance(viewpoint_camera, OrthographicCamera):
        camera_type = 1
        tanfovx = 0
        tanfovy = 0
        fast_flag = False


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
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    opacity = pc.get_opacity

    x_length = 40
    y_length = 50
    if fast_flag:
        min_xy, max_xy = get_rect(viewpoint_camera.cam2world, x_length, y_length)
        activate_mask1 = torch.bitwise_and(means3D[:, 0] > min_xy[0], means3D[:, 0] < max_xy[0])  
        activate_mask2 = torch.bitwise_and(means3D[:, 1] > min_xy[1], means3D[:, 1] < max_xy[1])
        activate_mask = torch.bitwise_and(activate_mask1, activate_mask2)  # (N,)
    else:
        activate_mask = torch.ones(means3D.shape[0], dtype=torch.bool, device="cuda")


    means3D = means3D[activate_mask]
    opacity = opacity[activate_mask]
    screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    means2D = screenspace_points
    

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
        cov3D_precomp = cov3D_precomp[activate_mask]
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
        scales = scales[activate_mask]
        rotations = rotations[activate_mask]

    if delta_pose is not None:
        delta_quat, delta_t = delta_pose
        means3D = quaternion_apply(delta_quat, means3D) + delta_t
        if rotations is not None:
            rotations = quaternion_multiply(delta_quat, rotations)

    shs = None
    colors_precomp = None
    if pc.use_rgb:
        colors_precomp = pc.get_rgb
        colors_precomp = colors_precomp[activate_mask]
    else:
        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
        dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        rgb = torch.clamp_min(sh2rgb + 0.5, 0.0)
        colors_precomp = rgb
        colors_precomp = colors_precomp[activate_mask]

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, depth, alpha = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp
    )

    visibility_filter = torch.zeros(pc.get_xyz.shape[0], dtype=torch.bool, device="cuda")
    visibility_filter[activate_mask] = radii > 0

    valid_mask = depth[0] > 0

    return {"render": rendered_image,
            "visibility_filter": visibility_filter,
            "depth": depth,
            "mask": valid_mask}


def render_label(viewpoint_camera, pc, pipe, bg_color, delta_pose=None):
    scaling_modifier = 1.0

    fast_flag = True
    if isinstance(viewpoint_camera, PerspectiveCamera):
        camera_type = 0
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    if isinstance(viewpoint_camera, OrthographicCamera):
        camera_type = 1
        tanfovx = 0
        tanfovy = 0
        fast_flag = False

    raster_settings = GaussianRasterizationSettingsLabel(
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
        just_color=True,
        debug=pipe.debug
    )

    means3D = pc.get_xyz
    opacity = pc.get_opacity

    x_length = 40
    y_length = 50

    if fast_flag:
        min_xy, max_xy = get_rect(viewpoint_camera.cam2world, x_length, y_length)
        activate_mask1 = torch.bitwise_and(means3D[:, 0] > min_xy[0], means3D[:, 0] < max_xy[0])  
        activate_mask2 = torch.bitwise_and(means3D[:, 1] > min_xy[1], means3D[:, 1] < max_xy[1])
        activate_mask = torch.bitwise_and(activate_mask1, activate_mask2)  # (N,)
    else:
        activate_mask = torch.ones(means3D.shape[0], dtype=torch.bool, device="cuda")

    means3D = means3D[activate_mask]
    opacity = opacity[activate_mask]
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
        cov3D_precomp = pc.get_covariance(scaling_modifier)
        cov3D_precomp = cov3D_precomp[activate_mask]
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
        scales = scales[activate_mask]
        rotations = rotations[activate_mask]
    colors_precomp = pc.get_label[activate_mask]

    if delta_pose is not None:
        delta_quat, delta_t = delta_pose
        means3D = quaternion_apply(delta_quat, means3D) + delta_t
        if rotations is not None:
            rotations = quaternion_multiply(delta_quat, rotations)

    rasterizer = GaussianRasterizerLabel(raster_settings=raster_settings)
    if colors_precomp.shape[-1] == 5:
        from diff_gs_label2 import GaussianRasterizer as GaussianRasterizerLabel2
        rasterizer = GaussianRasterizerLabel2(raster_settings=raster_settings)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, depth, alpha = rasterizer(
        means3D=means3D.detach(),
        means2D=means2D.detach(),
        shs=None,
        colors_precomp=colors_precomp,
        opacities=opacity.detach(),
        scales=scales.detach() if scales is not None else None,
        rotations=rotations.detach() if rotations is not None else None,
        cov3D_precomp=cov3D_precomp.detach() if cov3D_precomp is not None else None
    )

    valid_mask = depth[0] > 0

    return {"render": rendered_image,
            "mask": valid_mask}
