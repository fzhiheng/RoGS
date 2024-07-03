import os

import torch
import numpy as np
from torch import nn

from plyfile import PlyData, PlyElement
from pytorch3d.transforms import matrix_to_quaternion, quaternion_multiply

from utils.sh_utils import RGB2SH
from utils.general_utils import strip_symmetric, build_scaling_rotation, inverse_sigmoid, get_expon_lr_func, build_rotation

class GaussianModel2D(object):
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.label_activation = torch.nn.functional.softmax

        self.rotation_activation = torch.nn.functional.normalize

        # self.rgb_activation = lambda x: (torch.tanh(x) + 1) / 2
        # self.inverse_rgb_activation = lambda x: torch.atanh(2 * x - 1)
        self.rgb_activation = torch.sigmoid
        self.inverse_rgb_activation = inverse_sigmoid

    def __init__(self, cfg):
        self.ref_pose = torch.eye(4)
        self.active_sh_degree = 0
        self.max_sh_degree = cfg.sh_degree
        self.use_rgb = cfg.use_rgb
        self.opt_xy = cfg.opt_xy
        self._xy = torch.empty(0)
        self._z = torch.empty(0)
        self._label = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._rgb = torch.empty(0)
        self._scaling_xy = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def transform(self, transform):
        xyz = self.get_xyz
        xyz = torch.cat((xyz, torch.ones_like(xyz[:, :1])), dim=-1)  # (n,4)
        xyz = torch.matmul(transform, xyz.t()).t()  # (n,4)
        self._xy = xyz[:, :2]
        self._z = xyz[:, 2:3]

        self._rotation = quaternion_multiply(matrix_to_quaternion(transform[:3, :3]), self.get_rotation)

    def capture(self):
        return (
            self.ref_pose,
            self.active_sh_degree,
            self._xy,
            self._z,
            self._label,
            self._features_dc,
            self._features_rest,
            self._rgb,
            self._scaling_xy,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (self.ref_pose,
         self.active_sh_degree,
         self._xy,
         self._z,
         self._label,
         self._features_dc,
         self._features_rest,
         self._rgb,
         self._scaling_xy,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         xyz_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def _xyz(self):
        return torch.cat((self._xy, self._z), dim=-1)

    @property
    def _scaling(self):
        return self._scaling_xy
        # return torch.cat((self._scaling_xy, torch.zeros_like(self._scaling_xy[:, :1])), dim=-1)
        # return torch.cat((self._scaling_xy, torch.ones_like(self._scaling_xy[:, :1]) * 0.1), dim=-1)

    @property
    def get_scaling(self):
        scale_xy = self.scaling_activation(self._scaling_xy)
        return torch.cat((scale_xy, torch.zeros_like(self._scaling_xy[:, :1])), dim=-1)
        # return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_rgb(self):
        return self.rgb_activation(self._rgb)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_label(self):
        return self.label_activation(self._label, dim=-1)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def init_2d_gaussian(self, xyz: torch.Tensor, rotation: torch.Tensor, rgb: torch.Tensor, label: torch.Tensor, resolution, ref_pose, spatial_lr_scale: float):
        """

        :param xyz:  (n,3)
        :param rotation: wxyz (n,4)
        :param scale: (n,3)
        :param rgb: (n,3)
        :param label: (n,1)
        :param resolution: float
        :param ref_pose: (4,4)
        :param spatial_lr_scale: float
        :return:
        """
        self.ref_pose = ref_pose
        self.spatial_lr_scale = spatial_lr_scale
        num_points = xyz.shape[0]
        print("Number of points at initialisation : ", num_points)

        scale_s = 0.6
        rots = rotation
        scales_xy = torch.tensor([resolution * scale_s, resolution * scale_s], dtype=torch.float, device="cuda").repeat(num_points, 1)
        scales_xy = self.scaling_inverse_activation(scales_xy)
        opacities = inverse_sigmoid(1 * torch.ones((num_points, 1), dtype=torch.float, device="cuda"))

        self._xy = nn.Parameter(xyz[:, :2].contiguous().requires_grad_(True))
        self._z = nn.Parameter(xyz[:, 2:3].contiguous().requires_grad_(True))
        self._scaling_xy = nn.Parameter(scales_xy.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._label = nn.Parameter(label.requires_grad_(True))
        if self.use_rgb:
            self._rgb = nn.Parameter(rgb.requires_grad_(True))
        else:
            fused_color = RGB2SH(rgb)
            features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            features[:, :3, 0] = fused_color
            features[:, 3:, 1:] = 0.0
            self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
            self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._label], 'lr': training_args.label_lr, "name": "label"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling_xy], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._z], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "z"}
        ]
        if self.opt_xy:
            l.append({'params': [self._xy], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xy"})
        if self.use_rgb:
            l.append({'params': [self._rgb], 'lr': training_args.rgb_lr, "name": "rgb"})
        else:
            l.append({'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"})
            l.append({'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "z":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        if self.use_rgb:
            l.extend(['r', 'g', 'b'])
        else:
            for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
                l.append('f_dc_{}'.format(i))
            for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
                l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._label.shape[1]):
            l.append('label_{}'.format(i))
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = torch.cat((self._xy, self._z), dim=-1).detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling_xy.detach().cpu().numpy()
        label = self._label.detach().cpu().numpy()  # (n,n_class)
        rotation = self._rotation.detach().cpu().numpy()
        if self.use_rgb:
            rgb = self.get_rgb.detach().cpu().numpy()   # 方便直接在cloudcompare中查看
            attributes = np.concatenate((xyz, normals, rgb, opacities, label, scale, rotation), axis=1)
        else:
            f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, label, scale, rotation), axis=1)

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        if self.use_rgb:
            rgb = np.stack((np.asarray(plydata.elements[0]["r"]),
                            np.asarray(plydata.elements[0]["g"]),
                            np.asarray(plydata.elements[0]["b"])), axis=1)
            self._rgb = self.inverse_rgb_activation(torch.tensor(rgb, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
        else:

            features_dc = np.zeros((xyz.shape[0], 3, 1))
            features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
            features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
            features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

            extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
            extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
            assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
            features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
            self._features_dc = nn.Parameter(
                torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
            self._features_rest = nn.Parameter(
                torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))

        label_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("label_")]
        label_names = sorted(label_names, key=lambda x: int(x.split('_')[-1]))
        labels = np.zeros((xyz.shape[0], len(label_names)))
        for idx, attr_name in enumerate(label_names):
            labels[:, idx] = np.asarray(plydata.elements[0][attr_name])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xy = torch.tensor(xyz[:, :2], dtype=torch.float, device="cuda")
        self._z = nn.Parameter(torch.tensor(xyz[:, 2:3], dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
        self._label = nn.Parameter(torch.tensor(labels, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling_xy = nn.Parameter(torch.tensor(scales[:, :2], dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)
        self._z = optimizable_tensors["z"]
        if self.opt_xy:
            self._xy = optimizable_tensors["xy"]
        else:
            self._xy = self._xy[valid_points_mask]
        self._opacity = optimizable_tensors["opacity"]
        self._label = optimizable_tensors["label"]
        self._scaling_xy = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if self.use_rgb:
            self._rgb = optimizable_tensors["rgb"]
        else:
            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_label, new_scaling, new_rotation, new_rgb=None):
        d = {"z": new_xyz[:, 2:3],
             "opacity": new_opacities,
             "label": new_label,
             "scaling": new_scaling[:, :2],
             "rotation": new_rotation}
        if self.opt_xy:
            d["xy"] = new_xyz[:, :2]
        if self.use_rgb:
            d["rgb"] = new_rgb
        else:
            d["f_dc"] = new_features_dc
            d["f_rest"] = new_features_rest

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._z = optimizable_tensors["z"]
        if self.opt_xy:
            self._xy = optimizable_tensors["xy"]
        else:
            self._xy = torch.cat((self._xy, new_xyz[:, :2]), dim=0)
        if self.use_rgb:
            self._rgb = optimizable_tensors["rgb"]
        else:
            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._label = optimizable_tensors["label"]
        self._scaling_xy = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_label = self._label[selected_pts_mask].repeat(N, 1)

        new_rgb, new_features_dc, new_features_rest = None, None, None
        if self.use_rgb:
            new_rgb = self._rgb[selected_pts_mask].repeat(N, 1)
        else:
            new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
            new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_label, new_scaling, new_rotation, new_rgb)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_label = self._label[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_rgb, new_features_dc, new_features_rest = None, None, None
        if self.use_rgb:
            new_rgb = self._rgb[selected_pts_mask]
        else:
            new_features_dc = self._features_dc[selected_pts_mask]
            new_features_rest = self._features_rest[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_label, new_scaling, new_rotation, new_rgb)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
