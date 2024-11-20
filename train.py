import os
import datetime
import random
import argparse

GUI_FLAG = False
try:
    import mayavi.mlab as mlab
    from utils.vis import vis_gaussian_3d

    GUI_FLAG = True
except ImportError:
    pass
import cv2
import yaml
import pytz
import torch
import shutil
import addict
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

from pytorch3d.ops import knn_points, knn_gather
from pytorch3d.structures import Pointclouds
from torch.utils.data import DataLoader
from diff_gaussian_rasterization.scene.cameras import PerspectiveCamera

from utils.logging import create_logger
from utils.image import render_semantic
from utils.render import render, render_label
from utils.visualizer import loss2color, depth2color, CustomPointVisualizer
from models.road import Road
from models.loss import L1MaskedLoss, CELossWithMask
from models.exposure_model import ExposureModel
from models.gaussian_model import GaussianModel2D
from eval import eval_bev_metric, eval_z_metric


def set_randomness(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_configs():
    parser = argparse.ArgumentParser(description='G4M config')
    parser.add_argument('--config', default="configs/local_nusc_mini.yaml", help='config yaml path')
    args = parser.parse_args()
    with open(args.config) as file:
        configs = yaml.safe_load(file)
    configs["file"] = os.path.abspath(args.config)
    return configs


def gt_render(dataset, min_xy, max_xy, bev_cam_height, wh=None, resolution=None, save_root=None, device="cuda:0"):
    gt_bev_visualizer = CustomPointVisualizer(device, min_xy, max_xy, bev_cam_height, wh=wh, resolution=resolution)
    road_pointcloud = dataset.road_pointcloud
    for key, value in road_pointcloud.items():
        road_pointcloud[key] = torch.from_numpy(value.astype(np.float32)).to(device)
    features = torch.cat((road_pointcloud["rgb"], road_pointcloud["label"]), dim=1)
    pointclouds = Pointclouds(points=[road_pointcloud["xyz"]], features=[features])
    pointclouds.extend(1)

    point_feature, depth = gt_bev_visualizer(pointclouds)
    point_feature = point_feature[0].detach().cpu().numpy()

    point_rgb = point_feature[..., :3]  # (H,W,3)
    bev_label = dataset.remap_semantic(point_feature[..., -2])  # (H,W)
    bev_mask = point_feature[..., -1] > 0  # (H,W)
    bev_height = bev_cam_height - depth[0, :, :, 0]  # (H,W)
    bev_height = bev_height.detach().cpu().numpy()  # (H,W)

    os.makedirs(save_root, exist_ok=True)

    cv2.imwrite(os.path.join(save_root, "bev_mask.png"), bev_mask.astype(np.uint8) * 255)

    bev_image = cv2.cvtColor((point_rgb * 225).astype(np.uint8), cv2.COLOR_RGB2BGRA)
    bev_image[~bev_mask] = 0
    cv2.imwrite(os.path.join(save_root, "bev_image.png"), bev_image)

    bev_label_vis = render_semantic(bev_label, dataset.filted_color_map)
    bev_label_vis = cv2.cvtColor(bev_label_vis, cv2.COLOR_RGB2BGRA)
    bev_label_vis[~bev_mask] = 0
    cv2.imwrite(os.path.join(save_root, "bev_label.png"), bev_label)
    cv2.imwrite(os.path.join(save_root, "bev_label_vis.png"), bev_label_vis)

    if bev_mask.sum() < 0.3 * bev_mask.size:
        bev_height[~bev_mask] = 0
        sp_matrix = sp.csr_matrix(bev_height)
        np.savez(os.path.join(save_root, f"bev_height.npz"), data=sp_matrix.data, indices=sp_matrix.indices, indptr=sp_matrix.indptr,
                 shape=sp_matrix.shape)
    else:
        np.save(os.path.join(save_root, f"bev_height.npy"), bev_height)
    bev_height_vis = cv2.cvtColor(depth2color(bev_height, bev_mask), cv2.COLOR_RGB2BGRA)
    bev_height_vis[~bev_mask] = 0
    cv2.imwrite(os.path.join(save_root, "bev_height_vis.png"), bev_height_vis)


def train(configs):
    dataset_cfg = configs.dataset
    model_cfg = configs.model
    pipe = configs.pipeline
    opt = configs.optimization
    train_cfg = configs.train

    set_randomness(configs.seed)
    torch.autograd.set_detect_anomaly(False)

    tz = pytz.timezone("Asia/Shanghai")
    time_show = datetime.datetime.now(tz).strftime("%m-%d-%H-%M")
    scene_name = dataset_cfg.clip_list[0] if "clip_list" in dataset_cfg else "seq" + dataset_cfg.sequence
    output_root = os.path.join(
        configs.output,
        f'{scene_name}-{"z" if opt.z_weight > 0 else "no_z"}-{configs.model.cut_range}m-{time_show}',
    )
    os.makedirs(output_root, exist_ok=True)
    img_root = os.path.join(output_root, "images")
    ply_root = os.path.join(output_root, "ply")
    shutil.copy(configs.file, output_root)

    logger = create_logger(f"RoGS-{scene_name}", os.path.join(output_root, "train.log"))
    logger.info(f"Output root: {os.path.abspath(output_root)}")

    device = torch.device(configs["device"] if torch.cuda.is_available() else "cpu")
    if dataset_cfg["dataset"] == "NuscDataset":
        from datasets.nusc import NuscDataset as Dataset
    elif dataset_cfg["dataset"] == "KittiDataset":
        from datasets.kitti import KittiDataset as Dataset
    else:
        raise NotImplementedError("Dataset not implemented")

    dataset = Dataset(dataset_cfg, use_label=opt.seg_loss_weight > 0, use_depth=opt.depth_loss_weight > 0)
    logger.info(f"Dataset cameras_extent: {dataset.cameras_extent} - size: {len(dataset)}")
    road = Road(model_cfg, dataset, device=device, vis=train_cfg.vis and GUI_FLAG)

    logger.info(f"Road min_xy: {road.min_xy} - max_xy: {road.max_xy}")
    logger.info(f"bev_x_length: {road.bev_x_length} - bev_y_length: {road.bev_y_length}")

    gaussians = GaussianModel2D(model_cfg)
    gaussians.init_2d_gaussian(road.vertices, road.rotation, road.rgb, road.label, road.resolution, road.ref_pose, dataset.cameras_extent)
    opt["position_lr_max_steps"] = len(dataset) * opt.epochs
    gaussians.training_setup(opt)

    bev_cam = road.bev_camera
    bev_cam_height = bev_cam.cam2world[2, 3]

    road_pointcloud = dataset.road_pointcloud
    if road_pointcloud is not None:
        road_point_root = os.path.join(output_root, "road_point")
        gt_render(dataset, road.min_xy, road.max_xy, bev_cam_height, wh=(bev_cam.image_width, bev_cam.image_height), save_root=road_point_root, device=device)

    if model_cfg.use_exposure:
        exposure_model = ExposureModel(num_camera=len(dataset.camera_names)).to(device)
        exposure_optimizer = torch.optim.Adam(exposure_model.parameters(), lr=opt.exposure_lr)

    first_iter = 0
    if train_cfg.start_checkpoint:
        (model_params, first_iter) = torch.load(train_cfg.start_checkpoint)
        gaussians.restore(model_params, opt)
    bg_color = [1, 1, 1] if model_cfg.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    loss_function = L1MaskedLoss()
    depth_loss_function = L1MaskedLoss()
    CE_loss_with_mask = CELossWithMask()

    OPT_SMOOTH = opt.smooth_loss_weight > 0
    OPT_SEG = opt.seg_loss_weight > 0
    OPT_DEPTH = opt.depth_loss_weight > 0
    OPT_Z = opt.z_weight > 0 and road_pointcloud is not None

    gaussian_xy = gaussians.get_xyz[:, :2]
    if OPT_SMOOTH > 0:
        smooth_near_idx = road.four_indices  # (n,4) faster
    if OPT_Z:
        if gaussian_xy.shape[0] < road_pointcloud["xyz"].shape[0]:
            sample_idx = torch.randint(0, road_pointcloud["xyz"].shape[0], (gaussian_xy.shape[0],))
            sample_xyz = road_pointcloud["xyz"][sample_idx]
        else:
            sample_xyz = road_pointcloud["xyz"]
        z_near_idx = knn_points(gaussian_xy[None], sample_xyz[None, :, :2], K=1, return_nn=False).idx  # (1, n, 1)
        z_near_idx = z_near_idx.squeeze(0)  # (n,1)

    logger.info(f"render loss: ON")
    logger.info(f"smooth loss: {'ON-' + str(opt.smooth_loss_weight) if OPT_SMOOTH else 'OFF'}")
    logger.info(f"semantic loss: {'ON-' + str(opt.seg_loss_weight) if OPT_SEG else 'OFF'}")
    logger.info(f"z loss: {'ON-' + str(opt.z_weight) if OPT_Z else 'OFF'}")
    logger.info(f"depth loss: {'ON-' + str(opt.depth_loss_weight) if OPT_DEPTH else 'OFF'}")

    cost_time = 0
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.epochs * len(dataset)), desc="Training progress")
    first_iter += 1
    iteration = first_iter
    max_iter = opt.epochs * len(dataset)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=True, drop_last=True)
    for epoch in range(opt.epochs):
        for sample in dataloader:
            iter_start.record()
            for key, value in sample.items():
                if key != "image_name":
                    sample[key] = value[0].to(device)
                else:
                    sample[key] = value[0]
            gaussians.update_learning_rate(iteration)
            if not gaussians.use_rgb and iteration % 1000 == 0:
                gaussians.oneupSHdegree()

            image_name = sample["image_name"]
            gt_image = sample["image"]
            image_idx = sample["idx"].item()
            cam_idx = sample["cam_idx"].item()
            R, T = sample["R"], sample["T"]

            NEAR, FAR = 1, 20
            viewpoint_cam = PerspectiveCamera(R, T, sample["K"], sample["W"], sample["H"], NEAR, FAR, device)

            bg = torch.rand((3), device="cuda") if opt.random_background else background

            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
            src_render_image, render_depth, render_mask = render_pkg["render"], render_pkg["depth"][0], render_pkg["mask"]
            visibility_filter = render_pkg["visibility_filter"]
            hit_num = torch.sum(visibility_filter)
            if model_cfg.use_exposure:
                render_image = exposure_model(cam_idx, src_render_image)
            else:
                render_image = src_render_image
            render_image = render_image.permute(1, 2, 0)
            src_render_image = src_render_image.permute(1, 2, 0)  # (H, W, 3)
            src_render_image = src_render_image.detach().cpu().numpy() * 255
            src_render_image = cv2.cvtColor(src_render_image.astype(np.uint8), cv2.COLOR_RGB2BGR)

            valid_mask = torch.bitwise_and(render_depth.detach() > viewpoint_cam.znear, render_depth.detach() < viewpoint_cam.zfar)
            loss_mask = valid_mask.float()
            if "mask" in sample:
                seg_mask = sample["mask"]
                loss_mask *= seg_mask
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            loss_mask = loss_mask.cpu().numpy().astype(np.uint8)
            loss_mask = cv2.erode(loss_mask, kernel)
            loss_mask = cv2.dilate(loss_mask, kernel)
            loss_mask = torch.tensor(loss_mask, device=device)

            render_loss = loss_function(render_image, gt_image, loss_mask[:, :, None])
            total_loss = render_loss.mean()

            current_gaussian_xyz = gaussians.get_xyz
            if OPT_DEPTH > 0:
                gt_depth = sample["depth"]
                gt_mask_depth = gt_depth > 0
                depth_loss = depth_loss_function(render_depth, gt_depth, loss_mask * gt_mask_depth.float()) * opt.depth_loss_weight
                total_loss += depth_loss.mean()

            if OPT_SEG > 0:
                gt_seg = sample["label"]
                # 转为long类型
                gt_seg = gt_seg.long()
                label_feature = render_label(viewpoint_cam, gaussians, pipe, bg)
                render_seg = label_feature["render"]
                render_seg = render_seg.permute(1, 2, 0)
                seg_loss = CE_loss_with_mask(render_seg.reshape(-1, render_seg.shape[-1]), gt_seg.reshape(-1), loss_mask.reshape(-1)) * opt.seg_loss_weight
                total_loss += seg_loss

            if OPT_SMOOTH > 0:
                vis_z = current_gaussian_xyz[:, 2][visibility_filter]  # (m,)
                cur_near_idx = smooth_near_idx[visibility_filter]  # (m,4)
                smooth_near_z = knn_gather(current_gaussian_xyz[:, 2:].unsqueeze(0), cur_near_idx.unsqueeze(0))  # (1, m, 4, 1)
                smooth_near_z = smooth_near_z[0, :, :, 0]  # (m,4)
                z_smooth_loss = torch.mean((smooth_near_z - vis_z[:, None]) ** 2, dim=1).sum() * opt.smooth_loss_weight
                total_loss += z_smooth_loss

            if OPT_Z:
                cam_xy = T[:2]
                surround_min_xy = cam_xy - 10
                surround_max_xy = cam_xy + 10
                surround_filter1 = torch.logical_and(current_gaussian_xyz[:, 0] > surround_min_xy[0], current_gaussian_xyz[:, 0] < surround_max_xy[0])
                surround_filter2 = torch.logical_and(current_gaussian_xyz[:, 1] > surround_min_xy[1], current_gaussian_xyz[:, 1] < surround_max_xy[1])
                surround_filter = torch.logical_and(surround_filter1, surround_filter2)
                cur_near_idx = z_near_idx[surround_filter]  # (m,k)
                vis_z = current_gaussian_xyz[:, 2][surround_filter]  # (m,)

                # cur_near_idx = z_near_idx[visibility_filter]  # (m,k)
                # vis_z = current_gaussian_xyz[:, 2][visibility_filter]  # (m,)

                near_z = knn_gather(sample_xyz[:, 2:].unsqueeze(0), cur_near_idx.unsqueeze(0))  # (1, m, 1, 1)
                near_z = near_z[0, :, 0, 0]  # (m,k)
                z_loss = (near_z - vis_z) ** 2
                z_loss = z_loss.sum() * opt.z_weight

                total_loss += z_loss

            total_loss.backward()

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
            if model_cfg.use_exposure:
                exposure_optimizer.step()
                exposure_optimizer.zero_grad(set_to_none=True)

            iter_end.record()
            torch.cuda.synchronize()
            cost_time += iter_start.elapsed_time(iter_end)

            with torch.no_grad():
                ema_loss_for_log = 0.4 * total_loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "hit_num": f"{hit_num.item()}/{len(visibility_filter)}"})
                    progress_bar.update(10)
                if iteration == max_iter - 1:
                    progress_bar.close()
            iteration += 1

        logger.info(f"EPOCH-{epoch}. Cost time: {cost_time / 1000} s")

        if True:
            current_root = os.path.join(img_root, f"EPOCH-{epoch}_IDX-{image_idx}", f"{image_name}")
            if epoch == opt.epochs - 1:
                final_root = os.path.join(img_root, "final")
                current_root = final_root
            os.makedirs(current_root, exist_ok=True)

            # ===> mask
            mask = seg_mask.cpu().numpy() * 255
            vis_loss_mask = loss_mask.cpu().numpy().astype(np.uint8) * 255
            cv2.imwrite(os.path.join(current_root, f"loss_mask.png"), vis_loss_mask)

            # ===> rgb
            gt_image = gt_image * 255
            gt_image = gt_image.cpu().numpy().astype(np.uint8)
            gt_image = cv2.cvtColor(gt_image, cv2.COLOR_RGB2BGR)
            gt_image_mask = cv2.addWeighted(gt_image, 0.5, np.stack([mask, mask, mask], axis=-1).astype(np.uint8), 0.5, 0)
            cv2.imwrite(os.path.join(current_root, f"gt_image.png"), gt_image)
            cv2.imwrite(os.path.join(current_root, f"gt_mask.png"), gt_image_mask)

            render_mask = render_mask.detach().cpu().numpy()
            render_image = render_image.detach().cpu().numpy() * 255
            render_image = cv2.cvtColor(render_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
            render_image[~render_mask] = 0
            cv2.imwrite(os.path.join(current_root, f"render_src_image.png"), src_render_image)
            cv2.imwrite(os.path.join(current_root, f"render_image.png"), render_image)

            # ===> semantic
            if OPT_SEG:
                gt_label = render_semantic(gt_seg.cpu().numpy(), dataset.filted_color_map)
                gt_label = cv2.cvtColor(gt_label, cv2.COLOR_RGB2BGR)
                gt_blend = cv2.addWeighted(gt_image, 0.7, gt_label, 0.3, 0)
                cv2.imwrite(os.path.join(current_root, f"gt_label.png"), gt_label)
                cv2.imwrite(os.path.join(current_root, f"gt_blend.png"), gt_blend)

                vis_render_seg = render_semantic(np.argmax(render_seg.detach().cpu().numpy(), axis=-1), dataset.filted_color_map)
                vis_render_seg = cv2.cvtColor(vis_render_seg, cv2.COLOR_RGB2BGR)
                vis_render_seg[~render_mask] = 0
                render_blend = cv2.addWeighted(render_image, 0.7, vis_render_seg, 0.3, 0)
                cv2.imwrite(os.path.join(current_root, f"render_label_vis.png"), vis_render_seg)
                cv2.imwrite(os.path.join(current_root, f"render_blend.png"), render_blend)

            # ===> loss
            vis_render_loss = loss2color(render_loss.detach().cpu().numpy())
            cv2.imwrite(os.path.join(current_root, f"render_loss.png"), vis_render_loss)

            # ===> depth
            vis_render_depth = depth2color(render_depth.detach().cpu().numpy(), render_mask)
            vis_render_depth = cv2.cvtColor(vis_render_depth, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(current_root, f"render_depth_vis.png"), vis_render_depth)

            # ==============> BEV render
            # TODO for very large scene, should render chunk by chunk
            bev_pkg = render(bev_cam, gaussians, pipe, bg)
            src_bev_image, bev_depth, bev_mask = bev_pkg["render"], bev_pkg["depth"], bev_pkg["mask"]

            # ===> mask
            bev_mask = bev_mask.cpu().numpy()
            bev_mask = bev_mask.astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            bev_mask = cv2.erode(bev_mask, kernel)
            bev_mask = bev_mask > 0
            cv2.imwrite(os.path.join(current_root, f"bev_mask.png"), bev_mask * 255)

            # ===> height
            bev_height = bev_cam_height - bev_depth
            bev_height = bev_height[0].detach().cpu().numpy()
            if train_cfg.save:
                if bev_mask.sum() < 0.3 * bev_mask.size:
                    bev_height[~bev_mask] = 0
                    sp_matrix = sp.csr_matrix(bev_height)
                    np.savez(os.path.join(current_root, f"bev_height.npz"), data=sp_matrix.data, indices=sp_matrix.indices, indptr=sp_matrix.indptr,
                             shape=sp_matrix.shape)
                else:
                    np.save(os.path.join(current_root, f"bev_height.npy"), bev_height)
            vis_bev_height = depth2color(bev_height, mask=bev_mask)
            vis_bev_height = cv2.cvtColor(vis_bev_height, cv2.COLOR_RGB2BGRA)
            vis_bev_height[~bev_mask] = 0
            cv2.imwrite(os.path.join(current_root, f"bev_height_vis.png"), vis_bev_height)

            # ===> rbg
            if model_cfg.use_exposure:
                bev_image = exposure_model(0, src_bev_image)
            else:
                bev_image = src_bev_image

            bev_image = bev_image.permute(1, 2, 0)
            bev_image = bev_image.detach().cpu().numpy() * 255
            bev_image = cv2.cvtColor(bev_image.astype(np.uint8), cv2.COLOR_RGB2BGRA)
            bev_image[~bev_mask] = 0
            cv2.imwrite(os.path.join(current_root, f"bev_image.png"), bev_image)

            src_bev_image = src_bev_image.permute(1, 2, 0)
            src_bev_image = src_bev_image.detach().cpu().numpy() * 255
            src_bev_image = cv2.cvtColor(src_bev_image.astype(np.uint8), cv2.COLOR_RGB2BGRA)
            src_bev_image[~bev_mask] = 0
            cv2.imwrite(os.path.join(current_root, f"bev_scr_image.png"), src_bev_image)

            # ===> semantic
            if OPT_SEG:
                label_feature = render_label(bev_cam, gaussians, pipe, bg)
                bev_label = label_feature["render"].permute(1, 2, 0)  # (H, W, C)
                bev_label = np.argmax(bev_label.detach().cpu().numpy(), axis=-1)  # (H, W)
                vis_bev_label = render_semantic(bev_label, dataset.filted_color_map)
                vis_bev_label = cv2.cvtColor(vis_bev_label, cv2.COLOR_RGB2BGRA)
                vis_bev_label[~bev_mask] = 0
                cv2.imwrite(os.path.join(current_root, f"bev_label.png"), bev_label)
                cv2.imwrite(os.path.join(current_root, f"bev_label_vis.png"), vis_bev_label)

        if train_cfg.eval:
            if road_pointcloud is not None:
                logger.info(f"Just start eval .....")
                bev_metric = eval_bev_metric(road_point_root, current_root, dataset.num_class)
                for k, v in bev_metric.items():
                    logger.info(f"[Epoch{epoch}] - bev {k}: {v}")

                z_metric = eval_z_metric(road_pointcloud["xyz"], gaussians.get_xyz)
                logger.info(f"[Epoch{epoch}] - z_metric: {z_metric}")

    if train_cfg.save:
        os.makedirs(ply_root, exist_ok=True)
        gaussians.save_ply(os.path.join(ply_root, f"final.ply"))

        ckpt_path = os.path.join(output_root, "final.pth")
        torch.save(gaussians.capture(), ckpt_path)

        if model_cfg.use_exposure:
            exposure_ckpt_path = os.path.join(output_root, "exposure.pth")
            torch.save(exposure_model.state_dict(), exposure_ckpt_path)


if __name__ == "__main__":
    configs = get_configs()
    configs = addict.Dict(configs)
    train(configs)
