import os

import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from pytorch3d.ops import ball_query, knn_points
from pytorch3d.ops.utils import masked_gather
from diff_gaussian_rasterization.scene.cameras import PerspectiveCamera

from utils.render import render, render_label
from utils.metrics import eval_metrics
from utils.image import render_semantic
from models.loss import MESMaskedLoss


def mse2psnr(mse):
    """
    :param mse: scalar
    :return:    scalar np.float32
    """
    mse = np.maximum(mse, 1e-10)  # avoid -inf or nan when mse is very small.
    psnr = -10.0 * np.log10(mse)
    return psnr.astype(np.float32)


def eval_metric(gaussians, exposure_model, dataset, bg_color, pipe, eval_root=None):
    loss_all = []
    loss_fuction = MESMaskedLoss()
    image_segs = []
    gt_segs = []
    cnt = 0
    num_class = dataset.num_class
    device = "cuda:0"

    with torch.no_grad():
        dataloader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=True, drop_last=True)
        for sample in tqdm(dataloader):
            for key, value in sample.items():
                if key != "image_name":
                    sample[key] = value[0].to(device)  # 注意batch_size
                else:
                    sample[key] = value[0]

            gt_image = sample["image"]
            mask = sample["mask"]
            gt_seg = sample["label"]
            image_idx = sample["idx"].item()
            cam_idx = sample["cam_idx"].item()
            R = sample["R"]
            T = sample["T"]
            image_name = sample["image_name"]
            viewpoint_cam = PerspectiveCamera(R, T, sample["K"], sample["W"], sample["H"], 1, 10, device)

            render_pkg = render(viewpoint_cam, gaussians, pipe, bg_color)
            src_render_image, render_depth, valid_mask = render_pkg["render"], render_pkg["depth"], render_pkg["mask"]
            render_image = exposure_model(cam_idx, src_render_image)
            render_image = render_image.permute(1, 2, 0)
            mask = mask * valid_mask.float()

            label_feature = render_label(viewpoint_cam, gaussians, pipe, bg_color)
            render_seg = label_feature["render"]
            render_seg = render_seg.permute(1, 2, 0)

            mse_loss = loss_fuction(render_image, gt_image, mask[..., None])
            mse_loss_np = mse_loss.cpu().detach().numpy()
            loss_all.append(mse_loss_np)

            render_image = render_image.detach().cpu().numpy()
            gt_image = gt_image.detach().cpu().numpy()

            render_image = (render_image * 255).astype(np.uint8)[:, :, ::-1]  # RGB2BGR
            gt_image = (gt_image * 255).astype(np.uint8)[:, :, ::-1]  # RGB2BGR
            if eval_root:
                cv2.imwrite(os.path.join(eval_root, f"{image_name}-render.png"), render_image)
                cv2.imwrite(os.path.join(eval_root, f"{image_name}.png"), gt_image)

            # save seg numpy array
            images_seg_np = render_seg.detach().cpu().numpy()[None]  # (1, H, W, C)
            images_seg_np = np.argmax(images_seg_np, axis=-1)  # (1, H, W)

            vis_seg = render_semantic(images_seg_np[0], dataset.filted_color_map)[:, :, ::-1]
            blend_image = cv2.addWeighted(gt_image, 0.5, vis_seg, 0.5, 0)

            if eval_root:
                cv2.imwrite(os.path.join(eval_root, f"{image_name}-vis_seg.png"), vis_seg)
                cv2.imwrite(os.path.join(eval_root, f"{image_name}-blend.png"), blend_image)

            images_seg_np[images_seg_np == num_class - 1] = 255
            images_seg_np[images_seg_np == 0] = 255
            images_seg_np -= 1
            images_seg_np[images_seg_np == 254] = 255
            image_segs.append(images_seg_np)

            mask = mask.detach().cpu().numpy().astype(np.uint8)
            gt_seg_np = gt_seg.detach().cpu().numpy()
            gt_seg_np *= mask
            gt_seg_np = gt_seg_np[None]
            vis_gt_seg = render_semantic(gt_seg_np[0], dataset.filted_color_map)[:, :, ::-1]
            if eval_root:
                cv2.imwrite(os.path.join(eval_root, f"{image_name}-vis_gt_seg.png"), vis_gt_seg)
            gt_seg_np[gt_seg_np == num_class - 1] = 255
            gt_seg_np[gt_seg_np == 0] = 255
            gt_seg_np -= 1
            gt_seg_np[gt_seg_np == 254] = 255
            gt_segs.append(gt_seg_np)
            cnt += 1

    loss_all = np.array(loss_all)
    loss_mean = np.mean(loss_all)
    psnr_mean = mse2psnr(loss_mean)
    if len(image_segs) > 1:
        image_segs = np.concatenate(image_segs, axis=0)
        gt_segs = np.concatenate(gt_segs, axis=0)
    else:
        image_segs = np.array(image_segs)[None]
        gt_segs = np.array(gt_segs)[None]
    results = eval_metrics(image_segs, gt_segs,
                           num_classes=num_class - 2,
                           ignore_index=255,
                           metrics=['mIoU'],
                           nan_to_num=None,
                           label_map=dict(),
                           reduce_zero_label=False)
    return {"MSE": loss_mean, "PSNR": psnr_mean, "mIoU": results}


def eval_bev_metric(gt_root, pre_root, num_class):
    mask1 = cv2.imread(os.path.join(gt_root, "bev_mask.png"), cv2.IMREAD_GRAYSCALE)
    mask1 = mask1 > 0
    mask2 = cv2.imread(os.path.join(pre_root, "bev_mask.png"), cv2.IMREAD_GRAYSCALE)
    mask2 = mask2 > 0
    mask = mask1 * mask2  # (H,W)

    # ======> seg metric
    gt_label = cv2.imread(os.path.join(gt_root, "bev_label.png"), cv2.IMREAD_GRAYSCALE).astype(np.uint8)  # (H,W)
    pre_label = cv2.imread(os.path.join(pre_root, "bev_label.png"), cv2.IMREAD_GRAYSCALE).astype(np.uint8)  # (H,W)

    gt_label *= mask
    gt_label = gt_label[None]
    gt_label[gt_label == num_class - 1] = 255
    gt_label[gt_label == 0] = 255
    gt_label -= 1
    gt_label[gt_label == 254] = 255

    pre_label = pre_label[None]
    pre_label[pre_label == num_class - 1] = 255
    pre_label[pre_label == 0] = 255
    pre_label -= 1
    pre_label[pre_label == 254] = 255

    pre_label = pre_label[None]
    gt_label = gt_label[None]

    results = eval_metrics(pre_label, gt_label,
                           num_classes=num_class - 2,
                           ignore_index=255,
                           metrics=['mIoU'],
                           nan_to_num=None,
                           label_map=dict(),
                           reduce_zero_label=False)

    # =====> rgb metric
    loss_fuction = MESMaskedLoss()
    gt_image = cv2.imread(os.path.join(gt_root, "bev_image.png"), cv2.IMREAD_COLOR)  # (H,W,3)
    render_image = cv2.imread(os.path.join(pre_root, "bev_image.png"), cv2.IMREAD_COLOR)  # (H,W,3)
    gt_image = gt_image / 255.0
    render_image = render_image / 255.0
    mse_loss = loss_fuction(torch.from_numpy(render_image), torch.from_numpy(gt_image), torch.from_numpy(mask[..., None]))
    mse_loss_np = mse_loss.numpy()
    loss_mean = np.mean(mse_loss_np)
    psnr_mean = mse2psnr(loss_mean)

    return {"MSE": loss_mean, "PSNR": psnr_mean, "mIoU": results}


def eval_z_metric(lidar_xyz, guassian_xyz):
    near_idx = ball_query(guassian_xyz[None, :, :2], lidar_xyz[None, :, :2], K=1, return_nn=False, radius=0.1).idx  # (1, N, 1)
    valid_mask = near_idx.squeeze(0) != -1  # (N, 1) 
    gt_z = masked_gather(lidar_xyz[None, :, 2:3], near_idx).reshape(-1, 1)  # (N, 1)
    gt_z = gt_z[valid_mask]  # (M, 1)
    pre_z = guassian_xyz[:, 2:3][valid_mask]  # (M, 1)

    loss = torch.sqrt(torch.mean((gt_z - pre_z) ** 2))
    return loss.item()


def eval_chamfer_metric(lidar_xyz, guassian_xyz):
    dists1 = knn_points(guassian_xyz[None], lidar_xyz[None], K=1, return_nn=False).dists
    dists2 = knn_points(lidar_xyz[None], guassian_xyz[None], K=1, return_nn=False).dists

    # 排序
    dists1 = dists1.flatten()
    dists2 = dists2.flatten()

    dist1 = torch.sort(dists1)[0]
    dist1 = dist1[:int(0.97 * len(dist1))]

    dist2 = torch.sort(dists2)[0]
    dist2 = dist2[:int(0.97 * len(dist2))]

    eval_chamfer_metric = torch.sqrt(torch.mean(dist1 ** 2)) + torch.sqrt(torch.mean(dist2 ** 2))

    return eval_chamfer_metric.item()
