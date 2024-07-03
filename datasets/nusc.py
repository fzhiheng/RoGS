import os
from copy import deepcopy
from multiprocessing.pool import ThreadPool as Pool

import cv2
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from plyfile import PlyData
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points, transform_matrix

from datasets.base import BaseDataset


def get_nusc_filted_color_map():
    colors = np.zeros((256, 1, 3), dtype='uint8')
    colors[0, :, :] = [0, 0, 0]  # mask
    colors[1, :, :] = [0, 0, 255]  # all lane
    colors[2, :, :] = [255, 0, 0]  # curb
    colors[3, :, :] = [211, 211, 211]  # road and manhole
    colors[4, :, :] = [0, 191, 255]  # sidewalk
    colors[5, :, :] = [152, 251, 152]  # terrain
    colors[6, :, :] = [157, 234, 50]  # background
    return colors


def get_nusc_origin_color_map():
    colors = np.zeros((256, 1, 3), dtype='uint8')
    colors[0, :, :] = [165, 42, 42]  # Bird
    colors[1, :, :] = [0, 192, 0]  # Ground Animal
    colors[2, :, :] = [196, 196, 196]  # Curb
    colors[3, :, :] = [190, 153, 153]  # Fence
    colors[4, :, :] = [180, 165, 180]  # Guard Rail
    colors[5, :, :] = [90, 120, 150]  # Barrier
    colors[6, :, :] = [102, 102, 156]  # Wall
    colors[7, :, :] = [128, 64, 255]  # Bike Lane
    colors[8, :, :] = [140, 140, 200]  # Crosswalk - Plain
    colors[9, :, :] = [170, 170, 170]  # Curb Cut
    colors[10, :, :] = [250, 170, 160]  # Parking
    colors[11, :, :] = [96, 96, 96]  # Pedestrian Area
    colors[12, :, :] = [230, 150, 140]  # Rail Track
    colors[13, :, :] = [128, 64, 128]  # Road
    colors[14, :, :] = [110, 110, 110]  # Service Lane
    colors[15, :, :] = [244, 35, 232]  # Sidewalk
    colors[16, :, :] = [150, 100, 100]  # Bridge
    colors[17, :, :] = [70, 70, 70]  # Building
    colors[18, :, :] = [150, 120, 90]  # Tunnel
    colors[19, :, :] = [220, 20, 60]  # Person
    colors[20, :, :] = [255, 0, 0]  # Bicyclist
    colors[21, :, :] = [255, 0, 100]  # Motorcyclist
    colors[22, :, :] = [255, 0, 200]  # Other Rider
    colors[23, :, :] = [200, 128, 128]  # Lane Marking - Crosswalk
    colors[24, :, :] = [255, 255, 255]  # Lane Marking - General
    colors[25, :, :] = [64, 170, 64]  # Mountain
    colors[26, :, :] = [230, 160, 50]  # Sand
    colors[27, :, :] = [70, 130, 180]  # Sky
    colors[28, :, :] = [190, 255, 255]  # Snow
    colors[29, :, :] = [152, 251, 152]  # Terrain
    colors[30, :, :] = [107, 142, 35]  # Vegetation
    colors[31, :, :] = [0, 170, 30]  # Water
    colors[32, :, :] = [255, 255, 128]  # Banner
    colors[33, :, :] = [250, 0, 30]  # Bench
    colors[34, :, :] = [100, 140, 180]  # Bike Rack
    colors[35, :, :] = [220, 220, 220]  # Billboard
    colors[36, :, :] = [220, 128, 128]  # Catch Basin
    colors[37, :, :] = [222, 40, 40]  # CCTV Camera
    colors[38, :, :] = [100, 170, 30]  # Fire Hydrant
    colors[39, :, :] = [40, 40, 40]  # Junction Box
    colors[40, :, :] = [33, 33, 33]  # Mailbox
    colors[41, :, :] = [100, 128, 160]  # Manhole
    colors[42, :, :] = [142, 0, 0]  # Phone Booth
    colors[43, :, :] = [70, 100, 150]  # Pothole
    colors[44, :, :] = [210, 170, 100]  # Street Light
    colors[45, :, :] = [153, 153, 153]  # Pole
    colors[46, :, :] = [128, 128, 128]  # Traffic Sign Frame
    colors[47, :, :] = [0, 0, 80]  # Utility Pole
    colors[48, :, :] = [250, 170, 30]  # Traffic Light
    colors[49, :, :] = [192, 192, 192]  # Traffic Sign (Back)
    colors[50, :, :] = [220, 220, 0]  # Traffic Sign (Front)
    colors[51, :, :] = [140, 140, 20]  # Trash Can
    colors[52, :, :] = [119, 11, 32]  # Bicycle
    colors[53, :, :] = [150, 0, 255]  # Boat
    colors[54, :, :] = [0, 60, 100]  # Bus
    colors[55, :, :] = [0, 0, 142]  # Car
    colors[56, :, :] = [0, 0, 90]  # Caravan
    colors[57, :, :] = [0, 0, 230]  # Motorcycle
    colors[58, :, :] = [0, 80, 100]  # On Rails
    colors[59, :, :] = [128, 64, 64]  # Other Vehicle
    colors[60, :, :] = [0, 0, 110]  # Trailer
    colors[61, :, :] = [0, 0, 70]  # Truck
    colors[62, :, :] = [0, 0, 192]  # Wheeled Slow
    colors[63, :, :] = [32, 32, 32]  # Car Mount
    colors[64, :, :] = [120, 10, 10]  # Ego Vehicle
    # colors[65, :, :] = [0, 0, 0] # Unlabeled
    return colors


def get_nusc_label_remaps():
    colors = np.ones((256, 1), dtype="uint8")
    colors *= 6  # background
    colors[7, :] = 1  # Lane marking
    colors[8, :] = 1
    colors[14, :] = 1
    colors[23, :] = 1
    colors[24, :] = 1
    colors[2, :] = 2  # curb
    colors[9, :] = 2  # curb cut
    colors[41, :] = 3  # Manhole
    colors[13, :] = 3  # road
    colors[15, :] = 4  # sidewalk
    colors[29, :] = 5  # terrain
    return colors


def label2mask(label):
    # Bird, Ground Animal, Curb, Fence, Guard Rail,
    # Barrier, Wall, Bike Lane, Crosswalk - Plain, Curb Cut,
    # Parking, Pedestrian Area, Rail Track, Road, Service Lane,
    # Sidewalk, Bridge, Building, Tunnel, Person,
    # Bicyclist, Motorcyclist, Other Rider, Lane Marking - Crosswalk, Lane Marking - General,
    # Mountain, Sand, Sky, Snow, Terrain,
    # Vegetation, Water, Banner, Bench, Bike Rack,
    # Billboard, Catch Basin, CCTV Camera, Fire Hydrant, Junction Box,
    # Mailbox, Manhole, Phone Booth, Pothole, Street Light,
    # Pole, Traffic Sign Frame, Utility Pole, Traffic Light, Traffic Sign (Back),
    # Traffic Sign (Front), Trash Can, Bicycle, Boat, Bus,
    # Car, Caravan, Motorcycle, On Rails, Other Vehicle,
    # Trailer, Truck, Wheeled Slow, Car Mount, Ego Vehicle
    mask = np.ones_like(label)
    label_off_road = ((0 <= label) & (label <= 1)) | ((3 <= label) & (label <= 6)) | ((10 <= label) & (label <= 12)) \
                     | ((16 <= label) & (label <= 22)) | ((25 <= label) & (label <= 28)) | (
                             (30 <= label) & (label <= 40)) | (label >= 42)

    # dilate itereation 2 for moving objects
    label_movable = label >= 52
    kernel = np.ones((10, 10), dtype=np.uint8)
    label_movable = cv2.dilate(label_movable.astype(np.uint8), kernel, 2).astype(bool)

    label_off_road = label_off_road | label_movable
    mask[label_off_road] = 0
    label[~(mask.astype(bool))] = 64
    mask = mask.astype(np.float32)
    return mask, label


def loda_depth(depth_file):
    loaded_data = np.load(depth_file)
    depth_img = sp.csr_matrix((loaded_data['data'], loaded_data['indices'], loaded_data['indptr']), shape=loaded_data['shape'])
    return depth_img


def worldpoint2camera(points: np.ndarray, WH, cam2world, cam_intrinsic, min_dist: float = 1.0):
    """
    1. transform world points to camera points
    Args:
        points: (N, 3)
        image:  (H, W, 3)
        cam2world: (4, 4)
        cam_intrinsic: (3, 3)
        min_dist: float

    Returns:
        uv: (2, N)
        depths: (N, )
        mask: (N, )

    """
    width, height = WH
    world2cam = np.linalg.inv(cam2world)  # (4, 4)
    points_cam = world2cam[:3, :3] @ points.T + world2cam[:3, 3:4]  # (3, N)
    depths = points_cam[2, :]  # (N, )
    points_uv1 = view_points(points_cam, np.array(cam_intrinsic), normalize=True)  # (3, N)

    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
    # casing for non-keyframes which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points_uv1[0, :] > 1)
    mask = np.logical_and(mask, points_uv1[0, :] < width - 1)
    mask = np.logical_and(mask, points_uv1[1, :] > 1)
    mask = np.logical_and(mask, points_uv1[1, :] < height - 1)

    uv = points_uv1[:, mask][:2, :]
    uv = np.round(uv).astype(np.uint16)
    depths = depths[mask]
    return uv, depths, mask


class NuscDataset(BaseDataset):
    def __init__(self, configs, use_label=True, use_depth=False):
        self.nusc = NuScenes(version="v1.0-{}".format(configs["version"]), dataroot=configs["base_dir"], verbose=True)
        self.version = configs["version"]
        super().__init__()
        self.resized_image_size = (configs["image_width"], configs["image_height"])
        self.base_dir = configs["base_dir"]
        self.image_dir = configs["image_dir"]
        self.camera_names = configs["camera_names"]
        self.min_distance = configs["min_distance"]
        clip_list = configs["clip_list"]
        self.chassis2world_unique = []
        self.raw_wh = dict()

        self.use_label = use_label
        self.use_depth = use_depth
        self.lidar_times_all = []
        self.lidar_filenames_all = []
        self.lidar2world_all = []

        road_pointcloud = dict()
        for scene_name in tqdm(clip_list, desc="Loading data clips"):
            records = [samp for samp in self.nusc.sample if self.nusc.get("scene", samp["scene_token"])["name"] in scene_name]
            records.sort(key=lambda x: (x['timestamp']))

            print(f"Loading image from scene {scene_name}")
            cam_info, chassis_info = self.load_cameras(records)

            self.raw_wh[scene_name] = cam_info["wh"]
            self.camera2world_all.extend(cam_info["poses"])
            self.camera_times_all.extend(cam_info["times"])
            self.cameras_K_all.extend(cam_info["intrinsics"])
            self.cameras_idx_all.extend(cam_info["idxs"])
            self.image_filenames_all.extend(cam_info["filenames"])

            self.chassis2world_unique.extend(chassis_info["unique_poses"])
            self.chassis2world_all.extend(chassis_info["poses"])

            label_filenames = [rel_camera_path.replace("/CAM", "/seg_CAM").replace(".jpg", ".png") for rel_camera_path in cam_info["filenames"]]
            self.label_filenames_all.extend(label_filenames)

            lidar_info = self.load_lidars(records)
            self.lidar_times_all.extend(lidar_info["times"])
            self.lidar_filenames_all.extend(lidar_info["filenames"])
            self.lidar2world_all.extend(lidar_info["poses"])


            point_gt_path = os.path.join(configs["road_gt_dir"], f"{scene_name}.ply")
            xyz, rgb, label = self.load_gt_points(point_gt_path)
            road_pointcloud[scene_name] = {"xyz": xyz, "rgb": rgb, "label": label}

        self.file_check()
        if len(self.image_filenames_all) == 0:
            raise FileNotFoundError("No data found in the dataset")

        self.chassis2world_unique = np.array(self.chassis2world_unique)
        self.chassis2world_all = np.array(self.chassis2world_all)  # [N, 4, 4]
        self.camera2world_all = np.array(self.camera2world_all)  # [N, 4, 4]
        self.camera_times_all = np.array(self.camera_times_all)  # [N, ]

        self.lidar2world_all = np.array(self.lidar2world_all)  # [N, 4, 4]
        self.lidar_times_all = np.array(self.lidar_times_all)  # [N, ]

        self.ref_pose = self.chassis2world_unique[0]
        ref_pose_inv = np.linalg.inv(self.ref_pose)
        self.chassis2world_unique = ref_pose_inv @ self.chassis2world_unique
        self.camera2world_all = ref_pose_inv @ self.camera2world_all
        self.chassis2world_all = ref_pose_inv @ self.chassis2world_all
        self.lidar2world_all = ref_pose_inv @ self.lidar2world_all

        for scene_name in road_pointcloud.keys():
            xyz = road_pointcloud[scene_name]["xyz"]
            new_xyz = ref_pose_inv[:3, :3] @ xyz.T + ref_pose_inv[:3, 3:4]
            road_pointcloud[scene_name]["xyz"] = new_xyz.T

        self.road_pointcloud = {k: np.concatenate([road_pointcloud[s][k] for s in road_pointcloud.keys()], axis=0) for k in ("xyz", "rgb", "label")}

        nerf_normalization = self.getNerfppNorm()
        self.cameras_extent = nerf_normalization["radius"]

    def __len__(self):
        return len(self.image_filenames_all)

    def __getitem__(self, idx):
        cam_idx = self.cameras_idx_all[idx]
        cam2world = self.camera2world_all[idx]
        K = self.cameras_K_all[idx]
        camera_name = self.camera_names[cam_idx]
        image_path = os.path.join(self.base_dir, self.image_filenames_all[idx])
        image_name = os.path.basename(image_path).split(".")[0]
        input_image = cv2.imread(image_path)

        crop_cy = int(self.resized_image_size[1] * 0.5)
        origin_image_size = input_image.shape
        resized_image = cv2.resize(input_image, dsize=self.resized_image_size, interpolation=cv2.INTER_LINEAR)
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        resized_image = resized_image[crop_cy:, :, :]  # crop the sky
        gt_image = (np.asarray(resized_image) / 255.0).astype(np.float32)  # [H, W, 3]
        gt_image = np.clip(gt_image, 0.0, 1.0)
        width, height = gt_image.shape[1], gt_image.shape[0]

        new_K = deepcopy(K)
        width_scale = self.resized_image_size[0] / origin_image_size[1]
        height_scale = self.resized_image_size[1] / origin_image_size[0]
        new_K[0, :] *= width_scale
        new_K[1, :] *= height_scale
        new_K[1][2] -= crop_cy
        R = cam2world[:3, :3]
        T = cam2world[:3, 3]

        sample = {"image": gt_image, "idx": idx, "cam_idx": cam_idx, "image_name": image_name, "R": R, "T": T, "K": new_K, "W": width, "H": height}

        if self.use_label:
            label_path = os.path.join(self.image_dir, self.label_filenames_all[idx])
            label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
            resized_label = cv2.resize(label, dsize=self.resized_image_size, interpolation=cv2.INTER_NEAREST)
            mask, label = label2mask(resized_label)
            if camera_name == "CAM_BACK":
                h = mask.shape[0]
                mask[int(0.83 * h):, :] = 0
            label = self.remap_semantic(label).astype(int)
            mask = mask[crop_cy:, :]
            label = label[crop_cy:, :]
            sample["mask"] = mask
            sample["label"] = label

        if self.use_depth:
            cam_time = self.camera_times_all[idx]
            lidar_idx = np.argmin(np.abs(self.lidar_times_all - cam_time))
            lidar2world = self.lidar2world_all[lidar_idx]
            lidar_path = os.path.join(self.base_dir, self.lidar_filenames_all[lidar_idx])
            points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3]
            points_world = lidar2world[:3, :3] @ points.T + lidar2world[:3, 3:4]  # (3, N)
            uv, depths, mask = worldpoint2camera(points_world.T, (width, height), cam2world, new_K)
            sort_idx = np.argsort(depths)[::-1]
            uv = uv[:, sort_idx]
            depths = depths[sort_idx]
            depth_image = np.zeros((height, width), dtype=np.float32)
            depth_image[uv[1], uv[0]] = depths
            sample["depth"] = depth_image

        return sample

    def load_gt_points(self, ply_path):
        plydata = PlyData.read(ply_path)
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)  # [N, 3]
        rgb = np.stack((np.asarray(plydata.elements[0]["r"]),
                        np.asarray(plydata.elements[0]["g"]),
                        np.asarray(plydata.elements[0]["b"])), axis=1)
        label = np.asarray(plydata.elements[0]["label"]).astype(np.uint8)
        label = label[..., None]  # [N, 1]
        return xyz, rgb, label

    def load_lidars(self, records):
        lidar_times = []
        lidar_files = []
        lidar2worlds = []

        for rec in tqdm(records):
            samp = self.nusc.get("sample_data", rec["data"]["LIDAR_TOP"])
            flag = True
            while flag or not samp["is_key_frame"]:
                flag = False

                lidar_times.append(samp["timestamp"])
                lidar_files.append(samp["filename"])

                cs_record = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
                lidar2ego = np.eye(4)
                lidar2ego[:3, :3] = Quaternion(cs_record['rotation']).rotation_matrix
                lidar2ego[:3, 3] = cs_record['translation']

                poserecord = self.nusc.get('ego_pose', samp['ego_pose_token'])
                ego2global = np.eye(4)
                ego2global[:3, :3] = Quaternion(poserecord['rotation']).rotation_matrix
                ego2global[:3, 3] = poserecord['translation']

                lidar2global = ego2global @ lidar2ego
                lidar2worlds.append(lidar2global)

                if samp["next"] != "":
                    samp = self.nusc.get('sample_data', samp["next"])
                else:
                    break

        return {"times": lidar_times, "filenames": lidar_files, "poses": lidar2worlds}

    def load_cameras(self, records):
        chassis2world_unique = []
        chassis2worlds = []
        camera2worlds = []
        cameras_K = []
        cameras_idxs = []
        cameras_times = []
        image_filenames = []
        wh = dict()

        # interpolate images from 2HZ to 12 HZ  (sample + sweep)
        for rec in tqdm(records):
            chassis_flag = True
            for camera_idx, cam in enumerate(self.camera_names):
                # compute camera key frame poses
                rec_token = rec["data"][cam]
                samp = self.nusc.get("sample_data", rec_token)
                wh.setdefault(cam, (samp["width"], samp["height"]))
                flag = True
                while flag or not samp["is_key_frame"]:
                    flag = False
                    rel_camera_path = samp["filename"]
                    cameras_times.append(samp["timestamp"])
                    image_filenames.append(rel_camera_path)

                    camera2chassis = self.compute_extrinsic2chassis(samp)
                    c2w = self.compute_chassis2world(samp)
                    chassis2worlds.append(c2w)
                    if chassis_flag:
                        chassis2world_unique.append(c2w)
                    camera2world = c2w @ camera2chassis
                    camera2worlds.append(camera2world.astype(np.float32))

                    calibrated_sensor = self.nusc.get("calibrated_sensor", samp["calibrated_sensor_token"])
                    intrinsic = np.array(calibrated_sensor["camera_intrinsic"])
                    cameras_K.append(intrinsic.astype(np.float32))

                    cameras_idxs.append(camera_idx)
                    # not key frames
                    if samp["next"] != "":
                        samp = self.nusc.get('sample_data', samp["next"])
                    else:
                        break
                chassis_flag = False
        cam_info = {"poses": camera2worlds, "intrinsics": cameras_K, "idxs": cameras_idxs, "filenames": image_filenames, "times": cameras_times, "wh": wh}
        chassis_info = {"poses": chassis2worlds, "unique_poses": chassis2world_unique}

        return cam_info, chassis_info

    def compute_chassis2world(self, samp):
        """transform sensor in world coordinate"""
        # comput current frame Homogeneous transformation matrix : from chassis 2 global
        pose_chassis2global = self.nusc.get("ego_pose", samp['ego_pose_token'])
        chassis2global = transform_matrix(pose_chassis2global['translation'],
                                          Quaternion(pose_chassis2global['rotation']),
                                          inverse=False)
        return chassis2global

    def compute_extrinsic(self, samp_a, samp_b):
        """transform from sensor_a to sensor_b"""
        sensor_a2chassis = self.compute_extrinsic2chassis(samp_a)
        sensor_b2chassis = self.compute_extrinsic2chassis(samp_b)
        sensor_a2sensor_b = np.linalg.inv(sensor_b2chassis) @ sensor_a2chassis
        return sensor_a2sensor_b

    def compute_extrinsic2chassis(self, samp):
        calibrated_sensor = self.nusc.get("calibrated_sensor", samp["calibrated_sensor_token"])
        rot = np.array(Quaternion(calibrated_sensor["rotation"]).rotation_matrix)
        tran = np.expand_dims(np.array(calibrated_sensor["translation"]), axis=0)
        sensor2chassis = np.hstack((rot, tran.T))
        sensor2chassis = np.vstack((sensor2chassis, np.array([[0, 0, 0, 1]])))  # [4, 4] camera 3D
        return sensor2chassis

    def file_check(self):
        image_paths = [os.path.join(self.base_dir, image_path) for image_path in self.image_filenames_all]
        image_exists = np.asarray(self.check_filelist_exist(image_paths))
        print(f"Drop {len(image_paths) - len(np.where(image_exists)[0])} frames out of {len(image_paths)} by image exists check")
        exists = image_exists
        label_paths = [os.path.join(self.image_dir, label_path) for label_path in self.label_filenames_all]
        label_exists = np.asarray(self.check_filelist_exist(label_paths))
        print(f"Drop {len(image_paths) - len(np.where(label_exists)[0])} frames out of {len(image_paths)} by label exists check")
        exists *= label_exists

        lidar_paths = [os.path.join(self.base_dir, lidar_path) for lidar_path in self.lidar_filenames_all]
        lidar_exists = np.asarray(self.check_filelist_exist(lidar_paths))
        print(f"Drop {len(lidar_paths) - len(np.where(lidar_exists)[0])} lidar out of {len(lidar_paths)} by lidar exists check")
        lidar_available = list(np.where(lidar_exists)[0])
        self.lidar_times_all = [self.lidar_times_all[i] for i in lidar_available]
        self.lidar_filenames_all = [self.lidar_filenames_all[i] for i in lidar_available]
        self.lidar2world_all = [self.lidar2world_all[i] for i in lidar_available]

        available_index = list(np.where(exists)[0])
        print(f"Drop {len(image_paths) - len(available_index)} frames out of {len(image_paths)} by file exists check")
        self.filter_by_index(available_index)

    def label_valid_check(self):
        label_paths = [os.path.join(self.image_dir, label_path) for label_path in self.label_filenames_all]
        label_valid = np.asarray(self.check_label_valid(label_paths))
        available_index = list(np.where(label_valid)[0])
        print(f"Drop {len(label_paths) - len(available_index)} frames out of {len(label_paths)} by label valid check")
        self.filter_by_index(available_index)

    def label_valid(self, label_name):
        label = cv2.imread(label_name, cv2.IMREAD_UNCHANGED)
        label_movable = label >= 52
        ratio_movable = label_movable.sum() / label_movable.size
        label_off_road = ((0 <= label) & (label <= 1)) | ((3 <= label) & (label <= 6)) | ((10 <= label) & (label <= 12)) \
                         | ((15 <= label) & (label <= 22)) | ((25 <= label) & (label <= 40)) | (label >= 42)
        ratio_static = label_off_road.sum() / label_off_road.size
        if ratio_movable > 0.3 or ratio_static > 0.9:
            return False
        else:
            return True

    def check_label_valid(self, filelist):
        with Pool(32) as p:
            exist_list = p.map(self.label_valid, filelist)
        return exist_list

    def filter_by_index(self, index):
        self.image_filenames_all = [self.image_filenames_all[i] for i in index]
        self.camera2world_all = [self.camera2world_all[i] for i in index]
        self.cameras_K_all = [self.cameras_K_all[i] for i in index]
        self.cameras_idx_all = [self.cameras_idx_all[i] for i in index]
        self.camera_times_all = [self.camera_times_all[i] for i in index]
        self.chassis2world_all = [self.chassis2world_all[i] for i in index]
        self.label_filenames_all = [self.label_filenames_all[i] for i in index]

    @property
    def label_remaps(self):
        return get_nusc_label_remaps()

    @property
    def origin_color_map(self):
        return get_nusc_origin_color_map()

    @property
    def num_class(self):
        return 7

    @property
    def filted_color_map(self):
        return get_nusc_filted_color_map()
