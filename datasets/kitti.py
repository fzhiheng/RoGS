import os
from copy import deepcopy

import cv2
import numpy as np
from multiprocessing.pool import ThreadPool as Pool

from datasets.base import BaseDataset


def get_camxx2cam0(project_matrix, R0_rect):
    T_cam0tocamxx = np.eye(4)
    T_cam0tocamxx[:3, :3] = R0_rect
    T_cam0tocamxx[:3, 3:4] = np.linalg.inv(project_matrix[:3, :3]) @ project_matrix[:, 3:4]
    T_camxx2cam0 = np.linalg.inv(T_cam0tocamxx)
    return T_camxx2cam0


def read_kitti_calib(calib_path):
    with open(calib_path, "r") as f:
        lines = f.readlines()
    P2 = np.array([float(x) for x in lines[2].strip().split(" ")[1:]]).reshape(3, 4)
    P3 = np.array([float(x) for x in lines[3].strip().split(" ")[1:]]).reshape(3, 4)
    R0_rect = np.array([float(x) for x in lines[4].strip().split(" ")[1:]]).reshape(3, 3)
    Tr_velo2cam0 = np.array([float(x) for x in lines[5].strip().split(" ")[1:]]).reshape(3, 4)
    Tr_imu2velo = np.array([float(x) for x in lines[6].strip().split(" ")[1:]]).reshape(3, 4)

    Tr_velo2cam0 = np.vstack([Tr_velo2cam0, np.array([0, 0, 0, 1])])
    Tr_imu2velo = np.vstack([Tr_imu2velo, np.array([0, 0, 0, 1])])

    K2 = P2[:3, :3]
    K3 = P3[:3, :3]
    Tr_cam2tocam0 = get_camxx2cam0(P2, R0_rect)
    Tr_cam3tocam0 = get_camxx2cam0(P3, R0_rect)
    calib = {"K2": K2, "K3": K3, "T_cam2tocam0": Tr_cam2tocam0, "T_cam3tocam0": Tr_cam3tocam0, "Tr_velo2cam0": Tr_velo2cam0, "Tr_imu2velo": Tr_imu2velo}

    return calib


IMU_HEIGHT = 0.93
VEL_HEIGHT = 1.73


class KittiDataset(BaseDataset):
    def __init__(self, configs, use_label, use_depth):
        super().__init__()
        self.resized_image_size = (configs["image_width"], configs["image_height"])
        self.base_dir = configs["base_dir"]
        self.image_dir = configs["image_dir"]
        self.pose_dir = configs["pose_dir"]
        self.sequence = configs["sequence"]
        self.camera_names = configs["camera_names"]  # image_2 or image_3

        calib_file = os.path.join(self.base_dir, "sequences", self.sequence, "calib", "000000.txt")
        calib = read_kitti_calib(calib_file)
        self.intrinsic = np.asarray([
            [7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02],
            [0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02],
            [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]
        ], dtype=np.float32)  # image_2
        self.cam0tocam2 = np.asarray([[9.999758e-01, -5.267463e-03, -4.552439e-03, 5.956621e-02],
                                      [5.251945e-03, 9.999804e-01, -3.413835e-03, 2.900141e-04],
                                      [4.570332e-03, 3.389843e-03, 9.999838e-01, 2.577209e-03],
                                      [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        self.cam0tocam3 = np.asarray([[9.995599e-01, 1.699522e-02, -2.431313e-02, -4.731050e-01],
                                      [-1.704422e-02, 9.998531e-01, -1.809756e-03, 5.551470e-03],
                                      [2.427880e-02, 2.223358e-03, 9.997028e-01, -5.250882e-03],
                                      [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        cam3tocam0 = np.linalg.inv(self.cam0tocam3)
        cam2tocam0 = np.linalg.inv(self.cam0tocam2)
        self.camxx2cam0 = {"image_2": cam2tocam0, "image_3": cam3tocam0}

        # self.camxx2cam0 = {"image_2": calib["T_cam2tocam0"], "image_3": calib["T_cam3tocam0"]}

        pose_file = os.path.join(self.pose_dir, "dataset", "poses", f"{self.sequence}.txt")
        cam0_poses = np.loadtxt(pose_file).reshape((-1, 3, 4))  # (N, 3, 4)
        cam0_poses = np.concatenate([cam0_poses, np.repeat(np.array([[[0, 0, 0, 1]]]), cam0_poses.shape[0], axis=0)], axis=1)
        vel_pose = cam0_poses @ calib["Tr_velo2cam0"][None]
        imu_pose = vel_pose @ calib["Tr_imu2velo"][None]
        Tr_car2vel = np.eye(4)
        Tr_car2vel[2, 3] = -VEL_HEIGHT
        chassis_pose = vel_pose @ Tr_car2vel[None]

        self.ref_pose = chassis_pose[0]
        inv_ref_pose = np.linalg.inv(self.ref_pose)
        chassis2world_unique = inv_ref_pose @ chassis_pose
        cam0_poses = inv_ref_pose @ cam0_poses

        chassis_xy = chassis2world_unique[:, :2, 3]  # (N, 2)

        index = cam0_poses.shape[0] + 1
        if configs["max_x_length"] > 0:
            indexs = np.where(chassis_xy[:, 0] > configs["max_x_length"])[0]
            if len(indexs) > 0:
                index = min(indexs[0], index)
        if configs["max_y_length"] > 0:
            indexs = np.where(chassis_xy[:, 1] > configs["max_y_length"])[0]
            if len(indexs) > 0:
                index = min(indexs[0], index)

        self.chassis2world_unique = chassis2world_unique[:index]  # (N, 4, 4)

        for camera_idx, camera_name in enumerate(self.camera_names):
            file_names = os.listdir(os.path.join(self.base_dir, "sequences", self.sequence, camera_name))
            file_names.sort(key=lambda x: int(x[:-4]))

            image_paths = [os.path.join("sequences", self.sequence, camera_name, file_name) for file_name in file_names]
            label_paths = [os.path.join("seg_sequences", self.sequence, camera_name, file_name.replace(".jpg", ".png")) for file_name in file_names]
            camera2world = cam0_poses @ self.camxx2cam0[camera_name][None]  # (N, 4, 4)

            image_paths = image_paths[:index]
            label_paths = label_paths[:index]
            camera2world = camera2world[:index]

            num_cameras = len(image_paths)
            self.camera2world_all.extend([pose for pose in camera2world])
            self.cameras_K_all.extend([self.intrinsic] * num_cameras)
            self.cameras_idx_all.extend([camera_idx] * num_cameras)
            self.image_filenames_all.extend(image_paths)
            self.label_filenames_all.extend(label_paths)

        self.file_check()

        self.camera2world_all = np.array(self.camera2world_all)  # [N, 4, 4]
        self.road_pointcloud = None

        nerf_normalization = self.getNerfppNorm()
        self.cameras_extent = nerf_normalization["radius"]

    def __len__(self):
        return len(self.image_filenames_all)

    def file_check(self):
        image_paths = [os.path.join(self.base_dir, image_path) for image_path in self.image_filenames_all]
        label_paths = [os.path.join(self.image_dir, label_path) for label_path in self.label_filenames_all]
        image_exists = np.asarray(self.check_filelist_exist(image_paths))
        label_exists = np.asarray(self.check_filelist_exist(label_paths))
        available_index = list(np.where(image_exists * label_exists)[0])
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
        self.label_filenames_all = [self.label_filenames_all[i] for i in index]
        self.camera2world_all = [self.camera2world_all[i] for i in index]
        self.cameras_K_all = [self.cameras_K_all[i] for i in index]
        self.cameras_idx_all = [self.cameras_idx_all[i] for i in index]

    def label2mask(self, label):
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
                         | ((16 <= label) & (label <= 22)) | ((25 <= label) & (label <= 28)) | ((30 <= label) & (label <= 40)) | (label >= 42)
        # dilate itereation 2 for moving objects
        label_movable = label >= 52
        kernel = np.ones((10, 10), dtype=np.uint8)
        label_movable = cv2.dilate(label_movable.astype(np.uint8), kernel, 2).astype(np.bool)

        label_off_road = label_off_road | label_movable
        # label_off_road = label_movable
        mask[label_off_road] = 0
        label[~(mask.astype(np.bool))] = 64
        mask = mask.astype(np.float32)
        return mask, label

    def __getitem__(self, idx):
        cam_idx = self.cameras_idx_all[idx]
        cam2world = self.camera2world_all[idx]
        K = self.cameras_K_all[idx]

        image_path = os.path.join(self.base_dir, self.image_filenames_all[idx])
        image_name = os.path.basename(image_path).split(".")[0]
        input_image = cv2.imread(image_path)

        crop_cy = int(self.resized_image_size[1] * 0.4)
        origin_image_size = input_image.shape
        resized_image = cv2.resize(input_image, dsize=self.resized_image_size, interpolation=cv2.INTER_LINEAR)
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        resized_image = resized_image[crop_cy:, :, :]  # crop the sky
        gt_image = (np.asarray(resized_image) / 255.0).astype(np.float32)  # [H, W, 3]
        gt_image = np.clip(gt_image, 0.0, 1.0)
        width, height = gt_image.shape[1], gt_image.shape[0]

        label_path = os.path.join(self.image_dir, self.label_filenames_all[idx])
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        resized_label = cv2.resize(label, dsize=self.resized_image_size, interpolation=cv2.INTER_NEAREST)
        mask, label = self.label2mask(resized_label)
        label = self.remap_semantic(label).astype(int)
        mask = mask[crop_cy:, :]  # crop the sky
        label = label[crop_cy:, :]

        new_K = deepcopy(K)
        width_scale = self.resized_image_size[0] / origin_image_size[1]
        height_scale = self.resized_image_size[1] / origin_image_size[0]
        new_K[0, :] *= width_scale
        new_K[1, :] *= height_scale
        new_K[1, 2] -= crop_cy
        R = cam2world[:3, :3]
        T = cam2world[:3, 3]

        sample = {"image": gt_image, "idx": idx, "cam_idx": cam_idx, "mask": mask, "label": label, "image_name": image_name, "R": R, "T": T, "K": new_K,
                  "W": width, "H": height}
        return sample

    @property
    def label_remaps(self):
        colors = np.ones((256, 1), dtype="uint8")
        colors *= 4  # background
        colors[7, :] = 1  # Lane marking
        colors[8, :] = 1
        colors[14, :] = 1
        colors[23, :] = 1
        colors[24, :] = 1
        colors[2, :] = 2  # curb
        colors[9, :] = 2  # curb cut
        colors[41, :] = 3  # Manhole
        colors[13, :] = 3  # road
        return colors

    @property
    def origin_color_map(self):
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

    @property
    def num_class(self):
        return 5

    @property
    def filted_color_map(self):
        colors = np.zeros((256, 1, 3), dtype='uint8')
        colors[0, :, :] = [0, 0, 0]  # mask
        colors[1, :, :] = [0, 0, 255]  # all lane
        colors[2, :, :] = [255, 0, 0]  # curb
        colors[3, :, :] = [211, 211, 211]  # road and manhole
        colors[4, :, :] = [157, 234, 50]  # background
        return colors
