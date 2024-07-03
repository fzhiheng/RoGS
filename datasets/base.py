import os

import cv2
import numpy as np
from multiprocessing.pool import ThreadPool as Pool

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self):
        self.base_dir = ""
        self.image_filenames = []  # list of image relative path w.r.t to self.base_dir
        self.label_filenames = []  # list of label relative path w.r.t to self.base_dir
        self.ref_camera2world = []  # list of 4x4 ndarray camera2world transform
        self.cameras_K = []  # list of 3x3 ndarray camera intrinsics
        self.cameras_d = []  # list of camera distortion coefficients
        self.cameras_idx = []  # list of camera idx
        self.camera_times_all = []  # list of camera timestamp

        self.camera2world_all = []  # list of 4x4 ndarray camera2world transform
        self.chassis2world_all = []  # list of 4x4 ndarray camera2world transform
        self.image_filenames_all = []  # list of image relative path w.r.t to self.base_dir
        self.label_filenames_all = []  # list of label relative path w.r.t to self.base_dir
        self.lane_filenames_all = []  # list of lane relative path w.r.t to self.base_dir
        self.ref_camera2world_all = []  # list of 4x4 ndarray camera2world transform
        self.cameras_K_all = []  # list of 3x3 ndarray camera intrinsics
        self.cameras_d_all = []  # list of camera distortion coefficients
        self.cameras_idx_all = []  # list of camera idx

    def __len__(self):
        return len(self.image_filenames)

    def getNerfppNorm(self):
        def get_center_and_diag(cam_centers):
            cam_centers = np.hstack(cam_centers)
            avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
            center = avg_cam_center
            dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
            diagonal = np.max(dist)
            return center.flatten(), diagonal

        cam_centers = [pose[:3, 3:4] for pose in self.camera2world_all]

        center, diagonal = get_center_and_diag(cam_centers)
        radius = diagonal * 1.1
        translate = -center
        return {"translate": translate, "radius": radius}

    def filter_by_index(self, index):
        self.image_filenames_all = [self.image_filenames_all[i] for i in index]
        self.label_filenames_all = [self.label_filenames_all[i] for i in index]
        self.lane_filenames_all = [self.lane_filenames_all[i] for i in index]
        self.ref_camera2world_all = [self.ref_camera2world_all[i] for i in index]
        self.cameras_K_all = [self.cameras_K_all[i] for i in index]
        self.cameras_d_all = [self.cameras_d_all[i] for i in index]
        self.cameras_idx_all = [self.cameras_idx_all[i] for i in index]
        if hasattr(self, "depth_filenames_all"):
            self.depth_filenames_all = [self.depth_filenames_all[i] for i in index]

    @staticmethod
    def file_valid(file_name):
        if os.path.exists(file_name) and (os.path.getsize(file_name) != 0):
            return True
        else:
            return False

    @staticmethod
    def check_filelist_exist(filelist):
        with Pool(32) as p:
            exist_list = p.map(BaseDataset.file_valid, filelist)
        return exist_list

    def remap_semantic(self, semantic_label):
        semantic_label = semantic_label.astype('uint8')
        remaped_label = np.array(cv2.LUT(semantic_label, self.label_remaps))
        return remaped_label
