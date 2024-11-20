import torch
import numpy as np
import torch.nn as nn


class ExposureModel(nn.Module):
    def __init__(self, num_camera=None):
        super().__init__()
        self.num_camera = num_camera
        self.exposure_a = nn.Parameter(torch.zeros(size=(num_camera, 1), dtype=torch.float32))  # (N, 3) axis angle
        self.exposure_b = nn.Parameter(torch.zeros(size=(num_camera, 1), dtype=torch.float32))

    def forward(self, idx, image):
        if self.num_camera == 1:
            return image
        exposure_a = self.exposure_a[idx]
        exposure_b = self.exposure_b[idx]
        image = torch.exp(exposure_a) * image + exposure_b
        image = image.clamp(0, 1)
        return image

    def apply_numpy(self, idx, image):
        if self.num_camera == 1:
            return image
        exposure_a = self.exposure_a[idx].detach().cpu().numpy()
        exposure_b = self.exposure_b[idx].detach().cpu().numpy()
        image = np.exp(exposure_a) * image + exposure_b
        image = np.clip(image, 0, 1)
        return image
