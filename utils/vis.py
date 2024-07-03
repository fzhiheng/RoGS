# -*- coding: UTF-8 -*-

import numpy as np
from mayavi import mlab
from pyrotation.conversion import matrix_from_euler_angle, matrix_from_quaternion

from models.gaussian_model import GaussianModel2D


def plot_gaussion_3d(figure, center=None, R=None, S=None, color=None, num=50, plot_axis=True):
    if center is None:
        center = np.array([0, 0, 0])
    if R is None:
        R = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
    if S is None:
        S = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
    scale_factor = np.trace(S) / 3
    if color is None:
        color = (0, 1, 0, 0.8)
    if len(color) == 3:
        color = tuple(list(color) + [0.8])

    u = np.linspace(0, 2 * np.pi, num)
    v = np.linspace(0, np.pi, num)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    xyz = np.stack((x, y, z), axis=-1)

    # ST @ RT @ R @ S
    A = R @ S  # （3，3）


    xyz = A[None, None, :, :] @ xyz[..., None]  # (100, 100, 3, 1）
    xyz = xyz[..., 0] + center  # (100, 100, 3）


    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    mlab.mesh(x, y, z, color=color[:3], opacity=color[3], figure=figure)

    if plot_axis:
        mlab.quiver3d(center[0], center[1], center[2], R[0, 0], R[1, 0], R[2, 0], color=(1, 0, 0), scale_factor=scale_factor, line_width=3, figure=figure)
        mlab.quiver3d(center[0], center[1], center[2], R[0, 1], R[1, 1], R[2, 1], color=(0, 1, 0), scale_factor=scale_factor, line_width=3, figure=figure)
        mlab.quiver3d(center[0], center[1], center[2], R[0, 2], R[1, 2], R[2, 2], color=(0, 0, 1), scale_factor=scale_factor, line_width=3, figure=figure)


def vis_gaussian_3d(guassian: GaussianModel2D, mask=None):
    xyz = guassian.get_xyz  # (M, 3)
    rotation = guassian.get_rotation  # (M, 4)
    rotation = matrix_from_quaternion(rotation)  # (M, 3, 3)
    scale = guassian.get_scaling  # (M, 3)
    rgbs = guassian.get_rgb
    opacity = guassian.get_opacity

    xyz = xyz.detach().cpu().numpy()
    rotation = rotation.detach().cpu().numpy()
    scale = scale.detach().cpu().numpy()
    rgbs = rgbs.detach().cpu().numpy()
    if rgbs.shape[0] == 0:
        rgbs = np.zeros_like(xyz)
        rgbs[:, 1] = 1

    if mask is not None:
        mask = mask.detach().cpu().numpy()
        xyz = xyz[mask]
        rotation = rotation[mask]
        scale = scale[mask]
        rgbs = rgbs[mask]

    sample_idx = np.random.choice(range(xyz.shape[0]), 100)
    xyz = xyz[sample_idx]  # (100, 3)
    rotation = rotation[sample_idx]  # (100, 3, 3)
    scale = scale[sample_idx]  # (100, 3)
    rgbs = rgbs[sample_idx]

    fig = mlab.figure(bgcolor=(1, 1, 1), size=(800, 800))
    for r, center, s, color in zip(rotation, xyz, scale, rgbs):
        plot_gaussion_3d(figure=fig, R=r, center=center, S=np.diag(s), color=(color[0], color[1], color[2]))
    mlab.show()


if __name__ == "__main__":
    yaw = 0
    pitch = 0
    roll = np.pi / 9
    ypr = np.array([yaw, pitch, roll])

    R = matrix_from_euler_angle(ypr)
    S = np.array([[1, 0, 0],
                  [0, 2, 0],
                  [0, 0, 0]])
    fig = mlab.figure(bgcolor=(1, 1, 1), size=(800, 800))
    plot_gaussion_3d(fig, R=R, S=S)
    mlab.show()
