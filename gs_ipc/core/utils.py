import numpy as np

import torch
from scipy.spatial.transform import Rotation as R

def quat_to_rot(q):
    # q = (x, y, z, w) from Unity
    x, y, z, w = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1-2*(yy+zz),   2*(xy-wz),     2*(xz+wy)],
        [2*(xy+wz),     1-2*(xx+zz),   2*(yz-wx)],
        [2*(xz-wy),     2*(yz+wx),     1-2*(xx+yy)],
    ], dtype=np.float32)

def unity_pose_to_gs(pos, quat, flip_y=False, flip_z=False):
    R = quat_to_rot(quat)
    t = np.array(pos, dtype=np.float32).reshape(3, 1)

    # camera_to_world
    T_u = np.eye(4, dtype=np.float32)
    T_u[:3, :3] = R
    T_u[:3, 3] = t[:, 0]

    # world_to_camera (Unity view)
    V_u = np.linalg.inv(T_u)

    # coord convert
    C = np.eye(4, dtype=np.float32)
    C[1, 1] = -1.0 if flip_y else 1.0
    C[2, 2] = -1.0 if flip_z else 1.0

    gs_view = C @ V_u @ C
    world_view_transform = gs_view.T  # row-major rows 0..3

    # camera center in GS/world coords
    Rg = gs_view[:3, :3]
    tg = gs_view[:3, 3]
    camera_center = -Rg.T @ tg  # same as inv(gs_view)[:3,3]

    return camera_center, world_view_transform

# scipy: (x, y, z, w) -> (w, x, y, z)
def quat_mul(qg, qi):
    w0, x0, y0, z0 = qg
    w1, x1, y1, z1 = qi.unbind(-1)
    return torch.stack([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1,
    ], dim=-1)