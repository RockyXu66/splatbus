import math

from copy import deepcopy
import numpy as np
import torch


def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)


def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def getProjectionMatrixCenterShift(znear, zfar, cx, cy, fl_x, fl_y, w, h):
    top = cy / fl_y * znear
    bottom = -(h - cy) / fl_y * znear

    left = -(w - cx) / fl_x * znear
    right = cx / fl_x * znear

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


class IPCCamera:
    def __init__(
        self,
        width,
        height,
        R,
        t,
        fov_x,
        fov_y,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        timestamp=0.0,
        cx=-1,
        cy=-1,
        fl_x=-1,
        fl_y=-1,
        # resolution=None,
        zfar=100.0,
        znear=0.01,
    ):

        self.R = R
        self.t = t
        self.fov_x = fov_x
        self.fov_y = fov_y
        self.cx = cx
        self.cy = cy
        self.fl_x = fl_x
        self.fl_y = fl_y
        self.zfar = zfar
        self.znear = znear
        self.trans = trans
        self.scale = scale
        self.timestamp = timestamp
        self.image_width, self.image_height = width, height
        self.data_device = torch.device("cpu")

        self.world_view_transform = torch.tensor(
            getWorld2View2(R, t, trans, scale)
        ).transpose(0, 1)
        if cx > 0:
            self.projection_matrix = getProjectionMatrixCenterShift(
                self.znear,
                self.zfar,
                cx,
                cy,
                fl_x,
                fl_y,
                self.image_width,
                self.image_height,
            ).transpose(0, 1)
        else:
            self.projection_matrix = getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.fov_x, fovY=self.fov_y
            ).transpose(0, 1)
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    @property
    def FoVx(self):
        return self.fov_x

    @property
    def FoVy(self):
        return self.fov_y

    @staticmethod
    def init_from_view(width: int, height: int, view: object) -> "IPCCamera":
        if not hasattr(view, "R") or not hasattr(view, "T"):
            raise ValueError("View object must have R and T attributes")
        fov_x, fov_y = None, None
        cx, cy, fl_x, fl_y = -1, -1, -1, -1
        if hasattr(view, "FoVx") and hasattr(view, "FoVy"):
            fov_x, fov_y = getattr(view, "FoVx"), getattr(view, "FoVy")
        if (
            hasattr(view, "cx")
            and hasattr(view, "cy")
            and hasattr(view, "fl_x")
            and hasattr(view, "fl_y")
        ):
            cx = getattr(view, "cx")
            cy = getattr(view, "cy")
            fl_x = getattr(view, "fl_x")
            fl_y = getattr(view, "fl_y")
        return IPCCamera(
            width,
            height,
            R=view.R,
            t=view.T,
            fov_x=fov_x,
            fov_y=fov_y,
            scale=getattr(view, "scale", 1.0),
            trans=getattr(view, "trans", np.array([0.0, 0.0, 0.0])),
            timestamp=getattr(view, "timestamp", 0.0),
            cx=cx,
            cy=cy,
            fl_x=fl_x,
            fl_y=fl_y,
            zfar=getattr(view, "zfar", 100.0),
            znear=getattr(view, "znear", 0.01),
        )

    def set_rt(self, R: np.ndarray, t: np.ndarray, trans=np.array([0.0, 0.0, 0.0])):
        assert R.shape == (3, 3), f"R should be 3x3, got {R.shape}"
        assert t.shape == (3,), f"t should be 3-dimensional, got {t.shape}"
        self.world_view_transform = (
            torch.tensor(getWorld2View2(R, t, self.trans + trans, self.scale))
            .transpose(0, 1)
        )
        self.full_proj_transform = (
            (
                self.world_view_transform.unsqueeze(0).bmm(
                    self.projection_matrix.unsqueeze(0)
                )
            )
            .squeeze(0)
        )
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    
    def cuda(self) -> "IPCCamera":
        cuda_copy = deepcopy(self)
        for k, v in cuda_copy.__dict__.items():
            if isinstance(v, torch.Tensor):
                cuda_copy.__dict__[k] = v.to(cuda_copy.data_device)
        return cuda_copy


