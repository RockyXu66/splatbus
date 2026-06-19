import base64
import math
import socket
import warnings
from typing import Dict, Optional, List

import numpy as np
import torch
from loguru import logger
from scipy.spatial.transform import Rotation as SciRot

from splatbus.camera import IPCCamera

from .BaseSocket import BaseSocketServer
from .utils import quat_mul, quat_to_rot, unity_pose_to_gs


class MessageSocketServer(BaseSocketServer):
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 6000,
        flip_y: bool = False,
        flip_z: bool = False,
    ) -> None:
        self._viewpoint: Optional[IPCCamera] = None
        self._cam_list: Optional[List[IPCCamera]] = None
        self._cam_pose: Dict[str, Dict[str, float]] = {}
        self._point_cloud_pose = None
        self._timestamp_override: Optional[int] = None
        self.gaussians_xyz_ori = None
        self.gaussians_rotation_ori = None
        self.gaussians_xyz = None
        self.gaussians_color = None
        self.flip_y = flip_y
        self.flip_z = flip_z
        super().__init__(host=host, port=port, server_name="MessageSocketServer")

    def on_client_connected(self, conn: socket.socket, addr):
        self._set_active_connection(conn)
        try:
            with conn:
                self._recv_loop(conn)
        finally:
            self._clear_active_connection()

    @property
    def cam_pose(self) -> Dict[str, Dict[str, float]]:
        return self._cam_pose

    @property
    def point_cloud_pose(self):
        return self._point_cloud_pose

    @property
    def timestamp_override(self) -> Optional[int]:
        return self._timestamp_override

    def send_message(self, payload: dict):
        self.send_json(payload)

    def update_view(self, view):
        warnings.warn("update_view is deprecated. Please use the new IPCCamera object.")
        cam_pose = self.cam_pose
        if cam_pose:
            pos = cam_pose["position"]
            rot = cam_pose["rotation"]

            camera_center, world_view_transform = unity_pose_to_gs(
                [pos["x"], pos["y"], pos["z"]],
                [rot["x"], rot["y"], rot["z"], rot["w"]],
                flip_y=self.flip_y,
                flip_z=self.flip_z,
            )

            view.camera_center = torch.from_numpy(camera_center).to(view.data_device)
            view.world_view_transform = torch.from_numpy(world_view_transform).to(
                view.data_device
            )

            view.full_proj_transform = (
                view.world_view_transform @ view.projection_matrix.to(view.data_device)
            )

    def _update_viewpoint_from_cam_pose(self):
        cam_pose = self.cam_pose
        if cam_pose:
            pos = cam_pose["position"]  # camera to world position
            rot = cam_pose["rotation"]  # camera to world quaternion rotation
            t = np.array([[pos["x"], pos["y"], pos["z"]]], dtype=np.float32).reshape(3)
            R = quat_to_rot([float(x) for x in rot.values()])
            if self._viewpoint is None:
                raise ValueError("Viewpoint is not initialized. Please call init_view() first.")
            
            # We need world to camera transform to update the viewpoint.
            camera_to_world = np.eye(4)
            camera_to_world[:3, :3] = R
            camera_to_world[:3, 3] = t
            world_to_camera = np.linalg.inv(camera_to_world)
            self._viewpoint.set_rt(R=world_to_camera[:3, :3].T, t=world_to_camera[:3, 3])

            # Update intrinsics if the client provided them.  Prefer full
            # pinhole intrinsics (fl_x/fl_y/cx/cy in pixels, COLMAP PINHOLE
            # model) over plain FOV.
            fl_x = cam_pose.get("fl_x")
            fl_y = cam_pose.get("fl_y")
            cx = cam_pose.get("cx")
            cy = cam_pose.get("cy")
            intr_w = cam_pose.get("intr_width")
            intr_h = cam_pose.get("intr_height")
            if fl_x is not None and fl_y is not None and cx is not None and cy is not None:
                srv_w = self._viewpoint.image_width
                srv_h = self._viewpoint.image_height
                # Scale intrinsics from the client's resolution to the server's
                # render resolution so the FOV and principal point are preserved.
                if intr_w is not None and intr_h is not None and (intr_w != srv_w or intr_h != srv_h):
                    sx = srv_w / float(intr_w)
                    sy = srv_h / float(intr_h)
                    fl_x = fl_x * sx
                    fl_y = fl_y * sy
                    cx = cx * sx
                    cy = cy * sy
                    logger.info(f"[Server] scaled intrinsics from {intr_w}x{intr_h} to {srv_w}x{srv_h}: fl_x={fl_x:.2f}, fl_y={fl_y:.2f}, cx={cx:.2f}, cy={cy:.2f}")
                else:
                    logger.info(f"[Server] received intrinsics: fl_x={fl_x:.2f}, fl_y={fl_y:.2f}, cx={cx:.2f}, cy={cy:.2f}")
                self._viewpoint.set_intrinsics(
                    float(fl_x), float(fl_y), float(cx), float(cy),
                    width=srv_w,
                    height=srv_h,
                )
            else:
                fov_x = cam_pose.get("fov_x")
                fov_y = cam_pose.get("fov_y")
                if fov_x is not None:
                    if fov_y is None:
                        w = self._viewpoint.image_width
                        h = self._viewpoint.image_height
                        fov_y = 2.0 * math.atan(math.tan(fov_x / 2.0) * (h / w))
                    self._viewpoint.set_fov(float(fov_x), float(fov_y))

    def init_view(self, view: IPCCamera):
        if not isinstance(view, IPCCamera):
            raise TypeError("view must be an instance of IPCCamera")
        self._viewpoint = view

    def init_cam_list(self, cam_list: List[IPCCamera]):
        if not all(isinstance(cam, IPCCamera) for cam in cam_list):
            raise TypeError("cam_list must be a list of IPCCamera objects")
        self._cam_list = cam_list

    @property
    def viewpoint(self) -> IPCCamera:
        if self._viewpoint is None:
            raise ValueError("Viewpoint is not initialized. Please call init_view() first.")
        return self._viewpoint

    def update_gaussians(self, gaussians):
        if self.gaussians_xyz_ori is None:
            self.gaussians_xyz_ori = gaussians._xyz.clone()
        if self.gaussians_rotation_ori is None:
            self.gaussians_rotation_ori = gaussians._rotation.clone()

        point_cloud_pose = self.point_cloud_pose
        if point_cloud_pose:
            pos = point_cloud_pose["position"]
            rot = point_cloud_pose["rotation"]

            unity_rot = SciRot.from_quat([rot["x"], rot["y"], rot["z"], rot["w"]])
            unity_rot_mat = (
                torch.from_numpy(unity_rot.as_matrix())
                .float()
                .to(gaussians._xyz.device)
            )
            tmp_xyz = (unity_rot_mat @ self.gaussians_xyz_ori.T).T

            qx, qy, qz, qw = unity_rot.as_quat()
            qg = torch.tensor([qw, qx, qy, qz], device=gaussians._rotation.device)
            gaussians._rotation = quat_mul(qg, self.gaussians_rotation_ori)

            gaussians._xyz = tmp_xyz + torch.from_numpy(
                np.array([pos["x"], pos["y"], pos["z"]])
            ).float().to(gaussians._xyz.device)

        # Cache current posed positions and base RGB color for the Blender client.
        self.gaussians_xyz = gaussians._xyz.detach().clone()
        features_dc = getattr(gaussians, "_features_dc", None)
        if features_dc is not None:
            color = torch.sigmoid(features_dc.detach().clone())
            if color.dim() == 3 and color.shape[1] == 1:
                color = color.squeeze(1)
            self.gaussians_color = color
        else:
            self.gaussians_color = None

    def update_rgb_points(self, xyz, rgb):
        if isinstance(xyz, np.ndarray):
            xyz = torch.from_numpy(xyz).float()
        if isinstance(rgb, np.ndarray):
            rgb = torch.from_numpy(rgb).float()
        self.gaussians_xyz = xyz.detach().clone()
        self.gaussians_color = rgb.detach().clone() if rgb is not None else None

    def _handle_payload(self, payload: dict):
        if payload.get("type") == "camera_pose":
            self._cam_pose = payload
            self._timestamp_override = payload.get("timestamp_index", None)
            self._update_viewpoint_from_cam_pose()
        elif payload.get("type") == "point_cloud_pose":
            self._point_cloud_pose = payload
        elif payload.get("type") == "get_viewport_size":
            if self._viewpoint is not None:
                w, h = self._viewpoint.image_width, self._viewpoint.image_height
            else:
                w, h = 0, 0
            self.send_message({"viewport_size": (w, h)})
        elif payload.get("type") == "get_camera_pose":
            cam_idx = payload.get("cam_idx", 0)
            if self._cam_list is not None:
                cam: IPCCamera = self._cam_list[cam_idx%len(self._cam_list)]
                t = cam.camera_center.cpu().numpy()
                R = cam.R
                quat = SciRot.from_matrix(R).as_quat()
            else:
                t = np.array([0.0, 0.0, 0.0])
                quat = np.array([0.0, 0.0, 0.0, 1.0])
            self.send_message({"position": t.tolist(), "rotation": quat.tolist()})
        elif payload.get("type") == "get_camera_info":
            cam_idx = payload.get("cam_idx", 0)
            if self._cam_list is not None:
                cam: IPCCamera = self._cam_list[cam_idx % len(self._cam_list)]
                fov_x = float(cam.FoVx)
                fov_y = float(cam.FoVy)
                width = int(cam.image_width)
                height = int(cam.image_height)
            else:
                fov_x = fov_y = 1.0
                width = height = 0
            self.send_message({
                "fov_x": fov_x,
                "fov_y": fov_y,
                "width": width,
                "height": height,
            })
        elif payload.get("type") == "timestamp":
            self._timestamp_override = payload.get("timestamp_index", None)
        elif payload.get("type") == "get_gaussians":
            gaussians_xyz = self.gaussians_xyz
            gaussians_color = self.gaussians_color
            if gaussians_xyz is None:
                self.send_message({"count": 0, "positions": "", "colors": ""})
            else:
                xyz = gaussians_xyz.cpu().numpy().astype(np.float32)
                positions_b64 = base64.b64encode(xyz.tobytes()).decode("utf-8")
                if gaussians_color is not None:
                    rgb = gaussians_color.cpu().numpy().astype(np.float32)
                    colors_b64 = base64.b64encode(rgb.tobytes()).decode("utf-8")
                else:
                    colors_b64 = ""
                self.send_message({
                    "count": xyz.shape[0],
                    "positions": positions_b64,
                    "colors": colors_b64,
                })
        else:
            logger.debug(
                f"[MessageSocketServer] Unknown payload type: {payload.get('type')}"
            )
