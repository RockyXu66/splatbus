import socket
import warnings
from typing import Dict, Optional

import numpy as np
import torch
from loguru import logger
from scipy.spatial.transform import Rotation as R

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
        self._cam_pose: Dict[str, Dict[str, float]] = {}
        self._point_cloud_pose = None
        self.gaussians_xyz_ori = None
        self.gaussians_rotation_ori = None
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
            pos = cam_pose["position"]
            rot = cam_pose["rotation"]
            t = np.array([[pos["x"], pos["y"], pos["z"]]], dtype=np.float32).reshape(3)
            R = quat_to_rot([float(x) for x in rot.values()])
            if self._viewpoint is None:
                raise ValueError("Viewpoint is not initialized. Please call init_view() first.")
            self._viewpoint.set_rt(R=R, t=t)

    def init_view(self, view: IPCCamera):
        if not isinstance(view, IPCCamera):
            raise TypeError("view must be an instance of IPCCamera")
        self._viewpoint = view

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

            unity_rot = R.from_quat([rot["x"], rot["y"], rot["z"], rot["w"]])
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

    def _handle_payload(self, payload: dict):
        if payload.get("type") == "camera_pose":
            self._cam_pose = payload
            self._update_viewpoint_from_cam_pose()
        elif payload.get("type") == "point_cloud_pose":
            self._point_cloud_pose = payload
        else:
            logger.debug(
                f"[MessageSocketServer] Unknown payload type: {payload.get('type')}"
            )
