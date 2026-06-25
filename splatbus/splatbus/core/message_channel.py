import socket
import threading
import warnings
from collections import OrderedDict
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
        self.gaussians_xyz_ori = None
        self.gaussians_rotation_ori = None
        self.flip_y = flip_y
        self.flip_z = flip_z
        self._frame_info_history = OrderedDict()
        self._max_frame_info_history = 100

        # For encoded stream
        self.es_color_data = None
        self.es_depth_data = None
        self.es_frame_idx = None
        self._encoded_frame_lock = threading.Lock()
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
    
    def update_frame_info(self, ts_dict):
        frame_info = dict(ts_dict)
        frame_idx = frame_info.get("frame_idx")
        if frame_idx is None:
            return

        self._frame_info_history[int(frame_idx)] = frame_info
        self._frame_info_history.move_to_end(int(frame_idx))
        while len(self._frame_info_history) > self._max_frame_info_history:
            self._frame_info_history.popitem(last=False)

    def _handle_payload(self, payload: dict):
        if payload.get("type") == "camera_pose":
            self._cam_pose = payload
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
        elif payload.get("type") == "get_frame_info":
            frame_idx = payload.get("frame_idx", 0)
            ts_dict = self._frame_info_history.get(int(frame_idx))
            self.send_message({"frame_info": ts_dict})
        elif payload.get("type") == "get_encoded_stream_frame":
            with self._encoded_frame_lock:
                if (
                    self.es_color_data is None
                    or self.es_depth_data is None
                    or self.es_frame_idx is None
                ):
                    return
                color_data = self.es_color_data
                depth_data = self.es_depth_data
                frame_idx = self.es_frame_idx

            self.send_encoded_frame(
                frame_idx=frame_idx,
                width=color_data.shape[1],
                height=color_data.shape[0],
                color_nbytes=color_data.nbytes,
                depth_nbytes=depth_data.nbytes,
                color_data=color_data,
                depth_data=depth_data,
            )
        else:
            logger.debug(
                f"[MessageSocketServer] Unknown payload type: {payload.get('type')}"
            )

    def update_frame(self, color_data: torch.Tensor, depth_data: torch.Tensor, frame_idx: int = 0):
        color_snapshot = color_data.contiguous().numpy().copy()
        depth_snapshot = depth_data.contiguous().numpy().copy()

        with self._encoded_frame_lock:
            self.es_color_data = color_snapshot
            self.es_depth_data = depth_snapshot
            self.es_frame_idx = frame_idx