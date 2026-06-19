"""IPC renderer for Gaussian Splatting."""

from typing import List, Optional
import torch

from splatbus.camera import IPCCamera

from .core.ipc_channel import IPCSocketServer
from .core.ipc_handles import IPCHandleManager
from .core.message_channel import MessageSocketServer
from .core.shared_buffer import SharedBuffer


class GaussianSplattingIPCRenderer:
    def __init__(
        self,
        width: int,
        height: int,
        ipc_host: str = "127.0.0.1",
        ipc_port: int = 6001,
        msg_host: str = "127.0.0.1",
        msg_port: int = 6000,
        gaussian_max_points: int = 0,
    ) -> None:
        self.width = width
        self.height = height
        self.gaussian_max_points = gaussian_max_points

        # Set True to force a synthetic depth pattern (useful for Unity-side debugging).
        self.use_test_depth = False

        self.ipc_server = IPCSocketServer(host=ipc_host, port=ipc_port)
        self.msg_server = MessageSocketServer(host=msg_host, port=msg_port)
        self.color_buffer = SharedBuffer(self.width, self.height, channels=4)
        self.depth_buffer = SharedBuffer(self.width, self.height, channels=1)

        self.gaussians_buffer = None
        if self.gaussian_max_points > 0:
            self.gaussians_buffer = SharedBuffer(
                width=7, height=self.gaussian_max_points, channels=1
            )

        self.ipc = IPCHandleManager(
            self.color_buffer, self.depth_buffer,
            gaussian_buffer=self.gaussians_buffer,
        )

        # Wait for Unity client and send the initial packet
        color_buffer_info = self.color_buffer.get_info()
        depth_buffer_info = self.depth_buffer.get_info()
        ipc_handles_info = self.ipc.get_handle()
        device_index = self.color_buffer.buffer.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()

        gaussian_buffer_info = None
        if self.gaussians_buffer is not None:
            gaussian_buffer_info = self.gaussians_buffer.get_info()

        self.ipc_server.set_ipc_init(
            color_buffer_info, depth_buffer_info, ipc_handles_info,
            device=device_index,
            gaussian_buffer_info=gaussian_buffer_info,
        )

    def update_frame(
        self,
        color_data: torch.Tensor,
        depth_data: torch.Tensor,
        inverse_depth: bool = True,
    ):
        """
        Update IPC buffers with new frame data

        Args:
            color_data: RGB/RGBA color image [C, H, W]
            depth_data: Inverse depth from Gaussian Splatting [1, H, W]
        """
        if self.use_test_depth:
            inverse_depth = False

            width = depth_data.shape[2]
            third = width // 3

            depth_data[:, :, :third] = 0.1  # Left third: VERY NEAR (0.1m)
            depth_data[:, :, third : 2 * third] = 5.0  # Middle third: MEDIUM (5m)
            depth_data[:, :, 2 * third :] = 100.0  # Right third: VERY FAR (100m)

        self.color_buffer.update(color_data, inverse=False)
        self.depth_buffer.update(depth_data, inverse=inverse_depth)
        self.ipc.record_event()

    # TODO: Move all camera-related things *out* of msg_server into here. They don't
    # belong there.
    def update_view(self, view):
        self.msg_server.update_view(view)

    def init_view(self, width: int, height: int, view: object):
        self.msg_server.init_view(IPCCamera.init_from_view(width, height, view))
    
    def set_cam_list(self, width: int, height: int, views: List[object]):
        self.msg_server.init_cam_list([IPCCamera.init_from_view(width, height, view) for view in views])

    def get_current_view(self):
        return self.msg_server.viewpoint

    def update_gaussians(self, gaussians):
        self.msg_server.update_gaussians(gaussians)

    def update_rgb_points(self, xyz, rgb):
        self.msg_server.update_rgb_points(xyz, rgb)
        if self.gaussians_buffer is not None and xyz is not None:
            self._write_gaussians_shared(xyz, rgb)

    def _write_gaussians_shared(self, xyz, rgb):
        """Write gaussian positions and colors to the shared GPU buffer."""
        if isinstance(xyz, torch.Tensor):
            xyz_gpu = xyz.detach().float().cuda()
        else:
            xyz_gpu = torch.from_numpy(xyz).float().cuda()
        n = xyz_gpu.shape[0]
        n = min(n, self.gaussian_max_points)

        if rgb is not None:
            if isinstance(rgb, torch.Tensor):
                rgb_gpu = rgb.detach().float().cuda()
            else:
                rgb_gpu = torch.from_numpy(rgb).float().cuda()
        else:
            rgb_gpu = torch.ones((n, 3), dtype=torch.float32, device='cuda')

        buf = self.gaussians_buffer.buffer  # (max_points, 7, 1)

        buf[:n, :3, 0] = xyz_gpu[:n]
        buf[:n, 3:6, 0] = rgb_gpu[:n]
        buf[:n, 6, 0] = 1.0
        if n < self.gaussian_max_points:
            buf[n:, 6, 0] = 0.0

        self.ipc.record_gaussian_event()

    def close(self):
        self.ipc_server.close_socket()
        self.msg_server.close_socket()