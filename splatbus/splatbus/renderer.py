"""IPC renderer for Gaussian Splatting."""

from typing import List
import torch

from splatbus.camera import IPCCamera

from .core.ipc_channel import IPCSocketServer
from .core.ipc_handles import IPCHandleManager
from .core.message_channel import MessageSocketServer
from .core.shared_buffer import SharedBuffer, format_image_for_shared_buffer


class GaussianSplattingIPCRenderer:
    def __init__(
        self,
        width: int,
        height: int,
        ipc_host: str = "127.0.0.1",
        ipc_port: int = 6001,
        msg_host: str = "127.0.0.1",
        msg_port: int = 6000,
    ) -> None:
        self.width = width
        self.height = height

        # Set True to force a synthetic depth pattern (useful for Unity-side debugging).
        self.use_test_depth = False

        self.ipc_server = IPCSocketServer(host=ipc_host, port=ipc_port)
        self.msg_server = MessageSocketServer(host=msg_host, port=msg_port)
        self.color_buffer = SharedBuffer(self.width, self.height, channels=4)
        self.depth_buffer = SharedBuffer(self.width, self.height, channels=1)
        self.frame_idx_buffer = SharedBuffer(1, 1, channels=1)
        self.ipc = IPCHandleManager(self.color_buffer, self.depth_buffer, self.frame_idx_buffer)

        # Wait for Unity client and send the initial packet
        color_buffer_info = self.color_buffer.get_info()
        depth_buffer_info = self.depth_buffer.get_info()
        frame_idx_buffer_info = self.frame_idx_buffer.get_info()
        ipc_handles_info = self.ipc.get_handle()
        device_index = self.color_buffer.buffer.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        self.ipc_server.set_ipc_init(
            color_buffer_info, depth_buffer_info, frame_idx_buffer_info, ipc_handles_info, device=device_index
        )

    def update_frame(
        self,
        color_data: torch.Tensor,
        depth_data: torch.Tensor,
        frame_idx: int = 0,
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
        self.frame_idx_buffer.buffer.fill_(float(frame_idx))
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
    
    def update_frame_info(self, ts_dict):
        self.msg_server.update_frame_info(dict(ts_dict))

    def close(self):
        self.ipc_server.close_socket()
        self.msg_server.close_socket()


class PinnedBuffer:
    def __init__(self, width: int, height: int, channels: int):
        self.width = width
        self.height = height
        self.channels = channels
        self.buffer = torch.zeros(height, width, channels, dtype=torch.float32, pin_memory=True)

class GaussianSplattingEncodedStreamRenderer:
    def __init__(
        self,
        width: int,
        height: int,
        ipc_host: str = "127.0.0.1",
        ipc_port: int = 6002,
        msg_host: str = "127.0.0.1",
        msg_port: int = 6000,
    ) -> None:
        self.width = width
        self.height = height

        # Set True to force a synthetic depth pattern (useful for Unity-side debugging).
        self.use_test_depth = False

        self.ipc_server = IPCSocketServer(host=ipc_host, port=ipc_port)
        self.msg_server = MessageSocketServer(host=msg_host, port=msg_port)

        self.pin_color_buffer = PinnedBuffer(width, height, 4)
        self.pin_depth_buffer = PinnedBuffer(width, height, 1)

    def update_frame(
        self,
        color_data: torch.Tensor,
        depth_data: torch.Tensor,
        frame_idx: int = 0,
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
        
        formatted_color_data = format_image_for_shared_buffer(
            color_data,
            channels=4,
            inverse=False,
        )
        formatted_depth_data = format_image_for_shared_buffer(
            depth_data,
            channels=1,
            inverse=inverse_depth,
        )
        self.pin_color_buffer.buffer.copy_(formatted_color_data, non_blocking=True)
        self.pin_depth_buffer.buffer.copy_(formatted_depth_data, non_blocking=True)
        torch.cuda.synchronize()
        self.msg_server.update_frame(self.pin_color_buffer.buffer, self.pin_depth_buffer.buffer, frame_idx)


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
    
    def update_frame_info(self, ts_dict):
        self.msg_server.update_frame_info(dict(ts_dict))

    def close(self):
        self.ipc_server.close_socket()
        self.msg_server.close_socket()

