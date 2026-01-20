"""IPC renderer for Gaussian Splatting."""

import torch

from .core.cuda_buffer import CUDAFrameBuffer
from .core.ipc_handles import IPCHandleManager
from .core.ipc_channel import IPCSocketServer
from .core.message_channel import MessageSocketServer


class GaussianSplattingIPCRenderer:
    def __init__(
        self,
        ipc_host: str = "127.0.0.1",
        ipc_port: int = 6001,
        msg_host: str = "127.0.0.1",
        msg_port: int = 6000,
    ) -> None:
        self.width = 532
        self.height = 948
        # self.width = 1064
        # self.height = 1895
        
        # Set True to force a synthetic depth pattern (useful for Unity-side debugging).
        self.use_test_depth = False

        self.ipc_server = IPCSocketServer(host=ipc_host, port=ipc_port)
        self.msg_server = MessageSocketServer(host=msg_host, port=msg_port)
        self.color_buffer = CUDAFrameBuffer(self.width, self.height, channels=4)
        self.depth_buffer = CUDAFrameBuffer(self.width, self.height, channels=1)
        self.ipc = IPCHandleManager(self.color_buffer, self.depth_buffer)

        # Wait for Unity client and send the initial packet
        color_buffer_info = self.color_buffer.get_info()
        depth_buffer_info = self.depth_buffer.get_info()
        ipc_handles_info = self.ipc.get_handle()
        device_index = self.color_buffer.buffer.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        self.ipc_server.set_ipc_init(color_buffer_info, depth_buffer_info, ipc_handles_info, device=device_index)

    def update_frame(self, color_data: torch.Tensor, depth_data: torch.Tensor, inverse_depth: bool = True):
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

            depth_data[:, :, :third] = 0.1         # Left third: VERY NEAR (0.1m)
            depth_data[:, :, third:2*third] = 5.0  # Middle third: MEDIUM (5m)
            depth_data[:, :, 2*third:] = 100.0     # Right third: VERY FAR (100m)
        
        self.color_buffer.update(color_data, inverse=False)
        self.depth_buffer.update(depth_data, inverse=inverse_depth)
        self.ipc.record_event()
    
    def update_view(self, view):
        self.msg_server.update_view(view)

    def update_gaussians(self, gaussians):
        self.msg_server.update_gaussians(gaussians)

    def close(self):
        self.ipc_server.close_socket()
        self.msg_server.close_socket()
