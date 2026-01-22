"""IPC Package for Gaussian Splatting."""

from .core.cuda_buffer import CUDAFrameBuffer
from .core.ipc_handles import IPCHandleManager
# from .core.ipc_channel import IPCSocketServer
# from .core.message_channel import MessageSocketServer
from .renderer import GaussianSplattingIPCRenderer
# from .transform_server import TransformServer

__version__ = '0.1.0'
__all__ = [
    'CUDAFrameBuffer',
    'IPCHandleManager',
    # 'IPCSocketServer',
    # 'MessageSocketServer',
    'GaussianSplattingIPCRenderer',
    # 'TransformServer',
]
