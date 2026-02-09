"""IPC Package for Gaussian Splatting."""

from .core.shared_buffer import SharedBuffer
from .core.client_buffer import ClientBuffer
from .core.ipc_handles import IPCHandleManager
# from .core.ipc_channel import IPCSocketServer
# from .core.message_channel import MessageSocketServer
from .renderer import GaussianSplattingIPCRenderer
from .client import GaussianSplattingIPCClient
# from .transform_server import TransformServer

__version__ = '0.1.0'
__all__ = [
    'SharedBuffer',
    'ClientBuffer',
    'IPCHandleManager',
    # 'IPCSocketServer',
    # 'MessageSocketServer',
    'GaussianSplattingIPCRenderer',
    'GaussianSplattingIPCClient',
    # 'TransformServer',
]
