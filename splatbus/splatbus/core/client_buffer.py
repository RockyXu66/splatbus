import torch
import ctypes
import base64
from loguru import logger

from .cuda_utils import load_cuda_runtime, IpcMem, IpcEvt

class ClientBuffer:
    """ Client-side CUDA IPC buffer for reading shared memory """
    
    def __init__(self) -> None:
        self.cuda = load_cuda_runtime()
        self.dev_ptr = None
        self.evt_ptr = None
        self.owns_event = False
        self.stream = None
        self.width = 0
        self.height = 0
        self.channels = 0
        self.read_buffer = None
        
        # Create CUDA stream for async operations
        self.stream = ctypes.c_void_p()
        ret = self.cuda.cudaStreamCreate(ctypes.byref(self.stream))
        if ret != 0:
            logger.error(f"[ClientBuffer] cudaStreamCreate failed with code {ret}")
            self.stream = None
        
    def open_mem_handle(self, handle_b64: str, width: int, height: int, channels: int = 4, offset: int = 0) -> bool:
        """
        Open IPC handle from base64 string.

        Args:
            offset: Byte offset from the IPC handle base to the actual tensor data.
                    PyTorch's caching allocator may place tensors at an offset within
                    a larger cudaMalloc block.
        """
        self.width = width
        self.height = height
        self.channels = channels
        self.ipc_offset = offset

        try:
            handle_bytes = base64.b64decode(handle_b64)
            mh = IpcMem()
            ctypes.memmove(ctypes.byref(mh), handle_bytes, 64)
            
            dev_ptr = ctypes.c_void_p()
            # cudaIpcOpenMemHandle(void **devPtr, cudaIpcMemHandle_t handle, unsigned int flags)
            # https://nvidia.github.io/cuda-python/cuda-bindings/latest/module/runtime.html#cuda.bindings.runtime.cudaIpcOpenMemHandle
            ret = self.cuda.cudaIpcOpenMemHandle(
                ctypes.byref(dev_ptr),
                mh,
                ctypes.c_uint(1) # cudaIpcMemLazyEnablePeerAccess
            )
            
            if ret != 0:
                logger.error(f"[ClientBuffer] cudaIpcOpenMemHandle failed with code {ret}")
                return False
            if dev_ptr.value is None:
                logger.error("[ClientBuffer] cudaIpcOpenMemHandle returned NULL memory handle")
                return False

            self.dev_ptr = dev_ptr
            
            self.read_buffer = torch.empty((self.height, self.width, self.channels), dtype=torch.float32, device='cuda')
            
            logger.info(f"[ClientBuffer] Opened handle, dev_ptr: {hex(dev_ptr.value)}, ipc_offset: {offset}")
            logger.info(f"[ClientBuffer] Pre-allocated buffer (pytorch tensor): {self.read_buffer.shape}")
            return True
            
        except Exception as e:
            logger.error(f"[ClientBuffer] Failed to open handle: {e}")
            return False

    def open_event_handle(self, handle_b64: str) -> bool:
        """
        Open IPC event handle from base64 string
        """
        
        try:
            handle_bytes = base64.b64decode(handle_b64)
            eh = IpcEvt()
            ctypes.memmove(ctypes.byref(eh), handle_bytes, 64)
            
            evt_ptr = ctypes.c_void_p()
            ret = self.cuda.cudaIpcOpenEventHandle(
                ctypes.byref(evt_ptr),
                eh
            )
            
            if ret != 0:
                logger.error(f"[ClientBuffer] cudaIpcOpenEventHandle failed with code {ret}")
                return False
            if evt_ptr.value is None:
                logger.error("[ClientBuffer] cudaIpcOpenEventHandle returned NULL event handle")
                return False

            self.evt_ptr = evt_ptr
            self.owns_event = True
            
            logger.info(f"[ClientBuffer] Opened handle, evt_ptr: {hex(evt_ptr.value)}")
            return True
            
        except Exception as e:
            logger.error(f"[ClientBuffer] Failed to open handle: {e}")
            return False


    def enqueue_read(self, stream_handle=None) -> None:
        """
        Enqueue an async device-to-device copy from the IPC memory into read_buffer.

        The caller owns event waits and stream synchronization. This allows a
        client to wait once, enqueue color/depth/metadata copies together, then
        synchronize once so the buffers are behind the same event.
        """
        if not self.dev_ptr or self.dev_ptr.value is None or self.read_buffer is None:
            raise ValueError("ClientBuffer not initialized")

        # cudaMemcpyAsync(void *dst, const void *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream)
        size = self.width * self.height * self.channels * 4  # 4 bytes per float32
        src_ptr = ctypes.c_void_p(self.dev_ptr.value + self.ipc_offset)
        if stream_handle is None:
            stream_handle = self.stream if self.stream else ctypes.c_void_p(0)

        ret = self.cuda.cudaMemcpyAsync(
            ctypes.c_void_p(self.read_buffer.data_ptr()),
            src_ptr,
            ctypes.c_size_t(size),
            ctypes.c_int(3),        # cudaMemcpyDeviceToDevice = 3
            stream_handle
        )
        if ret != 0:
            logger.error(f"[ClientBuffer] cudaMemcpyAsync failed with code {ret}")

    def close(self):
        """Close IPC handle and free resources"""
        # Destroy stream first
        if self.stream:
            self.cuda.cudaStreamDestroy(self.stream)
            self.stream = None
            
        if self.dev_ptr:
            self.cuda.cudaIpcCloseMemHandle(self.dev_ptr)
            self.dev_ptr = None
            
        if self.evt_ptr:
            if self.owns_event:
                self.cuda.cudaEventDestroy(self.evt_ptr)
            self.evt_ptr = None
            self.owns_event = False
        
        # Clear read_buffer
        if self.read_buffer is not None:
            del self.read_buffer
            self.read_buffer = None
