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
        
    def open_mem_handle(self, handle_b64: str, width: int, height: int, channels: int = 4) -> bool:
        """
        Open IPC handle from base64 string
        """
        self.width = width
        self.height = height
        self.channels = channels
        
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
                
            self.dev_ptr = dev_ptr
            
            self.read_buffer = torch.empty((self.height, self.width, self.channels), dtype=torch.float32, device='cuda')
            
            logger.info(f"[ClientBuffer] Opened handle, dev_ptr: {hex(dev_ptr.value)}")
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
                
            self.evt_ptr = evt_ptr
            
            logger.info(f"[ClientBuffer] Opened handle, evt_ptr: {hex(evt_ptr.value)}")
            return True
            
        except Exception as e:
            logger.error(f"[ClientBuffer] Failed to open handle: {e}")
            return False

    def read(self) -> torch.Tensor:
        """
        Read data from client buffer to local tensor (read_buffer)
        """
        if not self.dev_ptr or self.read_buffer is None:
            raise ValueError("ClientBuffer not initialized")
        
        # Wait for renderer to finish writing
        if self.evt_ptr and self.stream:
            ret = self.cuda.cudaStreamWaitEvent(
                self.stream, 
                self.evt_ptr, 
                ctypes.c_uint(0)
            )
            if ret != 0:
                logger.error(f"[ClientBuffer] cudaStreamWaitEvent failed with code {ret}")
        
        # Async copy from IPC pointer to read_buffer
        # cudaMemcpyAsync(void *dst, const void *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream)
        size = self.width * self.height * self.channels * 4  # 4 bytes per float32
        
        stream_handle = self.stream if self.stream else ctypes.c_void_p(0)
        ret = self.cuda.cudaMemcpyAsync(
            ctypes.c_void_p(self.read_buffer.data_ptr()),
            self.dev_ptr,
            ctypes.c_size_t(size),
            ctypes.c_int(3),        # cudaMemcpyDeviceToDevice = 3
            stream_handle
        )
        
        if ret != 0:
            logger.error(f"[ClientBuffer] cudaMemcpyAsync failed with code {ret}")
        
        # Synchronize stream to ensure copy is complete before flipping
        if self.stream:
            ret = self.cuda.cudaStreamSynchronize(self.stream)
            if ret != 0:
                logger.error(f"[ClientBuffer] cudaStreamSynchronize failed with code {ret}")
        else:
            # Fallback to device synchronize if no stream
            self.cuda.cudaDeviceSynchronize()

        # Flip vertically (server sends opengl style image, we need cv2 style image)
        self.read_buffer = torch.flip(self.read_buffer, dims=[0])

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
            self.cuda.cudaIpcCloseEventHandle(self.evt_ptr)
            self.evt_ptr = None
        
        # Clear read_buffer
        if self.read_buffer is not None:
            del self.read_buffer
            self.read_buffer = None
