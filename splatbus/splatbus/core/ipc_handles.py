import torch
import ctypes
from .shared_buffer import SharedBuffer
from .cuda_utils import IpcMem, IpcEvt, load_cuda_runtime
from loguru import logger

class IPCHandleManager:
    """ Manage IPC handles """

    def __init__(self, color_buffer: SharedBuffer, depth_buffer: SharedBuffer) -> None:

        self.color_buffer: SharedBuffer = color_buffer
        self.depth_buffer: SharedBuffer = depth_buffer

        self.cuda = load_cuda_runtime()

        self.mem_handle_color = self._create_memory_handle(self.color_buffer)
        self.mem_handle_depth = self._create_memory_handle(self.depth_buffer)
        self.evt_handle, self.evt_ptr = self._create_event_handle()

    def _load_cuda_runtime(self):
        try:
            cuda = ctypes.CDLL("libcudart.so")  # Linux
        except OSError:
            try:
                cuda = ctypes.CDLL("libcudart.so.12")
            except OSError:
                cuda = ctypes.CDLL("libcudart.so.11")
        return cuda
    
    def _create_memory_handle(self, buffer: SharedBuffer) -> IpcMem:
        """ Create memory handle """
        mh = IpcMem()
        ret = self.cuda.cudaIpcGetMemHandle(
            ctypes.byref(mh), 
            ctypes.c_void_p(buffer.ptr)
        )
        if ret != 0:
            raise RuntimeError(f"cudaIpcGetMemHandle failed with error code {ret}")
        logger.info(f"[IPCHandleManager] ✓ Memory handle created (size: {len(mh.raw)} bytes)")
        return mh


    def _create_event_handle(self) -> IpcMem:
        """ Create event handle """

        # CUDA Event flags
        # ref: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html
        cudaEventDefault = 0x00
        cudaEventBlockingSync = 0x01
        cudaEventDisableTiming = 0x02
        cudaEventInterprocess = 0x04

        # IPC Event have to use Interprocess + DisableTiming
        flags = cudaEventInterprocess | cudaEventDisableTiming

        logger.info(f"[IPCHandleManager] Event flags: 0x{flags:02x} (Interprocess=0x{cudaEventInterprocess:02x}, DisableTiming=0x{cudaEventDisableTiming:02x})")

        # Create CUDA Event
        event_ptr = ctypes.c_void_p()
        ret = self.cuda.cudaEventCreateWithFlags(
            ctypes.byref(event_ptr),
            ctypes.c_uint(flags)
        )

        if ret != 0:
            raise RuntimeError(f"cudaEventCreateWithFlags failed with error code {ret}")

        if not event_ptr.value:
            raise RuntimeError("cudaEventCreateWithFlags returned NULL event handle")

        logger.info(f"[IPCHandleManager] Event created successfully at address: {hex(event_ptr.value)}")

        # Get IPC event handle
        eh_frame = IpcEvt()
        logger.info(f"[IPCHandleManager] Getting IPC event handle...")
        ret = self.cuda.cudaIpcGetEventHandle(ctypes.byref(eh_frame), event_ptr)

        if ret != 0:
            error_messages = {
                1: "cudaErrorInvalidValue",
                2: "cudaErrorMemoryAllocation",
                3: "cudaErrorInitializationError",
                400: "cudaErrorInvalidResourceHandle - Event doesn't support IPC",
                801: "cudaErrorNotSupported - IPC not supported on this system"
            }
            error_msg = error_messages.get(ret, f"Unknown error {ret}")
            
            raise RuntimeError(f"cudaIpcGetEventHandle failed: {error_msg}")

        logger.info(f"[IPCHandleManager] ✓ Event IPC handle created (size: {len(eh_frame.raw)} bytes)")
        return eh_frame, event_ptr
    
    def record_event(self):
        """ 
        Record event to signal that the buffers are ready to read
        Should be called after updating both color and depth buffers
        """
        # Synchronize to ensure all previous CUDA operations complete
        # Using default stream (0) since SharedBuffer uses non_blocking=True which uses default stream
        torch.cuda.synchronize()
        
        # Record event on default stream to signal completion
        self.cuda.cudaEventRecord(self.evt_ptr, ctypes.c_void_p(0))  # stream = 0 (default)
    
    def get_handle(self):
        """ Get all handles """

        return {
            'mem_handle_color': bytes(self.mem_handle_color.raw),
            'mem_handle_depth': bytes(self.mem_handle_depth.raw),
            'evt_handle': bytes(self.evt_handle.raw),
        }