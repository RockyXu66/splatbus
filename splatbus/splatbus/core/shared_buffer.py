import torch
from loguru import logger
from .cuda_utils import get_ipc_offset


def format_image_for_shared_buffer(
    image_data: torch.Tensor,
    channels: int,
    inverse: bool = False,
) -> torch.Tensor:
    """
    Convert renderer output to the exact GPU layout stored in SharedBuffer.

    Output shape is [H, W, channels]. Color buffers are RGBA32F with clamped
    color values and alpha=1 for RGB input. Depth buffers are R32F, vertically
    flipped, and optionally converted from inverse depth to linear depth.
    """
    # Change from [C, H, W] to [H, W, C].
    if image_data.dim() == 3:
        if image_data.shape[0] in [1, 3, 4]:  # [1, H, W] or [3, H, W] or [4, H, W]
            image_data = image_data.permute(1, 2, 0)

    # Flip vertically to convert from OpenCV (top-left origin) to OpenGL
    # (bottom-left origin) for Unity/OpenGL clients.
    image_data = torch.flip(image_data, dims=[0])

    # Convert inverse depth to linear depth if needed.
    if inverse and image_data.shape[-1] == 1:
        eps = 1e-6
        image_data = torch.where(
            image_data > eps,
            1.0 / image_data,
            torch.tensor(100.0, device=image_data.device, dtype=image_data.dtype),
        )

    if channels == 1:
        if image_data.shape[-1] != 1:
            raise ValueError(f"Expected one channel, got shape {tuple(image_data.shape)}")
        return image_data.contiguous()

    if channels == 4:
        if image_data.shape[-1] == 3:
            formatted = torch.empty(
                (*image_data.shape[:2], 4),
                dtype=image_data.dtype,
                device=image_data.device,
            )
            formatted[..., :3] = image_data.clamp(0, 1)
            formatted[..., 3] = 1.0
            return formatted.contiguous()
        if image_data.shape[-1] == 4:
            return image_data.clamp(0, 1).contiguous()

    raise ValueError(f"Invalid channel count for shape {tuple(image_data.shape)}: {channels}")


class SharedBuffer:
    """ Manage CUDA shared memory """

    def __init__(self, width: int, height: int, channels: int = 4, dtype=torch.float32) -> None:

        self.width = width
        self.height = height
        self.channels = channels
        self.dtype = dtype
        
        self.update_count = 0  # For debug logging

        # Initialize CUDA
        torch.cuda.init()
        torch.cuda.synchronize()

        # Allocate continuous memory
        self.buffer = torch.empty((height, width, channels), dtype=dtype, device='cuda').contiguous()

        self.ptr = self.buffer.data_ptr()
        self.pitch = width * channels * self.buffer.element_size()

        # Compute IPC offset: PyTorch's caching allocator may place this tensor
        # at an offset within a larger cudaMalloc block. cudaIpcGetMemHandle returns
        # the handle for the whole block, so the client needs this offset to read
        # from the correct location.
        self.ipc_offset = get_ipc_offset(self.buffer)

        logger.info(f"[SharedBuffer] Created buffer: {width}x{height}x{channels}, pitch={self.pitch} bytes, ptr=0x{self.ptr:x}, ipc_offset={self.ipc_offset}")
    
    def update(self, image_data: torch.Tensor, inverse: bool = False):
        """
        Update the buffer data

        Args:
            image_data: torch.Tensor, should be [C, H, W] or [H, W, C]
            is_depth: bool, if True, treats data as inverse depth and converts to linear depth
        """
        
        # Debug logging
        if self.update_count < 2:
            logger.info(f"[SharedBuffer] Update #{self.update_count} (inverse={inverse})")
            logger.info(f"  Input shape: {image_data.shape}, dtype: {image_data.dtype}")
            logger.info(f"  Input range: [{image_data.min():.6f}, {image_data.max():.6f}]")

        image_data = format_image_for_shared_buffer(
            image_data,
            channels=self.channels,
            inverse=inverse,
        )
        
        if self.update_count < 2:
            logger.info(f"  Formatted shape: {image_data.shape}")
         
        # Copy the data (GPU to GPU)
        self.buffer.copy_(image_data, non_blocking=True)
        
        self.update_count += 1
    
    def get_info(self):
        if self.channels == 4:
            format = 'RGBA32F'
        elif self.channels == 1:
            format = 'R32F'
        else:
            raise ValueError(f"Invalid number of channels: {self.channels}")
        return {
            'width': self.width,
            'height': self.height,
            'pitch': self.pitch,
            'format': format,
            'ipc_offset': self.ipc_offset,
        }
        
