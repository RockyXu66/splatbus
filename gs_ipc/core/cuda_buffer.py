import torch
from loguru import logger

class CUDAFrameBuffer:
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
        
        logger.info(f"[CUDAFrameBuffer] Created buffer: {width}x{height}x{channels}, pitch={self.pitch} bytes, ptr=0x{self.ptr:x}")
    
    def update(self, image_data: torch.Tensor, inverse: bool = False):
        """
        Update the buffer data

        Args:
            image_data: torch.Tensor, should be [C, H, W] or [H, W, C]
            is_depth: bool, if True, treats data as inverse depth and converts to linear depth
        """
        
        # Debug logging
        if self.update_count < 2:
            logger.info(f"[CUDAFrameBuffer] Update #{self.update_count} (inverse={inverse})")
            logger.info(f"  Input shape: {image_data.shape}, dtype: {image_data.dtype}")
            logger.info(f"  Input range: [{image_data.min():.6f}, {image_data.max():.6f}]")

        # Change from [C, H, W] to [H, W, C]
        if image_data.dim() == 3:
            if image_data.shape[0] in [1, 3, 4]: # [1, H, W] or [3, H, W] or [4, H, W]
                image_data = image_data.permute(1, 2, 0)
        
        if self.update_count < 2:
            logger.info(f"  After permute: {image_data.shape}")
        
        # Flip vertically to convert from OpenCV (top-left origin) to OpenGL (bottom-left origin) for Unity
        image_data = torch.flip(image_data, dims=[0])
        
        # Convert inverse depth to linear depth if needed
        if inverse and image_data.shape[-1] == 1:
            # Gaussian Splatting outputs inverse depth (1/z), Unity expects linear depth (z)
            eps = 1e-6
            image_data = torch.where(
                image_data > eps,
                1.0 / image_data,  # Convert inverse depth to linear depth
                torch.tensor(100.0, device=image_data.device, dtype=image_data.dtype)  # Far plane for invalid values
            )
         
        # Copy the data (GPU to GPU)
        if image_data.shape[-1] == 1: # Grayscale (depth)
            self.buffer.copy_(image_data, non_blocking=True)
        elif image_data.shape[-1] == 3: # RGB
            self.buffer[..., :3].copy_(image_data.clamp(0, 1), non_blocking=True)
            self.buffer[..., 3].fill_(1.0)
        else:
            self.buffer.copy_(image_data.clamp(0, 1), non_blocking=True)
        
        self.update_count += 1
    
    def get_info(self):
        if self.channels == 4:
            format = 'RGBA32F'
        elif self.channels == 1:
            format = 'R32F'
        else:
            raise ValueError(f"Invalid number of channels: {self.channels}")
        return {
            'ptr': self.ptr,
            'width': self.width,
            'height': self.height,
            'pitch': self.pitch,
            'format': format,
        }
        