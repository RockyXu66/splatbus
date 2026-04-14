"""
Basic usage example for splatbus package
"""
import numpy as np
import torch
import splatbus

def main():
    # Initialize the renderer
    width = 532
    height = 948
    print(f"Initializing renderer ({width}x{height})...")
    renderer = splatbus.GaussianSplattingIPCRenderer(width=width, height=height)

    default_view = splatbus.IPCCamera(
        width=width, height=height,
        R=np.eye(3, dtype=np.float32),
        t=np.zeros(3, dtype=np.float32),
        fov_x=1.0, fov_y=1.0,
    )
    renderer.msg_server.init_view(default_view)

    while True:
        # Generate or load your image data here
        image_data = torch.rand(4, height, width, device='cuda', dtype=torch.float32)
        depth_data = torch.zeros(1, height, width, device='cuda', dtype=torch.float32)

        # Create partition test: left half transparent, right opaque, middle fully transparent
        image_data[3, :, :width//3] = 0.7
        image_data[3, :, width//3:2*width//3] = 0.0
        image_data[3, :, 2*width//3:] = 1.0

        depth_data[:, :, :width//3] = 5.0               # Near distance
        depth_data[:, :, width//3:2*width//3] = 20.0    # Far distance
        depth_data[:, :, 2*width//3:] = 20.0            # Far distance

        renderer.update_frame(color_data=image_data, depth_data=depth_data, inverse_depth=False)
    
    # Close when done
    print("Closing renderer...")
    renderer.close()
    print("Done!")

if __name__ == "__main__":
    main()

