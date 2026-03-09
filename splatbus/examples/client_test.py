"""
Client simulator for splatbus.
"""
import argparse
import time
import os
import cv2
import shutil
import numpy as np
from loguru import logger
from splatbus import GaussianSplattingIPCClient

def main() -> None:
    parser = argparse.ArgumentParser(description="Client simulator for splatbus.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--ipc-port", type=int, default=6001)
    parser.add_argument("--msg-port", type=int, default=6000)
    parser.add_argument("--interval", type=float, default=0.5)
    args = parser.parse_args()

    client = GaussianSplattingIPCClient(
        host=args.host,
        ipc_port=args.ipc_port,
        msg_port=args.msg_port
    )
    
    logger.info(f"Connecting to {args.host}...")
    client.connect()
    logger.info("Connected!")

    folder_name = 'frames'
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
    os.makedirs(folder_name)

    try:
        t = 0.0
        frame_idx = 0
        while True:
            # Send poses
            client.send_camera_pose(
                position={"x": 0.1 * t, "y": 0.0, "z": 1.5},
                rotation={"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
            )
            client.send_point_cloud_pose(
                position={"x": 0.0, "y": 0.0, "z": 0.0},
                rotation={"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
            )
            
            # Receive frames
            frames = client.receive()
            
            if 'color' in frames:
                tensor = frames['color']
                # float32 [H, W, C]
                arr = tensor.cpu().numpy()
                
                # uint8 [0, 255]
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
                    
                cv2.imwrite(f"{folder_name}/frame_{frame_idx:04d}.png", arr)
                logger.info(f"Saved {folder_name}/frame_{frame_idx:04d}.png")
                frame_idx += 1

            # Update the translation of the camera for every frame
            t += 1.0
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        logger.info("Closing connection...")
        client.close()

if __name__ == "__main__":
    main()
