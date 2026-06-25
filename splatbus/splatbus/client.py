"""IPC Client for Gaussian Splatting."""

import socket
import struct
import json
import threading
import ctypes
from typing import Optional, Dict, Any, Tuple
import numpy as np
from loguru import logger
import torch

from .core.client_buffer import ClientBuffer

class GaussianSplattingIPCClient:
    """ Client for Gaussian Splatting """
    def __init__(self, host: str = "127.0.0.1", ipc_port: int = 6001, msg_port: int = 6000) -> None:
        """
        Initialize the client
        Args:
            host: the host to connect to
            ipc_port: the port to connect to for IPC
            msg_port: the port to connect to for message
        """
        self.host = host
        self.ipc_port = ipc_port
        self.msg_port = msg_port
        
        self.ipc_sock = None
        self.msg_sock = None
        
        self.stop_event = threading.Event()
        self.ipc_thread = None
        
        self.client_buffer_evt = None      # Keep reference to prevent cleanup
        self.client_buffer_color = None
        self.client_buffer_depth = None
        self.client_buffer_frameIdx = None
        
        self.connected = False

        self.ts_dict = {}
        
    def connect(self):
        """Connect to server"""
        try:
            # Connect IPC socket
            self.ipc_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.ipc_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.ipc_sock.connect((self.host, self.ipc_port))
            logger.info(f"[IPCClient] Connected to IPC {self.host}:{self.ipc_port}")
        except Exception as e:
            logger.error(f"[IPCClient] Connection to IPC failed: {e}")
            self.close()
            raise e
            
        # Start IPC listener thread
        self.ipc_thread = threading.Thread(
            target=self._ipc_listener, 
            daemon=True
        )
        self.ipc_thread.start()
            
        try:
            # Connect Message socket
            self.msg_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.msg_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.msg_sock.connect((self.host, self.msg_port))
            logger.info(f"[IPCClient] Connected to MSG {self.host}:{self.msg_port}")
            
            self.connected = True
        except Exception as e:
            logger.error(f"[IPCClient] Connection to MSG failed: {e}")
            self.close()
            raise e
            

    def _send_json(self, sock: socket.socket, payload: dict) -> None:
        if not sock:
            return
        try:
            data = json.dumps(payload).encode("utf-8")
            sock.sendall(struct.pack("<I", len(data)) + data)
        except Exception as e:
            logger.error(f"[IPCClient] Send failed: {e}")

    def _recv_exact(self, sock: socket.socket, size: int):
        buf = b""
        while len(buf) < size:
            try:
                chunk = sock.recv(size - len(buf))
                if not chunk:
                    return None
                buf += chunk
            except Exception:
                return None
        return buf

    def _recv_json(self, sock: socket.socket):
        header = self._recv_exact(sock, 4)
        if not header:
            return None
        (length,) = struct.unpack("<I", header)
        payload_bytes = self._recv_exact(sock, length)
        if not payload_bytes:
            return None
        return json.loads(payload_bytes.decode("utf-8"))
    
    def _recv_encoded_frame(self, sock: socket.socket):
        HEADER = struct.Struct("<IIIQQ")
        header = self._recv_exact(sock, HEADER.size)
        if not header:
            return None
        (frame_idx, width, height, color_nbytes, depth_nbytes) = HEADER.unpack(header)
        color_data = self._recv_exact(sock, color_nbytes)
        depth_data = self._recv_exact(sock, depth_nbytes)
        if color_data is None or depth_data is None:
            return None
        return frame_idx, width, height, color_data, depth_data

    def _ipc_listener(self) -> None:
        while not self.stop_event.is_set():
            try:
                payload = self._recv_json(self.ipc_sock)
            except (OSError, json.JSONDecodeError):
                break
                
            if payload is None:
                break
                
            logger.info(f"[IPCClient] Packet received: {payload.keys()}")
            
            # Handle init packet
            if "mem_color" in payload and "meta" in payload:
                self._init_buffers(payload)

    def _init_buffers(self, payload: dict):
        meta = payload["meta"]

        # Open event handle (shared by all buffers)
        evt_ptr = None
        if "evt_done" in payload:
            cb_evt = ClientBuffer()
            if cb_evt.open_event_handle(payload["evt_done"]):
                evt_ptr = cb_evt.evt_ptr
                # Keep reference to prevent cleanup of event and stream
                self.client_buffer_evt = cb_evt
                if evt_ptr is not None and evt_ptr.value is not None:
                    logger.info(f"[IPCClient] Event handle opened: {hex(evt_ptr.value)}")
        
        # Initialize Color Buffer with shared event
        if "mem_color" in payload:
            cb_color = ClientBuffer()
            offset = meta.get("offsetColor", 0)
            success = cb_color.open_mem_handle(payload["mem_color"], meta["w"], meta["h"], 4, offset=offset)
            if success:
                self.client_buffer_color = cb_color
                logger.info(f"[IPCClient] Color buffer initialized with event sync (ipc_offset={offset})")

        # Initialize Depth Buffer with shared event
        if "mem_depth" in payload:
            cb_depth = ClientBuffer()
            offset = meta.get("offsetDepth", 0)
            success = cb_depth.open_mem_handle(payload["mem_depth"], meta["w"], meta["h"], 1, offset=offset)
            if success:
                self.client_buffer_depth = cb_depth
                logger.info(f"[IPCClient] Depth buffer initialized with event sync (ipc_offset={offset})")

        # Initialize Frame Index Buffer with shared event
        if "mem_frameIdx" in payload:
            cb_frameIdx = ClientBuffer()
            offset = meta.get("offsetFrameIdx", 0)
            success = cb_frameIdx.open_mem_handle(payload["mem_frameIdx"], 1, 1, 1, offset=offset)
            if success:
                self.client_buffer_frameIdx = cb_frameIdx

    def receive(self) -> Dict[str, torch.Tensor]:
        """
        Receive latest frames from shared memory
        Returns dict with 'color' and 'depth' tensors (if available)
        """
        if self.client_buffer_evt is None:
            logger.warning("[IPCClient] No event synchronization available - reading without sync (may cause race condition)")

        result = {}

        buffers = [
            ("color", self.client_buffer_color, True),
            ("depth", self.client_buffer_depth, True),
            ("frame_idx", self.client_buffer_frameIdx, False),
        ]
        active_buffers = [(name, buf, flip) for name, buf, flip in buffers if buf is not None]

        if self.client_buffer_evt is not None and self.client_buffer_evt.evt_ptr and self.client_buffer_evt.stream:
            stream = self.client_buffer_evt.stream
            ret = self.client_buffer_evt.cuda.cudaStreamWaitEvent(
                stream,
                self.client_buffer_evt.evt_ptr,
                ctypes.c_uint(0),
            )
            if ret != 0:
                logger.error(f"[IPCClient] cudaStreamWaitEvent failed with code {ret}")

            for _, buf, _ in active_buffers:
                buf.enqueue_read(stream)

            ret = self.client_buffer_evt.cuda.cudaStreamSynchronize(stream)
            if ret != 0:
                logger.error(f"[IPCClient] cudaStreamSynchronize failed with code {ret}")

            for name, buf, flip in active_buffers:
                if buf.read_buffer is not None:
                    if flip:
                        buf.read_buffer = torch.flip(buf.read_buffer, dims=[0])
                    result[name] = buf.read_buffer
        else:
            logger.error("[IPCClient] No event synchronization available - reading without sync (may cause race condition)")
                
        return result
    
    def receive_encoded_stream(self) -> Dict[str, torch.Tensor]:
        """
        Receive latest frame over the CPU-transfer stream.

        Returns dict with 'color' and 'depth' tensors (if available)
        """
        result = {}

        payload = {
            "type": "get_encoded_stream_frame"
        }
        if self.msg_sock is None:
            return result

        try:
            self._send_json(self.msg_sock, payload)
            frame = self._recv_encoded_frame(self.msg_sock)  # Wait for response (can be empty)
        except Exception:
            frame = None

        if frame is None:
            return result

        frame_idx, width, height, color_data, depth_data = frame
        
        color = np.frombuffer(color_data, dtype=np.float32).reshape(height, width, 4)
        depth = np.frombuffer(depth_data, dtype=np.float32).reshape(height, width, 1)

        result = {
            'color': torch.from_numpy(color.copy()).to(device="cuda"),
            'depth': torch.from_numpy(depth.copy()).to(device="cuda"),
            'frame_idx': torch.tensor(frame_idx, dtype=torch.int32, device="cuda"),
        }
                
        return result

    def get_viewport_size(self) -> Tuple[int, int]:
        """
        Receive the viewport size from the server. This should be called after the
        initial connection and buffer setup, ie to setup the client window size.
        """
        if self.client_buffer_evt is None:
            logger.warning("[IPCClient] No event synchronization available - reading without sync (may cause race condition)")
        
        payload = {
            "type": "get_viewport_size"
        }
        try:
            self._send_json(self.msg_sock, payload)
            json_msg = self._recv_json(self.msg_sock)  # Wait for response (can be empty)
        except Exception:
            json_msg = None
        return json_msg.get("viewport_size", (0, 0)) if json_msg is not None else (0, 0)

    def get_camera_pose(self, cam_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Receive the camera pose from the server.
        """
        if self.client_buffer_evt is None:
            logger.warning("[IPCClient] No event synchronization available - reading without sync (may cause race condition)")
        
        payload = {
            "type": "get_camera_pose",
            "cam_idx": cam_idx
        }
        try:
            self._send_json(self.msg_sock, payload)
            json_msg = self._recv_json(self.msg_sock)  # Wait for response (can be empty)
        except Exception:
            json_msg = None
        default_position = [0, 0, 0]
        default_rotation = [0, 0, 0, 1]
        if json_msg is not None:
            position = json_msg.get("position", default_position)
            rotation = json_msg.get("rotation", default_rotation)
        else:
            position = default_position
            rotation = default_rotation
        return np.array(position), np.array(rotation)
        
        

    def send_camera_pose(self, position: Dict[str, float], rotation: Dict[str, float]):
        """
        Send camera pose
        position: {'x': float, 'y': float, 'z': float}
        rotation: {'x': float, 'y': float, 'z': float, 'w': float}
        """
        payload = {
            "type": "camera_pose",
            "position": position,
            "rotation": rotation,
        }
        self._send_json(self.msg_sock, payload)

    def send_point_cloud_pose(self, position: Dict[str, float], rotation: Dict[str, float]):
        """
        Send point cloud object pose
        """
        payload = {
            "type": "point_cloud_pose",
            "position": position,
            "rotation": rotation,
        }
        self._send_json(self.msg_sock, payload)
    
    def get_frame_info(self, frame_idx: int):
        """
        Receive the frame info from the server.
        """
        # if self.client_buffer_evt is None:
        #     logger.warning("[IPCClient] No event synchronization available - reading without sync (may cause race condition)")
        
        payload = {
            "type": "get_frame_info",
            "frame_idx": frame_idx
        }
        try:
            self._send_json(self.msg_sock, payload)
            json_msg = self._recv_json(self.msg_sock)  # Wait for response (can be empty)
        except Exception:
            json_msg = None
        default_frame_info = None
        if json_msg is not None:
            frame_info = json_msg.get("frame_info", default_frame_info)
        else:
            frame_info = default_frame_info
        return frame_info

    def close(self):
        self.connected = False
        self.stop_event.set()
        
        if self.client_buffer_color:
            self.client_buffer_color.close()
        if self.client_buffer_depth:
            self.client_buffer_depth.close()
        if self.client_buffer_frameIdx:
            self.client_buffer_frameIdx.close()
        if self.client_buffer_evt:
            self.client_buffer_evt.close()
            
        if self.ipc_sock:
            try:
                self.ipc_sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.ipc_sock.close()
            
        if self.msg_sock:
            try:
                self.msg_sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.msg_sock.close()
