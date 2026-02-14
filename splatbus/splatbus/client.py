"""IPC Client for Gaussian Splatting."""

import socket
import struct
import json
import threading
from typing import Optional, Dict, Any
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
        
        self.connected = False
        
    def connect(self):
        """Connect to server"""
        try:
            # Connect IPC socket
            self.ipc_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
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
            sock.sendall(struct.pack("<I", len(data)))
            sock.sendall(data)
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
                logger.info(f"[IPCClient] Event handle opened: {hex(evt_ptr.value)}")
        
        # Initialize Color Buffer with shared event
        if "mem_color" in payload:
            cb_color = ClientBuffer()
            success = cb_color.open_mem_handle(payload["mem_color"], meta["w"], meta["h"], 4)
            if success:
                cb_color.evt_ptr = evt_ptr  # Share the event
                self.client_buffer_color = cb_color
                logger.info("[IPCClient] Color buffer initialized with event sync")

        # Initialize Depth Buffer with shared event
        if "mem_depth" in payload:
            cb_depth = ClientBuffer()
            success = cb_depth.open_mem_handle(payload["mem_depth"], meta["w"], meta["h"], 1)
            if success:
                cb_depth.evt_ptr = evt_ptr  # Share the event
                self.client_buffer_depth = cb_depth
                logger.info("[IPCClient] Depth buffer initialized with event sync")

    def receive(self) -> Dict[str, torch.Tensor]:
        """
        Receive latest frames from shared memory
        Returns dict with 'color' and 'depth' tensors (if available)
        """
        if self.client_buffer_evt is None:
            logger.warning("[IPCClient] No event synchronization available - reading without sync (may cause race condition)")
        
        result = {}
        
        if self.client_buffer_color:
            self.client_buffer_color.read()
            if self.client_buffer_color.read_buffer is not None:
                result['color'] = self.client_buffer_color.read_buffer
                
        if self.client_buffer_depth:
            self.client_buffer_depth.read()
            if self.client_buffer_depth.read_buffer is not None:
                result['depth'] = self.client_buffer_depth.read_buffer
                
        return result

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

    def close(self):
        self.connected = False
        self.stop_event.set()
        
        if self.client_buffer_color:
            self.client_buffer_color.close()
        if self.client_buffer_depth:
            self.client_buffer_depth.close()
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
