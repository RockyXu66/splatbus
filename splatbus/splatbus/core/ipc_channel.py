import socket
import struct
import json
import base64
from typing import Optional, Dict
from loguru import logger

from .BaseSocket import BaseSocketServer

class IPCSocketServer(BaseSocketServer):

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 6001,
        max_retries: int = 30,
    ) -> None:

        self.max_retries = max_retries
        self.retry_delay = 1.0
        self._init_payload = None

        super().__init__(host=host, port=port, server_name="IPCSocketServer")

    def set_ipc_init(
        self,
        color_buffer_info: dict,
        depth_buffer_info: dict,
        ipc_handles_info: dict,
        device: Optional[int] = None,
    ):

        meta = {
            "w": color_buffer_info['width'],
            "h": color_buffer_info['height'],
            "fmtColor": color_buffer_info['format'],
            "pitchColor": color_buffer_info['pitch'],
            "fmtDepth": depth_buffer_info['format'],
            "pitchDepth": depth_buffer_info['pitch'],
            # Device pointers are needed to compute offsets from IPC base.
            "ptrColor": int(color_buffer_info['ptr']),
            "ptrDepth": int(depth_buffer_info['ptr']),
        }
        if device is not None:
            meta["device"] = int(device)

        init_packet = {
            "meta": meta,
            "mem_color": base64.b64encode(bytes(ipc_handles_info['mem_handle_color'])).decode('utf-8'),
            "mem_depth": base64.b64encode(bytes(ipc_handles_info['mem_handle_depth'])).decode('utf-8'),
            "evt_done": base64.b64encode(bytes(ipc_handles_info['evt_handle'])).decode('utf-8')
        }

        self._init_payload = init_packet

        self._send_init_if_ready(context="on init set")

    def _send_init_if_ready(self, context: str):
        if self._init_payload is None:
            return
        try:
            self.send_json(self._init_payload)
        except Exception as e:
            logger.info(f"[IPCSocketServer] ✗ Failed to send init {context}: {e}")
 
    def on_client_connected(self, conn: socket.socket, addr):
        self._set_active_connection(conn, close_old=True)

        if addr is not None:
            logger.info(f"[IPCSocketServer] ✓ Client connected from {addr[0]}:{addr[1]}")
        else:
            logger.info("[IPCSocketServer] ✓ Client connected")

        self._send_init_if_ready(context="on connect")