import json
import socket
import struct
import threading
import time
from abc import ABC, abstractmethod
from typing import Callable, Optional

from loguru import logger


class BaseSocketServer(threading.Thread, ABC):
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 6000,
        server_name: str = "BaseSocketServer",
        on_message: Optional[Callable[[dict], None]] = None,
        auto_start: bool = True,
    ) -> None:
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.on_message = on_message
        self.server_name = server_name
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen(1)

        logger.info(f"\n[{self.server_name}] Listening on {self.host}:{self.port}...")

        self._running = True
        self._conn_ready = threading.Event()
        self._conn_lock = threading.Lock()
        self._conn: Optional[socket.socket] = None
        if auto_start:
            self.start()

    def run(self):
        while self._running:
            conn, addr = self._accept_connection()
            if conn is None:
                break
            self.on_client_connected(conn, addr)

    @abstractmethod
    def on_client_connected(self, conn: socket.socket, addr):
        pass

    def wait_for_client(self, max_retries: int = 30, retry_delay: float = 1.0) -> bool:
        for attempt in range(max_retries):
            if self._conn_ready.is_set():
                return True
            logger.info(
                f"[{self.__class__.__name__}] Waiting for client ({attempt + 1}/{max_retries})..."
            )
            time.sleep(retry_delay)
        return False

    def send_json(self, data: dict):
        with self._conn_lock:
            if self._conn is None:
                raise RuntimeError("[{self.server_name}] Client connection is not available.")
            conn = self._conn
        self._send_json(conn, data)

    def close_socket(self):
        self._running = False
        try:
            with self._conn_lock:
                if self._conn is not None:
                    try:
                        self._conn.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    self._conn.close()
                    self._conn = None
            self.sock.close()
        except OSError:
            pass
        finally:
            self._conn_ready.clear()

    def _accept_connection(self):
        try:
            conn, addr = self.sock.accept()
        except OSError:
            if self._running:
                logger.info(f"[{self.__class__.__name__}] Accept failed")
            return None, None
        return conn, addr

    def _recv_loop(self, conn: socket.socket):
        while True:
            header = self._recv_exact(conn, 4)
            if not header:
                break
            (length,) = struct.unpack("<I", header)
            payload_bytes = self._recv_exact(conn, length)
            if not payload_bytes:
                break
            try:
                payload = json.loads(payload_bytes.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            self._handle_payload(payload)

    def _handle_payload(self, payload: dict):
        if self.on_message is not None:
            self.on_message(payload)
    
    def _send_json(self, conn: socket.socket, data: dict):

        # Send INIT packet (length prefix + JSON)
        json_data = json.dumps(data).encode('utf-8')

        try:
            conn.send(struct.pack('<I', len(json_data)))
            conn.send(json_data)
            logger.info(f"[{self.server_name}] ✓ Packet sent successfully!")
        except Exception as e:
            logger.info(f"[IPCSocketServer] ✗ Failed to send Packet: {e}")
            raise

    def _recv_exact(self, conn: socket.socket, size: int):
        buffer = b""
        while len(buffer) < size:
            chunk = conn.recv(size - len(buffer))
            if not chunk:
                return None
            buffer += chunk
        return buffer

    def _set_client_connected(self):
        self._conn_ready.set()

    def _set_active_connection(self, conn: socket.socket, close_old: bool = False):
        with self._conn_lock:
            if close_old and self._conn is not None and self._conn is not conn:
                try:
                    self._conn.close()
                except Exception:
                    pass
            self._conn = conn
            self._set_client_connected()

    def _clear_active_connection(self):
        with self._conn_lock:
            self._conn = None
            self._conn_ready.clear()
