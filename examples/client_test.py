"""
Simulated client for gs-ipc.

Connects to both channels:
- IPC channel (default 6001): receives init packets
- Message channel (default 6000): sends camera/point cloud pose messages
"""
import argparse
import json
import socket
import struct
import threading
import time


def send_json(sock: socket.socket, payload: dict) -> None:
    data = json.dumps(payload).encode("utf-8")
    sock.sendall(struct.pack("<I", len(data)))
    sock.sendall(data)


def recv_exact(sock: socket.socket, size: int):
    buf = b""
    while len(buf) < size:
        chunk = sock.recv(size - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def recv_json(sock: socket.socket):
    header = recv_exact(sock, 4)
    if not header:
        return None
    (length,) = struct.unpack("<I", header)
    payload_bytes = recv_exact(sock, length)
    if not payload_bytes:
        return None
    return json.loads(payload_bytes.decode("utf-8"))


def ipc_listener(sock: socket.socket, stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        try:
            payload = recv_json(sock)
        except (OSError, json.JSONDecodeError):
            break
        if payload is None:
            break
        print("[IPC] init packet received:", payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unity client simulator for gs-ipc.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--ipc-port", type=int, default=6001)
    parser.add_argument("--msg-port", type=int, default=6000)
    parser.add_argument("--interval", type=float, default=0.1)
    args = parser.parse_args()

    ipc_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ipc_sock.connect((args.host, args.ipc_port))
    print(f"[IPC] connected to {args.host}:{args.ipc_port}")

    stop_event = threading.Event()
    ipc_thread = threading.Thread(
        target=ipc_listener, args=(ipc_sock, stop_event), daemon=True
    )
    ipc_thread.start()

    msg_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    msg_sock.connect((args.host, args.msg_port))
    print(f"[MSG] connected to {args.host}:{args.msg_port}")

    try:
        t = 0.0
        while True:
            cam_payload = {
                "type": "camera_pose",
                "position": {"x": 0.1 * t, "y": 0.0, "z": 1.5},
                "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            }
            pc_payload = {
                "type": "point_cloud_pose",
                "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            }
            send_json(msg_sock, cam_payload)
            send_json(msg_sock, pc_payload)
            t += 1.0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        try:
            ipc_sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        ipc_sock.close()
        try:
            msg_sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        msg_sock.close()


if __name__ == "__main__":
    main()
