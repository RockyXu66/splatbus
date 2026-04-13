# splatbus

IPC (Inter-Process Communication) Package for Gaussian Splatting with CUDA support.

## Overview

`splatbus` provides a high-performance IPC mechanism for sharing CUDA frame buffers between a Python renderer and clients (OpenGL, Unity, or other applications) for real-time Gaussian Splatting rendering.

## Features

- **CUDA Frame Buffer Management**: GPU memory management for frame data
- **IPC Handle Management**: Share CUDA memory across processes
- **TCP Socket Communication**: Cross-platform inter-process communication
- **Unified Camera Interface**: `IPCCamera` for consistent camera controls across clients
- **Real-time Updates**: Optimized for low-latency frame updates

## Project Structure

```
splatbus/
├── splatbus/
│   ├── __init__.py
│   ├── renderer.py          # GaussianSplattingIPCRenderer
│   ├── client.py            # GaussianSplattingIPCClient
│   ├── camera.py            # IPCCamera interface
│   └── core/
│       ├── BaseSocket.py
│       ├── shared_buffer.py
│       ├── client_buffer.py
│       ├── ipc_channel.py
│       ├── ipc_handles.py
│       ├── message_channel.py
│       ├── cuda_utils.py
│       └── utils.py
├── examples/
│   ├── server_test.py       # Simulated renderer
│   └── client_test.py       # Simulated client
├── pyproject.toml
└── setup.py
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0 with CUDA support
- CUDA-capable GPU
- Localhost TCP connectivity between apps

## Examples

### Simulated renderer (server)

`examples/server_test.py` simulates a Python renderer loop and publishes CUDA buffers via IPC:

```bash
python examples/server_test.py
```

### Simulated client

`examples/client_test.py` connects to both channels to simulate a client (e.g. Unity, OpenGL viewer):

- IPC channel (default `6001`): receives CUDA frame buffers
- Message channel (default `6000`): sends camera and point cloud poses

Run it with:

```bash
python examples/client_test.py
```

Optional flags:

```bash
python examples/client_test.py --host 127.0.0.1 --ipc-port 6001 --msg-port 6000 --interval 0.5
```

