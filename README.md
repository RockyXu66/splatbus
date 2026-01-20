# gs-ipc

IPC (Inter-Process Communication) Package for Gaussian Splatting with CUDA support.

## Overview

`gs-ipc` provides a high-performance IPC mechanism for sharing CUDA frame buffers between Python and Unity (or other applications) for real-time Gaussian Splatting rendering.

## Features

- **CUDA Frame Buffer Management**: GPU memory management for frame data
- **IPC Handle Management**: Share CUDA memory across processes
- **TCP Socket Communication**: Cross-platform inter-process communication
- **Real-time Updates**: Optimized for low-latency frame updates

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0 with CUDA support
- CUDA-capable GPU
- Localhost TCP connectivity between apps

## Examples

### IPC test (Python renderer)

`examples/ipc_test.py` simulates the Python renderer loop and publishes CUDA buffers.

```bash
python examples/ipc_test.py
```

### Simulated client

`examples/client_test.py` connects to both channels to simulate client (e.g. Unity):

- IPC channel (default `6001`): receives init packets
- Message channel (default `6000`): sends camera and point cloud poses

Run it with:

```bash
python examples/client_test.py
```

Optional flags:

```bash
python examples/client_test.py --host 127.0.0.1 --ipc-port 6001 --msg-port 6000 --interval 0.1
```

