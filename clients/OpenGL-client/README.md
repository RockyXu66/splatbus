# OpenGL Viewer Client

A standalone OpenGL viewer that connects to a running SplatBus renderer (server) and displays the rendered Gaussian Splatting frames in real time via CUDA IPC.

## Requirements

- CUDA-capable GPU with OpenGL support (both must run on the same GPU)
- [uv](https://docs.astral.sh/uv/) for dependency management
- A running SplatBus renderer (see [examples](../../examples))

## Installation

```bash
cd clients/OpenGL-client
uv sync
```

This installs all Python dependencies (including `splatbus`) into a local virtual environment.

## Usage

First, start a SplatBus renderer (server), then launch the viewer:

```bash
uv run viewer.py
```

The viewer connects to `127.0.0.1` on IPC port `6001` and message port `6000` by default.

## Controls

| Key | Action |
|-----|--------|
| W / S | Move forward / backward |
| A / D | Move left / right |
| Q / E | Move up / down |
| Mouse drag | Look around (yaw / pitch) |
| Space | Pause / unpause |
| 0 | Toggle between keyboard/mouse and orbit controller |
