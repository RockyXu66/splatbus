# Unity Viewer for SplatBus

A Unity native rendering plugin for real-time Gaussian Splatting visualization, serving as a client for [SplatBus](https://github.com/RockyXu66/splatbus).

This project is adopted from Unity's official [NativeRenderingPlugin](https://github.com/Unity-Technologies/NativeRenderingPlugin) example.

## Features

- [x] **CUDA-OpenGL Interop**: GPU-to-GPU data transfer via CUDA IPC for high-performance rendering
- [x] **Depth-Aware Blending**: Properly blend Gaussian Splatting with Unity's 3D scene meshes
- [x] **Real-time Camera Sync**: Stream camera poses from Unity to the renderer server via TCP
- [x] **Interactive Point Clouds**: Manipulate point cloud objects for visualization control
- [ ] **Load Camera Intrinsics from Server**: Fetch camera intrinsics from the server to align point clouds with the rendered output
- [ ] **Multi-View Support**: Render multiple viewpoints (POVs)
- [ ] **VR Support**: Add stereoscopic rendering and support for VR
- [ ] **Multiple Gaussian Objects Support**
- [ ] **HDRP Compatibility**: Support the High Definition Render Pipeline (HDRP)
- [ ] **URP Compatibility**: Support the Universal Render Pipeline (URP)

## Requirements

### Build Dependencies

- CMake >= 3.10
- CUDA Toolkit (tested with CUDA 12.x)
- OpenGL (GLVND)
- C++14 compatible compiler

### Runtime Dependencies

- Unity 2023.1.15f1 or later
- Linux (tested on Ubuntu)
- NVIDIA GPU with CUDA support

## Project Structure

```
├── PluginSource/                    # Native C++ plugin source
│   ├── src/                         # Source files
│   │   ├── RenderingPlugin.cpp      # Main plugin logic & CUDA IPC
│   │   ├── RenderAPI.cpp            # Render API abstraction
│   │   └── RenderAPI_*.cpp          # Platform-specific implementations
│   ├── include/                     # Header files
│   │   ├── Unity/                   # Unity plugin interface headers
│   │   └── nlohmann/                # JSON library
│   ├── dependencies/                # Third-party dependencies
│   │   └── gl3w/                    # OpenGL loader
│   └── projects/GNUMake/            # Build configuration
│       └── CMakeLists.txt
│
└── UnityProject/                    # Unity project
    └── Assets/
        ├── Scripts/
        │   ├── GSViewer.cs          # Gaussian Splatting viewer with depth blending
        │   ├── UnityDataSender.cs   # Camera pose TCP sender
        │   ├── UseRenderingPlugin.cs # Basic plugin usage example
        │   └── ...
        ├── Materials/               # Shaders & materials for GS rendering
        ├── Scenes/
        │   ├── UnityViewerExample.unity  # SplatBus client scene
        │   └── scene.unity               # Original NativeRenderingPlugin demo
        └── Plugins/                 # Compiled native plugins
            └── x86_64/libRenderingPlugin.so
```

## Building the Native Plugin

```bash
cd PluginSource/projects/GNUMake

# Create build directory
mkdir -p build && cd build

# Configure and build
cmake ..
make -j$(nproc)

# Copy to Unity project
cp libRenderingPlugin.so ../../../../UnityProject/Assets/Plugins/x86_64/
```

## Usage

### Setting Up the Unity Scene

1. Open the Unity project in Unity 2023.1.15f1 or later
2. Open `Assets/Scenes/UnityViewerExample.unity` (already pre-configured with all necessary components)

**Optional**: If you need to customize the IPC settings, select the GSViewer component in main camera and adjust:
- `IPC Render Width/Height`: Match your SplatBus renderer resolution

### Connecting to SplatBus Server

1. Start the SplatBus renderer server (see [server_test.py](https://github.com/RockyXu66/splatbus/blob/main/splatbus/examples/server_test.py) for an example)
   - IPC server: port `6001`
   - Pose receiver: port `6000`
2. Run the Unity scene
3. The plugin will automatically connect and begin receiving rendered frames

### Component Overview

| Component | Description |
|-----------|-------------|
| `GSViewer` | Main viewer component. Handles IPC frame receiving and depth-aware blending |
| `UnityDataSender` | Sends camera/point cloud poses to the renderer server |
| `UseRenderingPlugin` | Basic example showing texture/mesh buffer modification |

### Network Configuration

- **IPC Server**: `127.0.0.1:6001` - Receives CUDA IPC handles for GPU buffer sharing
- **Pose Server**: `127.0.0.1:6000` - Sends camera poses (configurable in `UnityDataSender`)

## Key Modifications from Original

### Native Plugin

- Added CUDA IPC module for receiving GPU memory handles from Python
- Added GPU-to-GPU memory copy using CUDA-OpenGL interop
- Added TCP socket communication for initialization handshake

### Unity Project

- `GSViewer`: Depth-aware Gaussian Splatting blending (based on [GaussianSplattingVRViewerUnity](https://github.com/clarte53/GaussianSplattingVRViewerUnity))
- `UnityDataSender`: Real-time camera pose streaming with Unity-to-OpenCV coordinate conversion
- Custom shaders for proper depth compositing

## License

See [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Unity NativeRenderingPlugin](https://github.com/Unity-Technologies/NativeRenderingPlugin) - Base plugin framework
- [GaussianSplattingVRViewerUnity](https://github.com/clarte53/GaussianSplattingVRViewerUnity) - Depth blending approach
- [nlohmann/json](https://github.com/nlohmann/json) - JSON parsing library
