""" CUDA utilities """
import ctypes
import sys

# IPC handle structures (64 bytes)
class IpcMem(ctypes.Structure):
    _fields_ = [("raw", ctypes.c_byte * 64)]

class IpcEvt(ctypes.Structure):
    _fields_ = [("raw", ctypes.c_byte * 64)]

def load_cuda_runtime():
    if sys.platform == "win32":
        import os
        # Windows: cudart64_<ver>.dll
        dll_names = ["cudart64_110.dll", "cudart64_12.dll"]

        # Try loading from PATH first
        for name in dll_names:
            try:
                return ctypes.CDLL(name)
            except OSError:
                continue

        # Search CUDA toolkit installation directories
        cuda_dirs = []
        cuda_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
        if cuda_path:
            cuda_dirs.append(os.path.join(cuda_path, "bin"))
        # Standard NVIDIA install locations
        toolkit_base = os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                                    "NVIDIA GPU Computing Toolkit", "CUDA")
        if os.path.isdir(toolkit_base):
            for ver_dir in sorted(os.listdir(toolkit_base), reverse=True):
                cuda_dirs.append(os.path.join(toolkit_base, ver_dir, "bin"))

        for d in cuda_dirs:
            for name in dll_names:
                full_path = os.path.join(d, name)
                if os.path.isfile(full_path):
                    try:
                        return ctypes.CDLL(full_path)
                    except OSError:
                        continue

        raise OSError(f"Could not load CUDA runtime. Tried DLLs {dll_names} on PATH and in {cuda_dirs}")
    else:
        # Linux
        for name in ["libcudart.so", "libcudart.so.12", "libcudart.so.11"]:
            try:
                return ctypes.CDLL(name)
            except OSError:
                continue
        raise OSError("Could not load CUDA runtime (libcudart.so)")


def get_ipc_offset(tensor):
    """Get the IPC byte offset of a CUDA tensor from the base of its cudaMalloc block.

    PyTorch's caching allocator may place the tensor at an offset within a larger
    cudaMalloc block. cudaIpcGetMemHandle returns the handle for the whole block,
    so clients need this offset to read from the correct location.

    On Windows, PyTorch's _share_cuda_() may not be available, so we fall back
    to cuMemGetAddressRange from the CUDA driver API.
    """
    # Try PyTorch's built-in method first (works reliably on Linux)
    try:
        return tensor.untyped_storage()._share_cuda_()[3]
    except (RuntimeError, AttributeError):
        pass

    # Fallback: use CUDA driver API cuMemGetAddressRange to find the base pointer
    if sys.platform == "win32":
        try:
            nvcuda = ctypes.CDLL("nvcuda.dll")
        except OSError:
            nvcuda = ctypes.WinDLL("nvcuda")
    else:
        nvcuda = ctypes.CDLL("libcuda.so.1")

    base = ctypes.c_uint64(0)
    size = ctypes.c_size_t(0)
    data_ptr = tensor.data_ptr()
    ret = nvcuda.cuMemGetAddressRange(
        ctypes.byref(base), ctypes.byref(size), ctypes.c_uint64(data_ptr)
    )
    if ret != 0:
        raise RuntimeError(f"cuMemGetAddressRange failed with error code {ret}")
    return data_ptr - base.value
