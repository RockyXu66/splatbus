""" CUDA utilities """
import ctypes

# IPC handle structures (64 bytes)
class IpcMem(ctypes.Structure): 
    _fields_ = [("raw", ctypes.c_byte * 64)]

class IpcEvt(ctypes.Structure): 
    _fields_ = [("raw", ctypes.c_byte * 64)]

def load_cuda_runtime():
    try:
        cuda = ctypes.CDLL("libcudart.so")  # Linux
    except OSError:
        try:
            cuda = ctypes.CDLL("libcudart.so.12")
        except OSError:
            cuda = ctypes.CDLL("libcudart.so.11")
    return cuda