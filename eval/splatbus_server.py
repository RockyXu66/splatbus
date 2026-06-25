#!/usr/bin/env python3
#
# SplatBus IPC server for a 3DGS model, with cameras loaded from cameras.json
# (no source dataset needed). This is the renderer/server side that the OpenGL
# or Unity client connects to. 
#
# Usage:
#   python eval/splatbus_server.py --model /home/yixu/Downloads/room \
#       --iteration 30000 --width 1920 --height 1080 --fps 1000
#
import os
import sys
import json
import math
import time
from typing import Literal, Optional

GS_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "..", "examples", "gaussian-splatting"))
sys.path.insert(0, GS_DIR)

import numpy as np
import torch
from loguru import logger
import typer
app = typer.Typer()

from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from splatbus.camera import focal2fov
import splatbus

try:
    from diff_gaussian_rasterization import SparseGaussianAdam  # noqa: F401
    SEPARATE_SH = True
except Exception:
    SEPARATE_SH = False


class Pipe:
    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False
    antialiasing = False


class SimpleCam:
    """Minimal view object with the attributes IPCCamera.init_from_view reads."""
    def __init__(self, R, T, FoVx, FoVy):
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy


def load_cams(path, width, height):
    with open(path) as f:
        cams = json.load(f)
    out = []
    for c in cams:
        R_w2c = np.array(c["rotation"], dtype=np.float64)
        pos = np.array(c["position"], dtype=np.float64)
        R = R_w2c.T
        t = -R_w2c @ pos
        fovy = focal2fov(c["fy"], c["height"])
        fovx = 2.0 * math.atan(math.tan(fovy * 0.5) * width / height)
        out.append(SimpleCam(R.astype(np.float32), t.astype(np.float32), fovx, fovy))
    return out

@app.command()
def main(
    model: str = typer.Option(..., "--model"),
    cameras: Optional[str] = typer.Option(None, "--cameras"),
    iteration: int = typer.Option(30000, "--iteration"),
    sh_degree: int = typer.Option(3, "--sh-degree"),
    width: int = typer.Option(1920, "--width"),
    height: int = typer.Option(1080, "--height"),
    fps: int = typer.Option(1000, "--fps"),
    ipc_port: int = typer.Option(6001, "--ipc-port"),
    msg_port: int = typer.Option(6000, "--msg-port"),
    orbit: bool = typer.Option(False, "--orbit"),
    handoff_mode: Literal["CUDA-IPC", "encoded-stream", "CPU-shared-memory"] = typer.Option("CUDA-IPC", "--handoff-mode"),
):

    cam_json = cameras or os.path.join(model, "cameras.json")
    ply = os.path.join(model, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    assert os.path.isfile(ply), f"missing {ply}"

    logger.info(f"Loading {ply}")
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(ply)
    gaussians.active_sh_degree = gaussians.max_sh_degree
    logger.info(f"Loaded {gaussians.get_xyz.shape[0]:,} Gaussians")

    bg = torch.zeros(3, dtype=torch.float32, device="cuda")
    pipe = Pipe()
    cams = load_cams(cam_json, width, height)
    logger.info(f"{len(cams)} poses; resolution {width}x{height}; fps cap {fps}")

    if handoff_mode == "CUDA-IPC":
        renderer = splatbus.GaussianSplattingIPCRenderer(
            width=width, height=height,
            ipc_host="0.0.0.0", ipc_port=ipc_port,
            msg_host="0.0.0.0", msg_port=msg_port,
        )
    elif handoff_mode == "encoded-stream":
        renderer = splatbus.GaussianSplattingEncodedStreamRenderer(
            width=width, height=height,
            ipc_host="0.0.0.0", ipc_port=ipc_port,
            msg_host="0.0.0.0", msg_port=msg_port,
        )
    else:
        raise ValueError(f"Invalid handoff mode: {handoff_mode}")

    renderer.init_view(width=width, height=height, view=cams[0])
    renderer.set_cam_list(width=width, height=height, views=cams)

    target_dt = 1.0 / fps
    frame_idx = 0
    t_report = time.perf_counter()
    n_since = 0
    try:
        with torch.no_grad():
            while True:
                t0 = time.clock_gettime(time.CLOCK_MONOTONIC)
                if orbit:
                    base = splatbus.IPCCamera.init_from_view(width, height, cams[frame_idx % len(cams)])
                    view = base.cuda()
                else:
                    view = renderer.get_current_view().cuda()
                t1 = time.clock_gettime(time.CLOCK_MONOTONIC)
                renderer.update_gaussians(gaussians)
                t2 = time.clock_gettime(time.CLOCK_MONOTONIC)
                out = render(view, gaussians, pipe, bg, separate_sh=SEPARATE_SH)
                t3 = time.clock_gettime(time.CLOCK_MONOTONIC)
                renderer.update_frame(out["render"], out["depth"], frame_idx)
                torch.cuda.synchronize()
                t4 = time.clock_gettime(time.CLOCK_MONOTONIC)

                ts_dict = {
                    'frame_idx': frame_idx,
                    'renderer_t0': t0,
                    'renderer_t1': t1,
                    'renderer_t2': t2,
                    'renderer_t3': t3,
                    'renderer_t4': t4,
                }
                renderer.update_frame_info(ts_dict)

                frame_idx += 1
                n_since += 1
                now = time.perf_counter()
                if now - t_report >= 2.0:
                    logger.info(f"[server] {n_since/(now-t_report):6.1f} render FPS (frame_idx {frame_idx})")
                    t_report = now
                    n_since = 0

                dt = time.perf_counter() - t0
                if target_dt - dt > 0:
                    time.sleep(target_dt - dt)
    except KeyboardInterrupt:
        logger.info("stopping")
    finally:
        renderer.close()


if __name__ == "__main__":
    app()
