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
import argparse

GS_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "..", "examples", "gaussian-splatting"))
sys.path.insert(0, GS_DIR)

import numpy as np
import torch
from loguru import logger

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cameras", default=None)
    ap.add_argument("--iteration", type=int, default=30000)
    ap.add_argument("--sh-degree", type=int, default=3)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=1000, help="render cap; high = run as fast as possible")
    ap.add_argument("--ipc-port", type=int, default=6001)
    ap.add_argument("--msg-port", type=int, default=6000)
    ap.add_argument("--orbit", action="store_true", help="auto-cycle through poses if no client drives the camera")
    args = ap.parse_args()

    cam_json = args.cameras or os.path.join(args.model, "cameras.json")
    ply = os.path.join(args.model, "point_cloud", f"iteration_{args.iteration}", "point_cloud.ply")
    assert os.path.isfile(ply), f"missing {ply}"

    logger.info(f"Loading {ply}")
    gaussians = GaussianModel(args.sh_degree)
    gaussians.load_ply(ply)
    gaussians.active_sh_degree = gaussians.max_sh_degree
    logger.info(f"Loaded {gaussians.get_xyz.shape[0]:,} Gaussians")

    bg = torch.zeros(3, dtype=torch.float32, device="cuda")
    pipe = Pipe()
    cams = load_cams(cam_json, args.width, args.height)
    logger.info(f"{len(cams)} poses; resolution {args.width}x{args.height}; fps cap {args.fps}")

    ipc_render = splatbus.GaussianSplattingIPCRenderer(
        width=args.width, height=args.height,
        ipc_host="0.0.0.0", ipc_port=args.ipc_port,
        msg_host="0.0.0.0", msg_port=args.msg_port,
    )
    ipc_render.init_view(width=args.width, height=args.height, view=cams[0])
    ipc_render.set_cam_list(width=args.width, height=args.height, views=cams)

    target_dt = 1.0 / args.fps
    frame_idx = 0
    t_report = time.perf_counter()
    n_since = 0
    try:
        with torch.no_grad():
            while True:
                t0 = time.clock_gettime(time.CLOCK_MONOTONIC)
                if args.orbit:
                    base = splatbus.IPCCamera.init_from_view(args.width, args.height, cams[frame_idx % len(cams)])
                    view = base.cuda()
                else:
                    view = ipc_render.get_current_view().cuda()
                t1 = time.clock_gettime(time.CLOCK_MONOTONIC)
                ipc_render.update_gaussians(gaussians)
                t2 = time.clock_gettime(time.CLOCK_MONOTONIC)
                out = render(view, gaussians, pipe, bg, separate_sh=SEPARATE_SH)
                t3 = time.clock_gettime(time.CLOCK_MONOTONIC)
                ipc_render.update_frame(out["render"], out["depth"], frame_idx)
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
                ipc_render.update_frame_info(ts_dict)

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
        ipc_render.close()


if __name__ == "__main__":
    main()
