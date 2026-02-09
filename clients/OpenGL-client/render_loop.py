#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import signal
import sys
from argparse import ArgumentParser

import torch
from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import GaussianModel, render
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from scene import Scene
from utils.general_utils import safe_state

from splatbus import GaussianSplattingIPCRenderer

running = True


def signal_handler(sig, frame):
    global running
    print("Exiting...")
    running = False


signal.signal(signal.SIGINT, signal_handler)


def render_set(views, gaussians, pipeline, background):
    global running
    ipc_server = GaussianSplattingIPCRenderer()
    

    while running:
        for view in views:
            rendering = render(view[1].cuda(), gaussians, pipeline, background)[
                "render"
            ]
            # rendering = rendering[:, :948, :532]
            depth_data = torch.zeros_like(rendering)
            assert rendering.shape[0] == 3, "Expected rendering to have shape (3, H, W)"
            ipc_server.update_frame(
                color_data=rendering, depth_data=None, inverse_depth=False
            )
    ipc_server.close()

def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    time_duration,
    gaussian_dim,
    rot_4d,
    force_sh_3d,
):
    with torch.no_grad():
        gaussians = GaussianModel(
            dataset.sh_degree,
            gaussian_dim=4,
            time_duration=time_duration,
            rot_4d=True,
            force_sh_3d=force_sh_3d,
            sh_degree_t=2 if pipeline.eval_shfs_4d else 0,
        )
        scene = Scene(
            dataset,
            gaussians,
            shuffle=False,
            time_duration=time_duration,
        )
        if gaussians.env_map is not None:
            gaussians.env_map = gaussians.env_map.cuda()

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_set(
            scene.getTestCameras(),
            gaussians,
            pipeline,
            background,
        )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--config", type=str)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--3DGS", dest="use_3dgs", action="store_true")
    parser.add_argument("--gaussian_dim", type=int, default=3)
    parser.add_argument("--time_duration", nargs=2, type=float, default=[-0.5, 0.5])
    parser.add_argument("--num_pts", type=int, default=100_000)
    parser.add_argument("--num_pts_ratio", type=float, default=1.0)
    parser.add_argument("--rot_4d", action="store_true")
    parser.add_argument("--force_sh_3d", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--exhaust_test", action="store_true")
    parser.add_argument("--spherical_coords", action="store_true")
    parser.add_argument("--max-frames", type=int, required=False)
    args = parser.parse_args(sys.argv[1:])
    # args.save_iterations.append(args.iterations)

    cfg = OmegaConf.load(args.config)

    def recursive_merge(key, host):
        if isinstance(host[key], DictConfig):
            for key1 in host[key].keys():
                recursive_merge(key1, host[key])
        else:
            assert hasattr(args, key), key
            # TODO: If the arguments are specified in the CLI,we shouldn't setattr from
            # the config! But there seems to be no way of differentiating whether the
            # arg was set in the CLI or is in its default value... This is just flawed.
            # NOTE: I need this stupid hack
            if key == "loaded_pth":
                return
            setattr(args, key, host[key])

    for k in cfg.keys():
        recursive_merge(k, cfg)

    # args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        args.time_duration,
        args.gaussian_dim,
        args.rot_4d,
        args.force_sh_3d,
    )
