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

import sys
import time
from argparse import ArgumentParser
from contextlib import suppress

import splatbus
import torch
from loguru import logger
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.general_utils import safe_state
from utils.sh_utils import eval_sh, eval_shfs_4d

# width = 532; height = 948
width = 1600
height = 900


def print_view(view):
    if hasattr(view, "FoVx"):
        logger.info(f"FoVx: {view.FoVx}")
    else:
        logger.info("FoVx: None")
    if hasattr(view, "FoVy"):
        logger.info(f"FoVy: {view.FoVy}")
    else:
        logger.info("FoVy: None")
    if hasattr(view, "camera_center"):
        logger.info(f"Camera Center: {view.camera_center}")
    else:
        logger.info("Camera Center: None")
    if hasattr(view, "R"):
        logger.info(f"R: \n{view.R}")
    else:
        logger.info("R: None")
    if hasattr(view, "T"):
        logger.info(f"T: \n{view.T}")
    else:
        logger.info("T: None")
    if hasattr(view, "full_proj_transform"):
        logger.info(f"Full proj: \n{view.full_proj_transform}")
    else:
        logger.info("Full proj: None")
    if hasattr(view, "projection_matrix"):
        logger.info(f"Proj max: \n{view.projection_matrix}")
    else:
        logger.info("Proj max: None")
    if hasattr(view, "world_view_transform"):
        logger.info(f"world view transform: \n{view.world_view_transform}")
    else:
        logger.info("world view transform: None")


def loop_render(
    views,
    gaussians,
    pipeline,
    background,
    fps,
    time_duration,
    live_update_pts,
):
    idx = 0
    target_frame_time = 1.0 / fps
    ipc_render = splatbus.GaussianSplattingIPCRenderer(
        width=width,
        height=height,
        ipc_host="0.0.0.0",
        ipc_port=6001,
        msg_host="0.0.0.0",
        msg_port=6000,
    )
    ipc_render.init_view(width=width, height=height, view=views[idx][1])
    cam_list_views = views.viewpoint_stack[:20]
    ipc_render.set_cam_list(width=width, height=height, views=cam_list_views)

    pbar = tqdm(desc="Rendering", unit=" frame", dynamic_ncols=True)

    t_start, t_end = time_duration
    num_frames = len(views)
    frame_count = 0
    ipc_render.update_gaussians(gaussians)

    try:
        while True:
            loop_start_time = time.time()

            view: splatbus.IPCCamera = ipc_render.get_current_view().cuda()
            view.timestamp = t_start + (frame_count % num_frames) / num_frames * (
                t_end - t_start
            )
            frame_count += 1
            if live_update_pts:
                _, delta_mean = gaussians.get_current_covariance_and_mean_offset(
                    1, view.timestamp
                )
                marginal_t = gaussians.get_marginal_t(view.timestamp)
                means3D = gaussians.get_xyz + delta_mean
                shs_view = gaussians.get_features.transpose(1, 2).view(
                    -1, 3, gaussians.get_max_sh_channels
                )
                dir_pp = (
                    means3D - view.camera_center.repeat(gaussians.get_features.shape[0], 1)
                ).detach()
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                dir_t = (gaussians.get_t - view.timestamp).detach()
                sh2rgb = eval_shfs_4d(
                    gaussians.active_sh_degree,
                    gaussians.active_sh_degree_t,
                    shs_view,
                    dir_pp_normalized,
                    dir_t,
                    t_end - t_start,
                )
                # sh2rgb = eval_sh(gaussians.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
                mask = marginal_t[:, 0] > 0.05
                xyz, rgb = means3D[mask], colors_precomp[mask]
                ipc_render.update_rgb_points(xyz.detach(), rgb.detach())
            rendering = render(view, gaussians, pipeline, background)

            # Update IPC buffers
            depth_data = rendering["depth"]
            rendering_data = rendering["render"]

            # TODO: Accept channels=3. For now we'll padd alpha to 1:
            # if rendering_data.shape[0] == 3:
            #     alpha_channel = torch.ones_like(rendering_data[0:1, ...])
            #     rendering_data = torch.cat([rendering_data, alpha_channel], dim=0)
            # Update IPC buffers
            ipc_render.update_frame(rendering_data, depth_data, inverse_depth=False)

            # Control FPS
            elapsed_time = time.time() - loop_start_time
            sleep_time = target_frame_time - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

            pbar.update(1)
    except KeyboardInterrupt:
        logger.info("\n\n Stopping render loop... \n\n")
    except Exception:
        logger.exception("\n\n ERROR in render loop")
        raise
    finally:
        pbar.close()
        with suppress(Exception):
            ipc_render.close()


def main(
    dataset: ModelParams,
    pipeline: PipelineParams,
    time_duration,
    gaussian_dim,
    rot_4d,
    force_sh_3d,
    fps,
    live_update_pts,
):
    with torch.no_grad():
        gaussians = GaussianModel(
            dataset.sh_degree,
            gaussian_dim=gaussian_dim,
            time_duration=time_duration,
            rot_4d=rot_4d,
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

        loop_render(
            scene.getTestCameras(),
            gaussians,
            pipeline,
            background,
            fps,
            time_duration,
            live_update_pts,
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
    parser.add_argument("--time_duration", nargs=2, type=float, default=[0, 10])
    parser.add_argument("--num_pts", type=int, default=100_000)
    parser.add_argument("--num_pts_ratio", type=float, default=1.0)
    parser.add_argument("--rot_4d", action="store_true", default=True)
    parser.add_argument("--force_sh_3d", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--exhaust_test", action="store_true")
    parser.add_argument("--spherical_coords", action="store_true")
    parser.add_argument("--max-frames", type=int, required=False)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--live-update-pts", action="store_true", help="Whether to update RGB point cloud at every frame for SplatBus.")
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

    main(
        model.extract(args),
        pipeline.extract(args),
        args.time_duration,
        args.gaussian_dim,
        args.rot_4d,
        args.force_sh_3d,
        args.fps,
        args.live_update_pts,
    )
