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

import os
import sys
from argparse import ArgumentParser
from os import makedirs

import torch
import torchvision
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.general_utils import safe_state

from loguru import logger
import time
import splatbus
from contextlib import suppress

# width = 532; height = 948
width = 1600; height = 900

def print_view(view):
    if hasattr(view, 'FoVx'):
        logger.info(f'FoVx: {view.FoVx}')
    else:
        logger.info(f'FoVx: None')
    if hasattr(view, 'FoVy'):
        logger.info(f'FoVy: {view.FoVy}')
    else:
        logger.info(f'FoVy: None')
    if hasattr(view, 'camera_center'):
        logger.info(f'Camera Center: {view.camera_center}')
    else:
        logger.info(f'Camera Center: None')
    if hasattr(view, 'R'):
        logger.info(f'R: \n{view.R}')
    else:
        logger.info(f'R: None')
    if hasattr(view, 'T'):
        logger.info(f'T: \n{view.T}')
    else:
        logger.info(f'T: None')
    if hasattr(view, 'full_proj_transform'):
        logger.info(f'Full proj: \n{view.full_proj_transform}')
    else:
        logger.info(f'Full proj: None')
    if hasattr(view, 'projection_matrix'):
        logger.info(f'Proj max: \n{view.projection_matrix}')
    else:
        logger.info(f'Proj max: None')
    if hasattr(view, 'world_view_transform'):
        logger.info(f'world view transform: \n{view.world_view_transform}')
    else:
        logger.info(f'world view transform: None')

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view[1].cuda(), gaussians, pipeline, background)["render"]
        gt = view[0][0:3, :, :]
        torchvision.utils.save_image(
            rendering, os.path.join(render_path, "{0:05d}".format(idx) + ".png")
        )
        torchvision.utils.save_image(
            gt, os.path.join(gts_path, "{0:05d}".format(idx) + ".png")
        )

def loop_render(model_path, name, iteration, views, gaussians, pipeline, background, fps, time_duration):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

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

    try:
        while True:
            loop_start_time = time.time()

            view: splatbus.IPCCamera = ipc_render.get_current_view().cuda()
            view.timestamp = t_start + (frame_count % num_frames) / num_frames * (t_end - t_start)
            frame_count += 1
            # print_view(view)

            rendering = render(view, gaussians, pipeline, background)
            # gt = view[0][0:3, :, :]
            # torchvision.utils.save_image(
            #     gt, os.path.join(gts_path, "{0:05d}".format(idx) + ".png")
            # )

            # Update IPC buffers
            depth_data = rendering["depth"]
            rendering_data = rendering["render"]

            # torchvision.utils.save_image(
            #     rendering_data, os.path.join(render_path, "{0:05d}".format(idx) + ".png")
            # )

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
            

        ipc_render.close()
            
    except KeyboardInterrupt:
        logger.info("\n\n Stopping render loop... \n\n")
    except Exception:
        logger.exception("\n\n ERROR in render loop")
        raise
    finally:
        # pbar.close()
        with suppress(Exception):
            ipc_render.close()


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
    fps,
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

        if not skip_train:
            render_set(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
            )

        if not skip_test:
            loop_render(
                dataset.model_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
                fps,
                time_duration,
            )


# if __name__ == "__main__":
#    # Set up command line argument parser
#    parser = ArgumentParser(description="Testing script parameters")
#    model = ModelParams(parser, sentinel=True)
#    pipeline = PipelineParams(parser)
#    parser.add_argument("--iteration", default=-1, type=int)
#    parser.add_argument("--skip_train", action="store_true")
#    parser.add_argument("--skip_test", action="store_true")
#    parser.add_argument("--quiet", action="store_true")
#    args = get_combined_args(parser)
#    print("Rendering " + args.model_path)
#
#    # Initialize system state (RNG)
#    safe_state(args.quiet)
#
#
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
    parser.add_argument("--fps", type=int, default=30)
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
        args.fps,
    )
