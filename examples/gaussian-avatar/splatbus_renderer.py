"""
SplatBus Renderer - mmlphuman Rendering
Modified from test.py to work with SplatBus IPC pipeline

Usage:
    python splatbus_renderer.py --config ./config/CONFIG_FILE.yaml --model_dir <MODEL_DIR> --out_dir <IMAGE_OUT_DIR> --cam_path <CAM_PATH> --pose_path <POSE_PATH> --fps 30 --test
"""
import os
from os import path
import torch
from omegaconf import OmegaConf
import sys
from tqdm import tqdm
import numpy as np
import random
from dataclasses import dataclass
import pickle
import copy
import json
from argparse import ArgumentParser
import copy
from scipy.spatial.transform import Rotation
import imageio.v3 as iio
from torch.utils.data import DataLoader

from scene.dataset import get_dataset_type, data_to_cam
from scene.gaussian_model import GaussianModel
from scene.net_vis import load_model
from utils.config_utils import Config
from utils.image_utils import encode_bytes
from utils.smpl_utils import init_smpl_pose

import time
from contextlib import suppress
from loguru import logger
import splatbus

width = 900; height = 1600

logger.info("=" * 60)
logger.info("SplatBus Renderer - mmlphuman Rendering")
logger.info("=" * 60)

def fovx_to_intrinsic(fovx, H, W):
    focal = W / 2 / np.tan(fovx/2)
    K = np.zeros((3, 3))
    K[0, 0] = focal
    K[1, 1] = focal
    K[2, 2] = 1
    K[0, 2], K[1, 2] = W/2, H/2
    return K.astype(np.float32)

def load_amass_pose_list(pose_path):
    data = np.load(pose_path)
    pose_list = []
    poses = data['poses'].astype(np.float32)
    if 'pca_poses' in data:
        pca_poses = data['pca_poses'].astype(np.float32)
    else:
        pca_poses = np.zeros_like(poses)
    trans = data['trans'].astype(np.float32)
    N = len(poses)

    # AMASS poses are noisy
    OPTIMIZE_AMASS = False
    if OPTIMIZE_AMASS:
        foo = poses[:,3:]
        foo[:, 13 * 3 + 2] -= 0.25
        foo[:, 12 * 3 + 2] += 0.25
        foo[:, 19 * 3: 20 * 3] = 0.
        foo[:, 20 * 3: 21 * 3] = 0.
        foo[:, 14 * 3] = 0.

        poses[:,3:] = foo

        # smooth
        win_size = 1
        poses_clone = np.copy(poses)
        trans_clone = np.copy(trans)
        frame_num = poses_clone.shape[0]
        poses[win_size: frame_num-win_size] = 0
        trans[win_size: frame_num-win_size] = 0
        for i in range(-win_size, win_size + 1):
            poses[win_size: frame_num-win_size] += poses_clone[win_size+i: frame_num-win_size+i]
            trans[win_size: frame_num-win_size] += trans_clone[win_size+i: frame_num-win_size+i]
        poses[win_size: frame_num-win_size] /= (2 * win_size + 1)
        trans[win_size: frame_num-win_size] /= (2 * win_size + 1)

    for i in range(N):
        pose_list.append(dict(pose=poses[i], pca_pose=pca_poses[i], Th=trans[i], Rh=np.eye(3, dtype=np.float32)))

    return pose_list

def load_thuman_pose_list(pose_path):
    smpl_params = np.load(pose_path, allow_pickle=True)
    smpl_params = dict(smpl_params)

    pose_list = []
    N = len(smpl_params['global_orient'])
    for frame_id in range(N):
        pose = np.concatenate([smpl_params['global_orient'][frame_id],
                    smpl_params['body_pose'][frame_id],
                    np.zeros(3,dtype=np.float32),
                    np.zeros(6,dtype=np.float32),
                    smpl_params['left_hand_pose'][frame_id],
                    smpl_params['right_hand_pose'][frame_id],], axis=0)
        Th = smpl_params['transl'][frame_id]
        Rh = np.eye(3, dtype=np.float32)
        pose_list.append(dict(pose=pose, Th=Th, Rh=Rh))
    return pose_list

def testing_novel_cam_pose_speed(gaussians: GaussianModel, out_dir, frame_ids, pose_list, cam, background):

    # warm up
    pose = pose_list[0]
    gaussians.smpl_poses = torch.as_tensor(pose['pose'])
    gaussians.Th = torch.as_tensor(pose['Th'])
    gaussians.Rh = torch.as_tensor(pose['Rh'])
    image, alpha, info = gaussians.render(cam, background=background)
    torch.cuda.synchronize()

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    iter_start.record()

    for frame_id in frame_ids:
        pose = pose_list[frame_id]
        gaussians.smpl_poses = torch.as_tensor(pose['pose'])
        gaussians.Th = torch.as_tensor(pose['Th'])
        gaussians.Rh = torch.as_tensor(pose['Rh'])

        image, alpha, info = gaussians.render(cam, background=background)

        image = (torch.clamp(image, min=0, max=1.0) * 255).byte().contiguous()
        torch.cuda.synchronize()

    iter_end.record()
    torch.cuda.synchronize()

    run_time = iter_start.elapsed_time(iter_end)
    fps = len(frame_ids) / run_time * 1000
    print('Running time:', run_time)
    print('FPS:', fps)

@dataclass
class Camera:
    R: np.ndarray
    T: np.ndarray
    FoVx: float
    FoVy: float

def testing_novel_cam_pose(gaussians: GaussianModel, out_dir, frame_ids, pose_list, cam, background, fps):

    os.makedirs(path.join(out_dir), exist_ok=True)
    
    idx = 0        # pose index

    logger.info(f'Render with {fps} FPS.')
     
    pbar = tqdm(desc="Rendering", unit=" frame", dynamic_ncols=True)
    target_frame_time = 1.0 / fps

    w2c = cam['w2c'].detach().cpu().numpy()
    c2w = np.linalg.inv(w2c)
    R = c2w[:3, :3]             # R: camera to world rotation
    T = w2c[:3, 3]              # T: world to camera translation
    fovX = np.deg2rad(cam['fovx'])
    fovY = 2 * np.arctan(height / width * np.tan(fovX / 2))

    cam_info = Camera(R=R, T=T, FoVx=fovX, FoVy=fovY)
    ipc_render = splatbus.GaussianSplattingIPCRenderer(
        width=width, 
        height=height,
        ipc_host="0.0.0.0",
        ipc_port=6001,
        msg_host="0.0.0.0",
        msg_port=6000,
    )
    ipc_render.init_view(width=width, height=height, view=cam_info)
    ipc_render.set_cam_list(width=width, height=height, views=[cam_info])

    try:
        while True:
            loop_start_time = time.time()
            frame_id = idx % len(frame_ids)

            pose = pose_list[frame_id]
            pose = copy.deepcopy(pose)

            gaussians.smpl_poses = torch.as_tensor(pose['pose']).cpu()
            if 'pca_pose' in pose: gaussians.smpl_pca_poses = torch.as_tensor(pose['pca_pose'])
            gaussians.Th = torch.clone(torch.as_tensor(pose['Th']).cpu())
            gaussians.Rh = torch.as_tensor(pose['Rh']).cpu()

            view: splatbus.IPCCamera = ipc_render.get_current_view().cuda()
            focal_x = splatbus.camera.fov2focal(view.fov_x, width)
            focal_y = splatbus.camera.fov2focal(view.fov_y, height)
            K = torch.tensor([
                [focal_x,         0,      width/2],
                [      0,   focal_y,     height/2],
                [      0,         0,            1]
            ])
            cam['width'] = width
            cam['height'] = height
            cam['K'] = K.cuda()
            # gsplat use row-major matrix, so transpose column-major matrix (provided by the client) to row-major matrix
            cam['w2c'] = view.world_view_transform.transpose(0, 1)

            image, _, _ = gaussians.render(cam, background=background, with_depth=True)

            assert image.shape[2] == 4, 'Depth image should be 4 channels'
            image[:,:, 3:4] = torch.where(image[:,:, 3:4] == 0, 100.0, image[:,:, 3:4])
            image = image.permute(2, 0, 1)
            ipc_render.update_frame(image[:3, :, :], image[3:4, :, :], inverse_depth=False)

            if pbar.n == 0:
                import torchvision
                torchvision.utils.save_image(image[:3, :, :], f'{out_dir}/novel_pose_{frame_id:08d}.png')
            idx += 1

            pbar.update(1)

            # Control FPS
            elapsed_time = time.time() - loop_start_time
            sleep_time = target_frame_time - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

        ipc_render.close()
            
    except KeyboardInterrupt:
        logger.info("\n\n Stopping render loop... \n\n")
    except Exception:
        logger.exception("\n\n ERROR in render loop")
        raise
    finally:
        pbar.close()
        with suppress(Exception):
            ipc_render.close()


def testing_dataset(gaussians: GaussianModel, out_dir, dataset, background):
    test_dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    for k in ['gt', 'result', 'mask']:
        os.makedirs(path.join(out_dir, k), exist_ok=True)

    for cam in tqdm(test_dataloader):
        cam = data_to_cam(cam, non_blocking=False)
        frame_id = cam['frame_id']
        gaussians.smpl_poses = cam['pose']
        gaussians.Th, gaussians.Rh = cam['Th'], cam['Rh']

        image, alpha, info = gaussians.render(cam, background=background)

        image = (torch.clamp(image, min=0, max=1.0) * 255).byte().contiguous().cpu().numpy()

        image_gt = cam['image']
        image_gt[~cam['mask']] = background
        image_gt = (image_gt * 255).byte().contiguous().cpu().numpy()
        mask = cam['mask'].byte().contiguous().cpu().numpy() * 255

        iio.imwrite(path.join(out_dir, f'gt/{frame_id:08d}.png'), image_gt)
        iio.imwrite(path.join(out_dir, f'result/{frame_id:08d}.png'), image)
        iio.imwrite(path.join(out_dir, f'mask/{frame_id:08d}.png'), mask)


@torch.no_grad()
def testing(args: Config):
    init_smpl_pose()

    gaussians = load_model(args.model_dir)
    gaussians.is_test = args.test.is_test
    gaussians.prepare_test()
    background = torch.as_tensor(np.array(args.background)).float().cuda()

    # Dataset
    test_frame_ids = np.arange(args.test.begin_ith_frame, args.test.begin_ith_frame+args.test.frame_interval*args.test.num_frame, args.test.frame_interval).tolist()
    test_cam_ids = np.array(args.test.cam_ids).tolist()

    if args.test.cam_path is not None and args.test.pose_path is not None:
        with open(args.test.cam_path, 'r') as file:
            cam = json.load(file)
        cam['w2c'] = torch.as_tensor(np.array(cam['w2c']).reshape(4,4)).float().cuda()
        K = fovx_to_intrinsic(cam['fovx'] / 180 * np.pi, cam['height'], cam['width'])
        cam['K'] = torch.as_tensor(K).cuda()

        if 'smpl_params.npz' in args.test.pose_path:
            pose_list = load_thuman_pose_list(args.test.pose_path)
        else:
            pose_list = load_amass_pose_list(args.test.pose_path)

        if args.test.test_speed:
            testing_novel_cam_pose_speed(gaussians, args.out_dir, test_frame_ids, pose_list, cam, background)
        else:
            testing_novel_cam_pose(gaussians, args.out_dir, test_frame_ids, pose_list, cam, background, args.fps)
    else:
        DatasetType = get_dataset_type(args.data_dir)
        testset = DatasetType(
            datadir=args.data_dir,
            frame_ids=test_frame_ids,
            cam_ids=test_cam_ids,
            background=np.array(args.background),
            image_scaling=args.image_scaling,
        )

        testing_dataset(gaussians, args.out_dir, testset, background)

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing")

    parser.add_argument('--config', type=str, default='')
    parser.add_argument('--model_dir', type=str, default='')
    parser.add_argument('--out_dir', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='')

    parser.add_argument('--cam_path', type=str, default=None)
    parser.add_argument('--pose_path', type=str, default=None)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test_speed', action='store_true')
    parser.add_argument('--fps', type=int, default=30)
    pargs = parser.parse_args(sys.argv[1:])

    args = OmegaConf.load(pargs.config)
    args.data_dir, args.out_dir, args.model_dir, args.test.cam_path, args.test.pose_path = pargs.data_dir, pargs.out_dir, pargs.model_dir, pargs.cam_path, pargs.pose_path
    args.test.is_test, args.test.test_speed = pargs.test, pargs.test_speed
    args.fps = pargs.fps
    torch.backends.cuda.matmul.allow_tf32 = True

    testing(args)
