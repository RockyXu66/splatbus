from argparse import ArgumentParser

import splatbus
import torch
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.general_utils import safe_state

try:
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_set(views, gaussians, pipeline, background, train_test_exp):
    width, height = views[0].image_width, views[0].image_height
    ipc_server = splatbus.GaussianSplattingIPCRenderer(
        width,
        height,
        ipc_host="0.0.0.0",
        msg_host="0.0.0.0",
        ipc_port=6001,
        msg_port=6000,
    )

    try:
        ipc_server.init_view(width, height, views[0])
    except Exception as e:
        print(e)
        ipc_server.close()
    running = True
    while running:
        try:
            view: splatbus.IPCCamera = ipc_server.get_current_view().cuda()
            rendering = render(view, gaussians, pipeline, background)["render"]
            if train_test_exp:
                rendering = rendering[..., rendering.shape[-1] // 2 :]
            # TODO: Accept channels=3. For now we'll padd alpha to 1:
            if rendering.shape[0] == 3:
                alpha_channel = torch.ones_like(rendering[0:1, ...])
                rendering = torch.cat([rendering, alpha_channel], dim=0)
            depth_data = torch.zeros_like(rendering)[0][None, ...]
            assert rendering.shape[0] == 4, "Expected rendering to have shape (4, H, W)"
            ipc_server.update_frame(
                color_data=rendering, depth_data=depth_data, inverse_depth=False
            )
        except KeyboardInterrupt:
            running = False
            ipc_server.close()
        except:
            ipc_server.close()


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_set(
            scene.getTrainCameras(),
            gaussians,
            pipeline,
            background,
            dataset.train_test_exp,
        )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args))
