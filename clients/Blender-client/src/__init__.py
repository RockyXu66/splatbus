bl_info = {
    "name": "Splatbus",
    "author": "Théo Morales, Yinghan Xu",
    "version": (0, 0, 1),
    "blender": (4, 2, 0),
    "category": "Scene",
    "description": "Gaussian Splating unified rendering interface",
}

import atexit
import os
import socket
import subprocess
import sys
import time
import traceback
from typing import Optional

import bpy
import numpy as np
from bpy.props import (
    BoolProperty,
    IntProperty,
    StringProperty,
    PointerProperty,
)
from mathutils import Matrix, Vector, Quaternion

import gpu
from gpu_extras.batch import batch_for_shader

try:
    import torch
except ImportError:
    torch = None  # will be caught at connect time


# ===================== TORCH INSTALL HELPERS =====================


def _get_ext_site_packages() -> Optional[str]:
    """Find the extension-local site-packages directory from sys.path."""
    for p in sys.path:
        if 'extensions' in p and 'site-packages' in p:
            return p
    return None


def _find_uv() -> Optional[str]:
    """Locate the uv binary on PATH."""
    for path_dir in os.environ.get('PATH', '').split(os.pathsep):
        candidate = os.path.join(path_dir, 'uv')
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


# ===================== STATE =====================

class _SplatbusState:
    client: Optional[object] = None
    width: int = 0
    height: int = 0

    timer: Optional[object] = None
    is_running: bool = False
    compositor_ready: bool = False
    gpu_texture: Optional[object] = None
    draw_handler: Optional[object] = None
    canonical_c2w: Optional[np.ndarray] = None
    is_rendering: bool = False
    render_end_time: float = 0.0

_state = _SplatbusState()


# ===================== PROPERTIES =====================


class SplatbusProperties(bpy.types.PropertyGroup):
    in_use: BoolProperty(
        name="Render SplatBus content",
        description="Whether to render content received from SplatBus.",
        default=True,
    )
    host: StringProperty(
        name="Server host",
        default="127.0.0.1",
    )
    ipc_port: IntProperty(
        name="IPC port",
        default=6001,
        min=1024, max=65535,
    )
    msg_port: IntProperty(
        name="Message port",
        default=6000,
        min=1024, max=65535,
    )


# ===================== CAMERA MATH =====================


def get_blender_camera_resolution(scene):
    scale = scene.render.resolution_percentage / 100.0
    w = int(scene.render.resolution_x * scale)
    h = int(scene.render.resolution_y * scale)
    return w, h


# ===================== CORE IPC FUNCTIONS =====================


# Blender camera -> OpenCV camera coordinate conversion.
_R_BCAM2CV = np.array(
    [
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float64,
)


def _matrix_to_pose(c2w: np.ndarray):
    """Extract position and xyzw quaternion from a camera-to-world matrix."""
    position = c2w[:3, 3]
    rot = Matrix(c2w[:3, :3].tolist())
    q = rot.to_quaternion()
    return (
        {k: str(v) for k, v in zip("xyz", position)},
        {k: str(v) for k, v in zip("xyzw", np.array([q.x, q.y, q.z, q.w]))},
    )


def _get_active_viewport_rv3d():
    """Return the active RegionView3D or None."""
    screen = bpy.context.screen
    if screen is None:
        return None
    active = bpy.context.area
    if active and active.type == 'VIEW_3D':
        for space in active.spaces:
            if space.type == 'VIEW_3D':
                return space.region_3d
    for area in screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    return space.region_3d
    return None


def _get_viewport_camera_matrix():
    """Return the active 3D viewport's camera-to-world matrix (OpenCV convention)."""
    rv3d = _get_active_viewport_rv3d()
    if rv3d is None:
        return None
    bcam_view = np.array(rv3d.view_matrix)
    cv_view = _R_BCAM2CV @ bcam_view
    return np.linalg.inv(cv_view)


def _get_scene_camera_matrix():
    """Return the scene camera's camera-to-world matrix (OpenCV convention)."""
    cam = bpy.context.scene.camera
    if cam is None:
        return None
    bcam_view = np.array(cam.matrix_world.inverted())
    cv_view = _R_BCAM2CV @ bcam_view
    return np.linalg.inv(cv_view)


def _colmap_pose_to_blender_c2w(t, quat_xyzw):
    """Convert a server COLMAP camera pose to a Blender/OpenGL c2w matrix."""
    x, y, z, w = quat_xyzw
    q = Quaternion((w, x, y, z))
    c2w_colmap = np.eye(4, dtype=np.float64)
    c2w_colmap[:3, :3] = np.array(q.to_matrix())
    c2w_colmap[:3, 3] = np.array(t).reshape(3)
    # Blender/OpenGL camera convention differs from COLMAP by a Y/Z flip.
    return c2w_colmap @ _R_BCAM2CV


def _apply_server_canonical_pose(t, quat_xyzw):
    """Set the scene camera to the server's canonical/test camera pose."""
    scene = bpy.context.scene
    cam_obj = scene.camera
    if cam_obj is None:
        cam_data = bpy.data.cameras.new(name="SplatBusCamera")
        cam_obj = bpy.data.objects.new(name="SplatBusCamera", object_data=cam_data)
        scene.collection.objects.link(cam_obj)
        scene.camera = cam_obj
    c2w_blender = _colmap_pose_to_blender_c2w(t, quat_xyzw)
    cam_obj.matrix_world = Matrix(c2w_blender.tolist())
    _state.canonical_c2w = c2w_blender.copy()


def _send_pose(c2w=None):
    if _state.client is None or not _state.client.connected:
        return False
    if c2w is None:
        c2w = _get_scene_camera_matrix()
    if c2w is None:
        return False

    position, rotation = _matrix_to_pose(c2w)
    _state.client.send_camera_pose(position=position, rotation=rotation)
    return True


def _receive_frames() -> dict:
    if _state.client is None or not _state.client.connected:
        return {}
    result = _state.client.receive()
    if not result:
        return {}
    frames = {}
    if "color" in result:
        tensor = result["color"]
        frames["color"] = tensor.cpu().numpy() if tensor.is_cuda else tensor.numpy()
    if "depth" in result:
        tensor = result["depth"]
        frames["depth"] = tensor.cpu().numpy() if tensor.is_cuda else tensor.numpy()
    return frames


def _safe_close_client(client):
    """Close the client without calling CUDA driver cleanup routines.

    Calling ``client.close()`` inside Blender can segfault in ``cuEventDestroy``
    because the CUDA IPC event was opened in a context that conflicts with
    Blender's own CUDA use.  We close the sockets and drop our Python references
    to the shared buffers; the OS/CUDA runtime reclaims the IPC handles when
    Blender exits.
    """
    if client is None:
        return
    try:
        client.connected = False
        client.stop_event.set()

        # Release torch tensors held by the buffers without invoking the buffer
        # ``close()`` methods that call cudaStreamDestroy / cudaIpcCloseMemHandle /
        # cudaEventDestroy.
        for buf in (client.client_buffer_color, client.client_buffer_depth, client.client_buffer_evt):
            if buf is not None:
                buf.read_buffer = None

        # Close sockets only.
        for sock_name in ("ipc_sock", "msg_sock"):
            sock = getattr(client, sock_name, None)
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    sock.close()
                except OSError:
                    pass
                setattr(client, sock_name, None)
    except Exception:
        traceback.print_exc()


def _update_depth_image(depth_np: np.ndarray):
    """Update the SplatbusDepth image from a 1‑channel depth numpy array."""
    if depth_np is None or depth_np.size == 0:
        return None
    # Server depth is already flipped by ClientBuffer to top‑down (CV2);
    # Blender expects bottom‑up (OpenGL).
    depth_np = np.flipud(depth_np)
    h, w = depth_np.shape[:2]
    img = bpy.data.images.get("SplatbusDepth")
    if img is None or img.size[0] != w or img.size[1] != h:
        if img is not None:
            bpy.data.images.remove(img)
        img = bpy.data.images.new("SplatbusDepth", width=w, height=h, alpha=False, float_buffer=True)
        img.use_fake_user = True
    flat = np.empty(w * h * 4, dtype=np.float32)
    flat[0::4] = depth_np.ravel()
    flat[1::4] = depth_np.ravel()
    flat[2::4] = depth_np.ravel()
    flat[3::4] = 1.0
    img.pixels.foreach_set(flat)
    img.update()
    img.update_tag()
    return img


def _update_blender_image(color_np: np.ndarray, touch_gpu: bool = False, alpha_mask: np.ndarray = None):
    """Update the SplatbusOutput image pixels from a numpy RGB(A) frame.

    The server's alpha channel is not meaningful (it is 1.0 everywhere), so the
    alpha channel of the Blender image is computed from ``alpha_mask`` when
    provided. When not provided it falls back to the maximum RGB value.
    """
    if color_np is None or color_np.size == 0:
        return None
    # ClientBuffer flips the server image to top-down (CV2) order; Blender's
    # Image.pixels / gpu textures expect bottom-up (OpenGL) order.
    color_np = np.flipud(color_np)
    if alpha_mask is not None:
        alpha_mask = np.flipud(alpha_mask)
    h, w = color_np.shape[:2]
    _state.width, _state.height = w, h
    img = bpy.data.images.get("SplatbusOutput")
    if img is None or img.size[0] != w or img.size[1] != h:
        if img is not None:
            bpy.data.images.remove(img)
        img = bpy.data.images.new("SplatbusOutput", width=w, height=h, alpha=True, float_buffer=True)
        img.use_fake_user = True
    flat = np.empty(w * h * 4, dtype=np.float32)
    flat[0::4] = color_np[..., 0].ravel()
    flat[1::4] = color_np[..., 1].ravel()
    flat[2::4] = color_np[..., 2].ravel()
    if alpha_mask is not None:
        flat[3::4] = alpha_mask.ravel()
    elif color_np.shape[2] == 4 and color_np[..., 3].max() < 0.999:
        flat[3::4] = color_np[..., 3].ravel()
    else:
        flat[3::4] = np.maximum(np.maximum(color_np[..., 0], color_np[..., 1]), color_np[..., 2]).ravel()
    img.pixels.foreach_set(flat)
    img.update()
    img.update_tag()
    if touch_gpu:
        img.gl_touch()
    return img


def _update_compositor_image(color_np: np.ndarray, depth_np: np.ndarray = None):
    """Update the compositor image and the viewport GPU texture."""
    alpha_mask = None
    if depth_np is not None and depth_np.size > 0:
        # Pixels with the sentinel depth value (>= 100.0) are empty background.
        alpha_mask = (depth_np < 99.9).astype(np.float32)
    img = _update_blender_image(color_np, touch_gpu=True, alpha_mask=alpha_mask)
    if img is None:
        return
    tex = gpu.texture.from_image(img)
    _state.gpu_texture = tex


# ===================== TICK =====================


def _tick():
    if not _state.is_running:
        return None
    if _state.client is None or not _state.client.connected:
        _stop()
        return None
    # Avoid touching GPU resources when Blender is shutting down or the active
    # window/context has gone away.
    if bpy.context.window is None or bpy.context.screen is None:
        _stop()
        return None

    # After a render finishes, keep GPU operations paused for a 2 s cooldown
    # so closing the render window doesn't crash Blender.
    if _state.is_rendering and _state.render_end_time > 0:
        if time.time() - _state.render_end_time > 2.0:
            _state.is_rendering = False
            _state.render_end_time = 0.0

    try:
        c2w = _get_viewport_camera_matrix()
        _send_pose(c2w)
        frames = _receive_frames()
        if frames:
            color = frames.get("color")
            depth = frames.get("depth")
            if color is not None:
                if _state.is_rendering:
                    _update_blender_image(color, alpha_mask=(depth < 99.9).astype(np.float32) if depth is not None else None)
                else:
                    _update_compositor_image(color, depth)
            if depth is not None:
                _update_depth_image(depth)
    except Exception:
        traceback.print_exc()

    return 1.0 / 30.0


# ===================== VIEWPORT DRAW =====================


def _draw_viewport():
    tex = _state.gpu_texture
    if tex is None:
        return
    region = bpy.context.region
    if region is None:
        return
    vw, vh = region.width, region.height

    shader = gpu.shader.from_builtin("IMAGE")
    shader.bind()
    shader.uniform_sampler("image", tex)

    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.depth_mask_set(False)
    with gpu.matrix.push_pop_projection():
        proj = Matrix((
            (2 / vw, 0, 0, -1),
            (0, 2 / vh, 0, -1),
            (0, 0, -1, 0),
            (0, 0, 0, 1),
        ))
        gpu.matrix.load_projection_matrix(proj)
        gpu.matrix.load_identity()
        batch = batch_for_shader(
            shader, "TRI_FAN",
            {
                "pos": ((0, 0, -1), (vw, 0, -1), (vw, vh, -1), (0, vh, -1)),
                "texCoord": ((0, 0), (1, 0), (1, 1), (0, 1)),
            },
        )
        batch.draw(shader)
    gpu.state.depth_test_set('LESS')
    gpu.state.depth_mask_set(True)


# ===================== LIFECYCLE =====================


def _start():
    if _state.is_running:
        return
    _state.is_running = True

    _state.timer = bpy.app.timers.register(_tick, first_interval=0.0, persistent=True)

    if not _state.compositor_ready:
        _setup_compositor()
        _state.compositor_ready = True

    for h in (bpy.app.handlers.frame_change_pre, bpy.app.handlers.render_init):
        if _on_frame_change not in h:
            h.append(_on_frame_change)
    if _on_render_complete not in bpy.app.handlers.render_complete:
        bpy.app.handlers.render_complete.append(_on_render_complete)

    if _state.draw_handler is None:
        _state.draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            _draw_viewport, (), 'WINDOW', 'POST_VIEW'
        )


def _stop():
    _state.is_running = False

    if _state.timer is not None:
        try:
            bpy.app.timers.unregister(_tick)
        except ValueError:
            pass
        _state.timer = None

    if _state.draw_handler is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_state.draw_handler, 'WINDOW')
        except (ValueError, TypeError):
            pass
        _state.draw_handler = None

    _state.gpu_texture = None
    _state.canonical_c2w = None

    for h in (bpy.app.handlers.frame_change_pre, bpy.app.handlers.render_init):
        try:
            h.remove(_on_frame_change)
        except ValueError:
            pass
    try:
        bpy.app.handlers.render_complete.remove(_on_render_complete)
    except ValueError:
        pass


def _on_frame_change(_scene=None, _depsgraph=None):
    if not _state.is_running:
        return
    _state.is_rendering = True
    try:
        print(f"[Splatbus] render/frame handler fired")
        c2w = _get_scene_camera_matrix()
        _send_pose(c2w)
        frames = _receive_frames()
        color = frames.get("color")
        depth = frames.get("depth")
        print(f"[Splatbus] received frame: color={color is not None}, depth={depth is not None}, shape={color.shape if color is not None else None}")
        if color is not None:
            alpha_mask = None
            if depth is not None and depth.size > 0:
                alpha_mask = (depth < 99.9).astype(np.float32)
            img = _update_blender_image(color, alpha_mask=alpha_mask)
            print(f"[Splatbus] updated color image: {img}, size={img.size if img else None}")
            print(f"[Splatbus] color shape: {color.shape}, dtype: {color.dtype}")
            print(f"[Splatbus] RGB mean: {color[..., :3].mean():.4f}, alpha mean: {alpha_mask.mean() if alpha_mask is not None else -1:.4f}")
            if color.shape[2] == 4:
                alpha = color[..., 3]
                print(f"[Splatbus] server alpha stats: min={alpha.min():.4f}, max={alpha.max():.4f}, nonzero={np.count_nonzero(alpha)}")
        if depth is not None:
            _update_depth_image(depth)
            print(f"[Splatbus] depth min={depth.min():.4f} max={depth.max():.4f}")

        # ── Force compositor Image nodes to refresh after update ──
        try:
            scene = bpy.context.scene
            if scene.node_tree is not None:
                for node in scene.node_tree.nodes:
                    if node.type == "IMAGE" and node.image is not None:
                        if node.image.name in ("SplatbusOutput", "SplatbusDepth"):
                            img = node.image
                            node.image = None
                            node.image = img
                            print(f"[Splatbus] refreshed compositor image node: {img.name}")
                scene.node_tree.update_tag()
        except Exception as refresh_err:
            print(f"[Splatbus] image node refresh failed: {refresh_err}")
    except Exception:
        traceback.print_exc()


def _on_render_complete(_scene=None):
    # Keep is_rendering=True for a 2-second cooldown after render finishes.
    # This prevents GPU touch while the render window is being opened/closed.
    _state.render_end_time = time.time()


# ===================== COMPOSITOR =====================


def _setup_compositor():
    scene = bpy.context.scene
    scene.use_nodes = True
    scene.render.film_transparent = True
    # Ensure compositing is actually executed during renders.
    try:
        scene.render.use_compositing = True
    except Exception:
        pass
    # Enable the Z‑depth pass for depth‑based compositing.
    scene.view_layers[0].use_pass_z = True
    tree = scene.node_tree
    if tree is None:
        print("[Splatbus] ERROR: scene.node_tree is None")
        return

    # Remove all existing compositor nodes so repeated clicks always give a
    # clean Splatbus setup.
    for n in list(tree.nodes):
        tree.nodes.remove(n)

    w, h = _state.width or 1920, _state.height or 1080
    print(f"[Splatbus] Setting up compositor for {w}x{h}")

    # ── Splatbus colour image ──
    img_color = bpy.data.images.get("SplatbusOutput")
    if img_color is None or img_color.size[0] != w or img_color.size[1] != h:
        if img_color is not None:
            bpy.data.images.remove(img_color)
        img_color = bpy.data.images.new(
            "SplatbusOutput", width=w, height=h, alpha=True, float_buffer=True
        )
        img_color.use_fake_user = True
        if bpy.context.window is not None:
            img_color.gl_touch()

    # ── Splatbus depth image ──
    img_depth = bpy.data.images.get("SplatbusDepth")
    if img_depth is None or img_depth.size[0] != w or img_depth.size[1] != h:
        if img_depth is not None:
            bpy.data.images.remove(img_depth)
        img_depth = bpy.data.images.new(
            "SplatbusDepth", width=w, height=h, alpha=False, float_buffer=True
        )
        img_depth.use_fake_user = True
        # Depth is a data buffer, not a colour image.
        if img_depth.colorspace_settings is not None:
            try:
                img_depth.colorspace_settings.name = "Non-Color"
            except Exception:
                pass
        if bpy.context.window is not None:
            img_depth.gl_touch()

    print(f"[Splatbus] compositor images: color={img_color}, depth={img_depth}, color.size={img_color.size[:]}, depth.size={img_depth.size[:]}")

    rl = tree.nodes.new("CompositorNodeRLayers")
    rl.location = (-900, 0)

    node_color = tree.nodes.new("CompositorNodeImage")
    node_color.location = (-900, 300)
    node_color.image = img_color

    node_depth = tree.nodes.new("CompositorNodeImage")
    node_depth.location = (-900, 550)
    node_depth.image = img_depth

    # ── Depth-based compositing ──
    # The server's alpha channel is 1.0 everywhere, so we build a mask from the
    # depth buffer instead:
    #   mask = (SplatbusDepth < BlenderZ  OR  BlenderAlpha == 0)
    #          AND (SplatbusDepth < 100.0)
    # The first term occludes Blender geometry that is behind the splats. The
    # second term handles transparent background pixels, whose Z-pass is 0. The
    # third term discards empty background pixels from the server.

    # Extract depth value from the SplatbusDepth image (stored in RGB channels).
    sep_depth = tree.nodes.new("CompositorNodeSepRGBA")
    sep_depth.location = (-700, 550)
    tree.links.new(node_depth.outputs["Image"], sep_depth.inputs[0])

    # Empty-pixel guard: depth < 99.9 → 1.0, else 0.0
    empty_mask = tree.nodes.new("CompositorNodeMath")
    empty_mask.operation = "LESS_THAN"
    empty_mask.inputs[1].default_value = 99.9
    empty_mask.location = (-500, 700)
    tree.links.new(sep_depth.outputs["R"], empty_mask.inputs[0])

    # Depth comparison: splatbus depth < Blender Z-pass → 1.0, else 0.0
    depth_diff = tree.nodes.new("CompositorNodeMath")
    depth_diff.operation = "SUBTRACT"
    depth_diff.location = (-500, 500)
    tree.links.new(sep_depth.outputs["R"], depth_diff.inputs[0])
    tree.links.new(rl.outputs["Depth"], depth_diff.inputs[1])

    depth_mask_lt = tree.nodes.new("CompositorNodeMath")
    depth_mask_lt.operation = "LESS_THAN"
    depth_mask_lt.inputs[1].default_value = 0.0
    depth_mask_lt.location = (-300, 500)
    tree.links.new(depth_diff.outputs[0], depth_mask_lt.inputs[0])

    # Background mask: Blender render alpha == 0 → 1.0, else 0.0
    bg_mask = tree.nodes.new("CompositorNodeMath")
    bg_mask.operation = "LESS_THAN"
    bg_mask.inputs[1].default_value = 0.001
    bg_mask.location = (-500, 300)
    tree.links.new(rl.outputs["Alpha"], bg_mask.inputs[0])

    # Combine: show Splatbus if it is closer OR if Blender has no geometry.
    depth_mask = tree.nodes.new("CompositorNodeMath")
    depth_mask.operation = "MAXIMUM"
    depth_mask.location = (-100, 500)
    tree.links.new(depth_mask_lt.outputs[0], depth_mask.inputs[0])
    tree.links.new(bg_mask.outputs[0], depth_mask.inputs[1])

    # Final mask = depth_mask * empty_mask
    final_mask = tree.nodes.new("CompositorNodeMath")
    final_mask.operation = "MULTIPLY"
    final_mask.location = (100, 600)
    tree.links.new(depth_mask.outputs[0], final_mask.inputs[0])
    tree.links.new(empty_mask.outputs[0], final_mask.inputs[1])

    # Composite: show Splatbus where mask=1, Blender render where mask=0.
    # In Blender 4.x MixRGB, Fac=1 selects input 2 and Fac=0 selects input 1,
    # so Splatbus must be wired to input 2 and the render to input 1.
    mix = tree.nodes.new("CompositorNodeMixRGB")
    mix.location = (300, 0)
    mix.blend_type = "MIX"

    composite = tree.nodes.new("CompositorNodeComposite")
    composite.location = (600, 0)

    tree.links.new(rl.outputs["Image"], mix.inputs[1])
    tree.links.new(node_color.outputs["Image"], mix.inputs[2])
    tree.links.new(final_mask.outputs[0], mix.inputs[0])
    tree.links.new(mix.outputs["Image"], composite.inputs["Image"])

    # Force the node tree to refresh.
    tree.update_tag()
    print(f"[Splatbus] Compositor setup complete: {len(tree.nodes)} nodes")


# ===================== OPERATORS =====================


class SplatbusConnectOperator(bpy.types.Operator):
    bl_idname = "splatbus.connect"
    bl_label = "Connect to SplatBus"
    bl_description = "Connect to the SplatBus Gaussian Splatting server and start the render loop"

    def execute(self, context):
        if _state.is_running:
            self.report({"INFO"}, "Already connected")
            return {"CANCELLED"}

        if torch is None:
            self.report(
                {"ERROR"},
                "PyTorch is not available. SplatBus requires PyTorch with CUDA support.",
            )
            return {"CANCELLED"}

        try:
            from splatbus import GaussianSplattingIPCClient
        except ImportError as e:
            self.report(
                {"ERROR"},
                f"Cannot import splatbus: {e}. Make sure PyTorch is installed.",
            )
            return {"CANCELLED"}

        props = context.scene.splatbus_setup
        _state.client = GaussianSplattingIPCClient(
            host=props.host, ipc_port=props.ipc_port, msg_port=props.msg_port,
        )
        try:
            _state.client.connect()
        except Exception as e:
            self.report({"ERROR"}, f"Failed to connect: {e}")
            _state.client = None
            return {"CANCELLED"}

        scene = context.scene
        # Use the server's fixed rendering resolution for the shared image so the
        # compositor image does not get recreated (and lose its GPU texture) on
        # the first render.
        sv_w, sv_h = _state.client.get_viewport_size()
        print(f"[Splatbus] Server viewport size: {sv_w}x{sv_h}")
        print(f"[Splatbus] Blender render resolution: {scene.render.resolution_x}x{scene.render.resolution_y} ({scene.render.resolution_percentage}%)")
        if sv_w <= 0 or sv_h <= 0:
            sv_w, sv_h = get_blender_camera_resolution(scene)
        _state.width, _state.height = sv_w, sv_h
        img = bpy.data.images.get("SplatbusOutput")
        if img is None or img.size[0] != sv_w or img.size[1] != sv_h:
            if img is not None:
                bpy.data.images.remove(img)
            img = bpy.data.images.new(
                "SplatbusOutput", width=sv_w, height=sv_h, alpha=True, float_buffer=True
            )
            img.use_fake_user = True
            if bpy.context.window is not None:
                img.gl_touch()

        t, quat = _state.client.get_camera_pose(cam_idx=0)
        print(f"[Splatbus] Server initial pose — t: {t}, quat: {quat}")
        _apply_server_canonical_pose(t, quat)

        _start()
        self.report({"INFO"}, "Connected to SplatBus")
        return {"FINISHED"}


class SplatbusDisconnectOperator(bpy.types.Operator):
    bl_idname = "splatbus.disconnect"
    bl_label = "Disconnect SplatBus"
    bl_description = "Disconnect from the SplatBus server and stop the render loop"

    def execute(self, context):
        _stop()
        if _state.client is not None:
            _safe_close_client(_state.client)
            _state.client = None
        self.report({"INFO"}, "Disconnected from SplatBus")
        return {"FINISHED"}


class SplatbusInstallTorchOperator(bpy.types.Operator):
    bl_idname = "splatbus.install_torch"
    bl_label = "Install PyTorch"
    bl_description = "Download and install PyTorch with CUDA support"

    _process = None
    _timer = None
    _elapsed = 0

    def modal(self, context, event):
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}
        if self._process is None:
            return {'FINISHED'}

        wm = context.window_manager
        ret = self._process.poll()
        if ret is None:
            self._elapsed += 1
            secs = self._elapsed
            wm.progress_update(secs % 100)
            if secs <= 15 or secs % 5 == 0:
                self.report({'INFO'}, f"Installing PyTorch... ({secs}s elapsed)")
            return {'PASS_THROUGH'}

        if self._timer is not None:
            wm.event_timer_remove(self._timer)
        wm.progress_end()

        stdout, stderr = self._process.communicate()
        if ret != 0:
            msg = (stderr or b'').decode(errors='replace')[:300]
            self.report({'ERROR'}, f"Installation failed: {msg}")
            return {'FINISHED'}

        self.report({'INFO'}, "PyTorch installed! Please save your work and restart Blender.")
        return {'FINISHED'}

    def invoke(self, context, event):
        target = _get_ext_site_packages()
        if target is None:
            self.report({'ERROR'}, "Cannot locate extension site-packages directory")
            return {'CANCELLED'}

        blender_python = sys.executable
        uv_path = _find_uv()
        # TODO: automatic versioning
        idx_url = "https://download.pytorch.org/whl/cu124"

        if uv_path:
            cmd = [uv_path, 'pip', 'install', 'torch',
                   '--index-url', idx_url,
                   '--python', blender_python,
                   '--target', target]
        else:
            cmd = [blender_python, '-m', 'pip', 'install', 'torch',
                   '--index-url', idx_url,
                   '--target', target]

        self._elapsed = 0
        self.report({'INFO'}, "Installing PyTorch...")
        self._process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        wm = context.window_manager
        wm.progress_begin(0, 100)
        self._timer = wm.event_timer_add(1.0, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}


class SplatbusSetupCompositorOperator(bpy.types.Operator):
    bl_idname = "splatbus.setup_compositor"
    bl_label = "Setup Compositor"
    bl_description = "Create compositor nodes to composite SplatBus output over the final render"

    def execute(self, context):
        _state.width, _state.height = get_blender_camera_resolution(context.scene)
        _setup_compositor()
        _state.compositor_ready = True
        self.report({"INFO"}, "Compositor nodes created")
        return {"FINISHED"}


# ===================== PANEL =====================


class SCENE_PT_splatbus(bpy.types.Panel):
    bl_label = "SplatBus"
    bl_idname = "SCENE_PT_splatbus"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SplatBus"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        props = context.scene.splatbus_setup

        if torch is None:
            box = layout.box()
            box.label(text="PyTorch not found", icon="ERROR")
            box.label(text="SplatBus requires PyTorch with CUDA.")
            row = box.row()
            row.scale_y = 1.5
            row.operator("splatbus.install_torch", icon="IMPORT")
            return

        box = layout.box()
        box.label(text="Connection", icon="WORLD_DATA")
        box.prop(props, "host")
        row = box.row()
        row.prop(props, "ipc_port")
        row.prop(props, "msg_port")

        row = layout.row()
        row.scale_y = 1.4
        if _state.is_running:
            row.operator("splatbus.disconnect", icon="PAUSE")
        else:
            row.operator("splatbus.connect", icon="PLAY")

        layout.separator()

        box = layout.box()
        box.label(text="Compositor", icon="NODETREE")
        box.prop(props, "in_use")
        box.operator("splatbus.setup_compositor", icon="NODETREE")

        col = layout.column()
        col.enabled = _state.is_running
        row = col.row()
        row.scale_y = 1.75
        row.scale_x = 1.75

        if not _state.is_running:
            info = layout.box()
            info.label(text="Not connected", icon="ERROR")
        else:
            box = layout.box()
            box.label(text=f"Streaming {_state.width}x{_state.height}", icon="RENDER_RESULT")


# ===================== REGISTRATION =====================


classes = (
    SplatbusProperties,
    SplatbusConnectOperator,
    SplatbusDisconnectOperator,
    SplatbusInstallTorchOperator,
    SplatbusSetupCompositorOperator,
    SCENE_PT_splatbus,
)


@bpy.app.handlers.persistent
def _load_handler(_dummy):
    pass  # placeholder for future reload-state logic


@bpy.app.handlers.persistent
def _shutdown_handler(_dummy=None):
    """Stop the render loop before file load or Blender shutdown."""
    _stop()
    if _state.client is not None:
        _safe_close_client(_state.client)
        _state.client = None


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.splatbus_setup = PointerProperty(type=SplatbusProperties)
    bpy.app.handlers.load_post.append(_load_handler)
    bpy.app.handlers.load_pre.append(_shutdown_handler)
    atexit.register(_shutdown_handler)


def unregister():
    _shutdown_handler()
    atexit.unregister(_shutdown_handler)

    bpy.app.handlers.load_post.remove(_load_handler)
    try:
        bpy.app.handlers.load_pre.remove(_shutdown_handler)
    except ValueError:
        pass

    del bpy.types.Scene.splatbus_setup
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
