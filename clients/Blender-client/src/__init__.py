bl_info = {
    "name": "Splatbus",
    "author": "Théo Morales, Yinghan Xu",
    "version": (0, 0, 1),
    "blender": (4, 2, 0),
    "category": "Scene",
    "description": "Gaussian Splating unified rendering interface",
}

import sys
import traceback
from typing import Optional

import bpy
import gpu
import numpy as np
from bpy.props import (
    BoolProperty,
    IntProperty,
    StringProperty,
    PointerProperty,
)
from gpu.types import GPUTexture
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix, Vector

try:
    import torch
except ImportError:
    from unittest.mock import MagicMock
    mock_torch = MagicMock()
    mock_torch.device = MagicMock()
    mock_torch.Tensor = MagicMock()
    sys.modules['torch'] = mock_torch

from splatbus import GaussianSplattingIPCClient
from scipy.spatial.transform import Rotation as SciRot


# ===================== STATE =====================

class _SplatbusState:
    client: Optional[GaussianSplattingIPCClient] = None
    width: int = 0
    height: int = 0

    color_buffer_np: Optional[np.ndarray] = None
    gpu_texture: Optional[GPUTexture] = None

    draw_handler: Optional[object] = None
    timer: Optional[object] = None
    is_running: bool = False
    compositor_ready: bool = False

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


def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == "VERTICAL":
        return sensor_y
    return sensor_x


def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == "AUTO":
        if size_x >= size_y:
            return "HORIZONTAL"
        else:
            return "VERTICAL"
    return sensor_fit


def get_calibration_matrix_K_from_blender(camd):
    assert isinstance(camd, bpy.types.Camera)
    if camd.type != "PERSP":
        raise ValueError("Non-perspective cameras not supported")
    scene = bpy.context.scene
    f_in_mm = camd.lens
    scale = scene.render.resolution_percentage / 100
    resolution_x_in_px = scale * scene.render.resolution_x
    resolution_y_in_px = scale * scene.render.resolution_y
    sensor_size_in_mm = get_sensor_size(
        camd.sensor_fit, camd.sensor_width, camd.sensor_height
    )
    sensor_fit = get_sensor_fit(
        camd.sensor_fit,
        scene.render.pixel_aspect_x * resolution_x_in_px,
        scene.render.pixel_aspect_y * resolution_y_in_px,
    )
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    if sensor_fit == "HORIZONTAL":
        view_fac_in_px = resolution_x_in_px
    else:
        view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px
    pixel_size_mm_per_px = sensor_size_in_mm / f_in_mm / view_fac_in_px
    s_u = 1 / pixel_size_mm_per_px
    s_v = 1 / pixel_size_mm_per_px / pixel_aspect_ratio

    u_0 = resolution_x_in_px / 2 - camd.shift_x * view_fac_in_px
    v_0 = resolution_y_in_px / 2 + camd.shift_y * view_fac_in_px / pixel_aspect_ratio
    skew = 0

    K = Matrix(((s_u, skew, u_0), (0, s_v, v_0), (0, 0, 1)))
    return K, {
        "width": resolution_x_in_px * scale,
        "height": resolution_y_in_px * scale,
        "focal_len": f_in_mm,
    }


def get_3x4_RT_matrix_from_blender(cam, to_cv: bool):
    R_bcam2cv = Matrix(
        (
            (1, 0, 0),
            (0, -1, 0),
            (0, 0, -1),
        )
    )

    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()
    T_world2bcam = -1 * R_world2bcam @ location

    R_world2cv = R_bcam2cv @ R_world2bcam
    T_world2cv = R_bcam2cv @ T_world2bcam

    cvRT = Matrix(
        (
            R_world2cv[0][:] + (T_world2cv[0],),
            R_world2cv[1][:] + (T_world2cv[1],),
            R_world2cv[2][:] + (T_world2cv[2],),
        )
    )
    glRT = Matrix(
        (
            R_world2bcam[0][:] + (T_world2bcam[0],),
            R_world2bcam[1][:] + (T_world2bcam[1],),
            R_world2bcam[2][:] + (T_world2bcam[2],),
        )
    )

    return cvRT if to_cv else glRT


def get_3x4_P_matrix_from_blender(cam, to_cv: bool):
    assert isinstance(cam, bpy.types.Object)
    K, intrinsics = get_calibration_matrix_K_from_blender(cam.data)
    RT = get_3x4_RT_matrix_from_blender(cam, to_cv)
    return K @ RT, K, intrinsics, RT


def get_blender_camera_resolution(scene):
    scale = scene.render.resolution_percentage / 100.0
    w = int(scene.render.resolution_x * scale)
    h = int(scene.render.resolution_y * scale)
    return w, h


# ===================== CORE IPC FUNCTIONS =====================


def _send_pose():
    if _state.client is None or not _state.client.connected:
        return False
    scene = bpy.context.scene
    cam = scene.camera
    if cam is None:
        return False

    cvRT = get_3x4_RT_matrix_from_blender(cam, to_cv=True)
    R = np.array([[cvRT[r][c] for c in range(3)] for r in range(3)])
    t = np.array([cvRT[r][3] for r in range(3)])

    bottom = np.array([[0.0, 0.0, 0.0, 1.0]])
    RT4 = np.concatenate([np.column_stack([R, t]), bottom], axis=0)
    c2w = np.linalg.inv(RT4)
    position = c2w[:3, 3]
    q_xyzw = SciRot.from_matrix(c2w[:3, :3]).as_quat()

    _state.client.send_camera_pose(
        position={k: str(v) for k, v in zip("xyz", position)},
        rotation={k: str(v) for k, v in zip("xyzw", q_xyzw)},
    )
    return True


def _receive_frame() -> Optional[np.ndarray]:
    if _state.client is None or not _state.client.connected:
        return None
    result = _state.client.receive()
    if not result or "color" not in result:
        return None
    tensor = result["color"]
    if tensor.is_cuda:
        arr = tensor.cpu().numpy()
    else:
        arr = tensor.numpy()
    return arr


def _update_display_texture(color_np: np.ndarray):
    if color_np is None or color_np.size == 0:
        return
    h, w = color_np.shape[:2]
    uint8 = (np.clip(color_np[..., :3], 0, 1) * 255).astype(np.uint8)
    rgba = np.empty((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = uint8
    rgba[..., 3] = 255
    _state.color_buffer_np = rgba
    _state.width, _state.height = w, h
    _state.gpu_texture = GPUTexture((w, h), format="RGBA8", data=rgba.flatten())


def _update_compositor_image(color_np: np.ndarray):
    if color_np is None or color_np.size == 0:
        return
    h, w = color_np.shape[:2]
    img = bpy.data.images.get("SplatbusOutput")
    if img is None:
        img = bpy.data.images.new("SplatbusOutput", width=w, height=h, alpha=True, float_buffer=True)
    if img.size[0] != w or img.size[1] != h:
        img.scale(w, h)
    flat = np.empty(w * h * 4, dtype=np.float32)
    flat[0::4] = color_np[..., 0].ravel()
    flat[1::4] = color_np[..., 1].ravel()
    flat[2::4] = color_np[..., 2].ravel()
    single_alpha = color_np[..., 3:4].ravel() if color_np.shape[2] == 4 else np.ones(w * h, dtype=np.float32)
    flat[3::4] = single_alpha
    img.pixels = flat.tolist()


# ===================== TICK =====================


def _tick():
    if not _state.is_running:
        return None
    if _state.client is None or not _state.client.connected:
        _stop()
        return None

    try:
        _send_pose()
        color = _receive_frame()
        if color is not None:
            _update_display_texture(color)
            _update_compositor_image(color)
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
    w, h = region.width, region.height

    gpu.state.blend_set("ALPHA")
    shader = gpu.shader.from_builtin("2D_IMAGE")
    shader.bind()
    shader.uniform_sampler("image", tex)
    batch = batch_for_shader(
        shader, "TRI_FAN",
        {
            "pos": ((0, 0), (w, 0), (w, h), (0, h)),
            "texCoord": ((0, 0), (1, 0), (1, 1), (0, 1)),
        },
    )
    batch.draw(shader)
    gpu.state.blend_set("NONE")


# ===================== LIFECYCLE =====================


def _start():
    if _state.is_running:
        return
    _state.is_running = True

    _state.timer = bpy.app.timers.register(_tick, first_interval=0.0, persistent=True)

    if _state.draw_handler is None:
        _state.draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            _draw_viewport, (), "WINDOW", "POST_PIXEL"
        )

    if not _state.compositor_ready:
        _setup_compositor()
        _state.compositor_ready = True

    for h in (bpy.app.handlers.frame_change_pre, bpy.app.handlers.render_pre):
        if _on_frame_change not in h:
            h.append(_on_frame_change)


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
            bpy.types.SpaceView3D.draw_handler_remove(_state.draw_handler, "WINDOW")
        except (ValueError, RuntimeError):
            pass
        _state.draw_handler = None

    for h in (bpy.app.handlers.frame_change_pre, bpy.app.handlers.render_pre):
        try:
            h.remove(_on_frame_change)
        except ValueError:
            pass

    _state.gpu_texture = None


def _on_frame_change(_scene=None, _depsgraph=None):
    if not _state.is_running:
        return
    try:
        _send_pose()
        color = _receive_frame()
        if color is not None:
            _update_compositor_image(color)
    except Exception:
        traceback.print_exc()


# ===================== COMPOSITOR =====================


def _setup_compositor():
    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree
    if tree is None:
        return

    if any(n.type == "IMAGE" and n.image and n.image.name == "SplatbusOutput" for n in tree.nodes):
        return

    w, h = _state.width or 1920, _state.height or 1080
    img = bpy.data.images.get("SplatbusOutput")
    if img is None:
        img = bpy.data.images.new("SplatbusOutput", width=w, height=h, alpha=True, float_buffer=True)

    rl = tree.nodes.new("CompositorNodeRLayers")
    rl.location = (0, 0)

    img_node = tree.nodes.new("CompositorNodeImage")
    img_node.location = (0, 200)
    img_node.image = img

    alpha_over = tree.nodes.new("CompositorNodeAlphaOver")
    alpha_over.location = (400, 0)

    composite = tree.nodes.new("CompositorNodeComposite")
    composite.location = (600, 0)

    tree.links.new(img_node.outputs["Image"], alpha_over.inputs[2])
    tree.links.new(rl.outputs["Image"], alpha_over.inputs[1])
    tree.links.new(alpha_over.outputs["Image"], composite.inputs["Image"])


# ===================== OPERATORS =====================


class SplatbusConnectOperator(bpy.types.Operator):
    bl_idname = "splatbus.connect"
    bl_label = "Connect to SplatBus"
    bl_description = "Connect to the SplatBus Gaussian Splatting server and start the render loop"

    def execute(self, context):
        if _state.is_running:
            self.report({"INFO"}, "Already connected")
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
        _state.width, _state.height = get_blender_camera_resolution(scene)

        t, quat = _state.client.get_camera_pose(cam_idx=0)
        print(f"[Splatbus] Server initial pose — t: {t}, quat: {quat}")

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
            _state.client.close()
            _state.client = None
        _state.color_buffer_np = None
        _state.gpu_texture = None
        self.report({"INFO"}, "Disconnected from SplatBus")
        return {"FINISHED"}


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
    SplatbusSetupCompositorOperator,
    SCENE_PT_splatbus,
)


@bpy.app.handlers.persistent
def _load_handler(_dummy):
    pass  # placeholder for future reload-state logic


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.splatbus_setup = PointerProperty(type=SplatbusProperties)
    bpy.app.handlers.load_post.append(_load_handler)


def unregister():
    _stop()
    if _state.client is not None:
        _state.client.close()
        _state.client = None

    bpy.app.handlers.load_post.remove(_load_handler)

    del bpy.types.Scene.splatbus_setup
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
