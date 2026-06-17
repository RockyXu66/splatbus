bl_info = {
    "name": "Splatbus",
    "author": "Théo Morales, Yinghan Xu",
    "version": (0, 0, 1),
    "blender": (4, 2, 0),
    "category": "Scene",
    "description": "Gaussian Splating unified rendering interface",
}

import os
import socket
import subprocess
import sys
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
from mathutils import Matrix, Vector

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


def _update_compositor_image(color_np: np.ndarray):
    if color_np is None or color_np.size == 0:
        return
    h, w = color_np.shape[:2]
    _state.width, _state.height = w, h
    img = bpy.data.images.get("SplatbusOutput")
    if img is None or img.size[0] != w or img.size[1] != h:
        if img is not None:
            bpy.data.images.remove(img)
        img = bpy.data.images.new("SplatbusOutput", width=w, height=h, alpha=True, float_buffer=True)
    flat = np.empty(w * h * 4, dtype=np.float32)
    flat[0::4] = color_np[..., 0].ravel()
    flat[1::4] = color_np[..., 1].ravel()
    flat[2::4] = color_np[..., 2].ravel()
    flat[3::4] = 1.0
    img.pixels.foreach_set(flat)
    img.update()
    img.gl_touch()

    tex = gpu.texture.from_image(img)
    _state.gpu_texture = tex


# ===================== TICK =====================


def _tick():
    if not _state.is_running:
        return None
    if _state.client is None or not _state.client.connected:
        _stop()
        return None

    try:
        c2w = _get_viewport_camera_matrix()
        _send_pose(c2w)
        color = _receive_frame()
        if color is not None:
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

    for h in (bpy.app.handlers.frame_change_pre, bpy.app.handlers.render_pre):
        if _on_frame_change not in h:
            h.append(_on_frame_change)

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

    for h in (bpy.app.handlers.frame_change_pre, bpy.app.handlers.render_pre):
        try:
            h.remove(_on_frame_change)
        except ValueError:
            pass


def _on_frame_change(_scene=None, _depsgraph=None):
    if not _state.is_running:
        return
    try:
        c2w = _get_scene_camera_matrix()
        _send_pose(c2w)
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


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.splatbus_setup = PointerProperty(type=SplatbusProperties)
    bpy.app.handlers.load_post.append(_load_handler)


def unregister():
    _stop()
    if _state.client is not None:
        _safe_close_client(_state.client)
        _state.client = None

    bpy.app.handlers.load_post.remove(_load_handler)

    del bpy.types.Scene.splatbus_setup
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
