bl_info = {
    "name": "Splatbus",
    "author": "Théo Morales, Yinghan Xu",
    "version": (0, 4, 2),
    "blender": (4, 2, 0),
    "category": "Scene",
    "description": "Gaussian Splating unified rendering interface",
}

import math
import os
import pickle
import shutil
from pathlib import Path
from typing import List

import bpy
import numpy as np
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    IntVectorProperty,
    StringProperty,
)
from mathutils import Euler, Matrix, Vector

# Mock torch dynamically if it's not installed in Blender's Python environment,
# since the Blender client only uses camera/socket communication and doesn't need PyTorch.
import sys
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

# ---------------------PROPERTY DEFINITIONS


class SplatbusProperties(bpy.types.PropertyGroup):
    in_use: BoolProperty(
        name="Render SplatBus content",
        description="Whether to render content received from SplatBus.",
        default=True,
    )


# ---------------------INIT SCRIPT FUNCTIONS

# I copied the blender camera parameter extraction code (following 2 functions) from stack exchange:
# https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera


# BKE_camera_sensor_size
def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == "VERTICAL":
        return sensor_y
    return sensor_x


# BKE_camera_sensor_fit
def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == "AUTO":
        if size_x >= size_y:
            return "HORIZONTAL"
        else:
            return "VERTICAL"
    return sensor_fit


# Build intrinsic camera parameters from Blender camera data
# See notes on this in
# blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
# as well as
# https://blender.stackexchange.com/a/120063/3581
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

    # Parameters of intrinsic calibration matrix K
    u_0 = resolution_x_in_px / 2 - camd.shift_x * view_fac_in_px
    v_0 = resolution_y_in_px / 2 + camd.shift_y * view_fac_in_px / pixel_aspect_ratio
    skew = 0  # only use rectangular pixels

    K = Matrix(((s_u, skew, u_0), (0, s_v, v_0), (0, 0, 1)))
    return K, {
        "width": resolution_x_in_px * scale,
        "height": resolution_y_in_px * scale,
        "focal_len": f_in_mm,
    }  # For a simple pinhole model without distortions


# Returns camera rotation and translation matrices from Blender.
#
# There are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates
#         used in digital images)
#       - right-handed: positive z look-at direction
def get_3x4_RT_matrix_from_blender(cam, to_cv: bool):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        (
            (1, 0, 0),
            (0, -1, 0),
            (0, 0, -1),
        )
    )

    # Transpose since the rotation is object rotation,
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam @ cam.location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]  # Vector, Quaternion
    # Transpose to represent coordinate change instead of camera rotation (inverse)!
    R_world2bcam = rotation.to_matrix().transposed()  # Quaternion to matrix.

    # Convert camera location to translation vector used in coordinate changes
    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * R_world2bcam @ location

    # Build the coordinate transform matrix from world to computer vision camera
    # NOTE: Use * instead of @ here for older versions of Blender
    # TODO: detect Blender version
    R_world2cv = R_bcam2cv @ R_world2bcam
    T_world2cv = R_bcam2cv @ T_world2bcam

    # put into 3x4 matrix
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


def export_main_camera_animation(self, n_frames: int, output_path: Path):
    """
    Export the main camera animation poses for all frames.
    This exports intrinsics and per-frame extrinsics (world2cam and cam2world).

    Args:
        n_frames: Number of frames in the animation
        output_path: Path where to save the numpy files

    Returns:
        bool: True if export was successful, False if skipped
    """
    if n_frames <= 0:
        print(
            f"Warning: Invalid number of frames ({n_frames}). Skipping main camera export."
        )
        return False

    scene = bpy.context.scene
    main_cam = scene.camera
    if main_cam is None:
        self.report(
            {"WARNING"}, "No main camera found in scene. Skipping main camera export."
        )
        return False
    elif main_cam.name.startswith("Camera_"):
        self.report(
            {"WARNING"},
            "The main camera appears to be one of the scaffold cameras. Skipping main camera export.",
        )

    if main_cam.data.type != "PERSP":
        self.report(
            {"WARNING"}, "Not a perspective camera. Skipping main camera export."
        )
        return False

    fly_world2cams = np.zeros((n_frames, 3, 4), dtype=np.float32)
    fly_cam2worlds = np.zeros((n_frames, 3, 4), dtype=np.float32)

    # Get intrinsics (should be constant across frames for the main camera)
    # Note: This assumes intrinsics are not animated. If animated intrinsics are needed,
    # this should be moved inside the loop below.
    _, K, pinhole_params, _ = get_3x4_P_matrix_from_blender(main_cam, to_cv=False)
    fly_intrinsics = np.array(
        [
            pinhole_params["height"],
            pinhole_params["width"],
            pinhole_params["focal_len"],
        ]
    ).reshape((3, 1))
    fly_full_intrinsics = K

    original_frame = scene.frame_current
    for frame_idx in range(n_frames):
        scene.frame_set(frame_idx + 1)
        cvRT = get_3x4_RT_matrix_from_blender(main_cam, to_cv=True)
        # Build the full 4x4 transformation matrix
        bottom = np.array([0, 0, 0, 1.0]).reshape([1, 4])
        cvM = np.concatenate([cvRT, bottom], 0)
        cv_cam2world = np.linalg.inv(cvM)
        fly_world2cams[frame_idx] = cvRT  # world2cam (3, 4) matrix
        fly_cam2worlds[frame_idx] = cv_cam2world[:3, :4]  # cam2world (3, 4) matrix

    scene.frame_set(original_frame)
    np.save(Path.joinpath(output_path, "fly_cam_intrinsics.npy"), fly_intrinsics)
    np.save(
        Path.joinpath(output_path, "fly_cam_full_intrinsics.npy"), fly_full_intrinsics
    )
    np.save(Path.joinpath(output_path, "fly_world2cam.npy"), fly_world2cams)
    np.save(Path.joinpath(output_path, "fly_cam2world.npy"), fly_cam2worlds)
    print(f"Exported main camera animation: {n_frames} frames")
    print(f"  - fly_cam_intrinsics.npy: {fly_intrinsics.shape}")
    print(f"  - fly_world2cam.npy: {fly_world2cams.shape}")
    print(f"  - fly_cam2world.npy: {fly_cam2worlds.shape}")
    return True


# ---------------------MAIN FUNCTIONS


def get_blender_camera_resolution(scene):
    """Return (width, height) in pixels for the current render settings."""
    scale = scene.render.resolution_percentage / 100.0
    w = int(scene.render.resolution_x * scale)
    h = int(scene.render.resolution_y * scale)
    return w, h


def init_splatbus(self, context):
    print("Initializing splatbus...")
    self.report({"INFO"}, "Connecting to SplatBus...")
    self.client = GaussianSplattingIPCClient(
        host="127.0.0.1", ipc_port=6001, msg_port=6000
    )
    try:
        self.client.connect()
    except Exception as e:
        self.report(
            {"ERROR"}, f"Failed to connect to Gaussian Splatting IPC Server: {e}"
        )
        return

    scene = context.scene
    self.width, self.height = get_blender_camera_resolution(scene)
    print(f"Camera resolution: {self.width}x{self.height}")

    t, quat = self.client.get_camera_pose(cam_idx=0)
    print(f"Server initial pose — t: {t}, quat: {quat}")


def on_render(self, context):
    """Send the current Blender camera pose to the splatbus server."""
    scene = context.scene
    cam = scene.camera
    if cam is None:
        self.report({"WARNING"}, "No active camera in scene.")
        return

    # Get the CV-convention world-to-camera 3x4 matrix
    cvRT = get_3x4_RT_matrix_from_blender(cam, to_cv=True)

    # Extract rotation (3x3) and translation (3,)
    R = np.array([[cvRT[r][c] for c in range(3)] for r in range(3)])
    t = np.array([cvRT[r][3] for r in range(3)])

    # Build cam-to-world: invert the 4x4
    bottom = np.array([[0.0, 0.0, 0.0, 1.0]])
    RT4 = np.concatenate([np.column_stack([R, t]), bottom], axis=0)
    c2w = np.linalg.inv(RT4)

    # Camera centre in world space
    position = c2w[:3, 3]

    # Rotation as quaternion (scipy: [x, y, z, w])
    q_xyzw = SciRot.from_matrix(c2w[:3, :3]).as_quat()

    self.client.send_camera_pose(
        position={k: str(v) for k, v in zip("xyz", position)},
        rotation={k: str(v) for k, v in zip("xyzw", q_xyzw)},
    )


# ---------------------OPERATORS


class InitSplatbusOperator(bpy.types.Operator):
    bl_idname = "splatbus.init"
    bl_label = "Init SplatBus"

    def execute(self, context):
        init_splatbus(self, context)
        return {"FINISHED"}


class ApplySplatbusOperator(bpy.types.Operator):
    bl_idname = "splatbus.setup"
    bl_label = "Apply splatbus config"

    def execute(self, context):
        # setup_scaffold(self, context)
        return {"FINISHED"}


# ---------------------PANEL LAYOUT


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
        # Camera init properties
        layout.label(text="Settings", icon="OUTLINER_OB_CAMERA")

        box0 = layout.box()
        box0.operator("splatbus.init")

        # Individual camera settings
        box1 = layout.box()
        box1.prop(props, "in_use")
        # Apply button
        row = layout.row()
        row.scale_y = 1.75
        row.scale_x = 1.75
        box = row.box()
        box.operator("splatbus.setup")


# ---------------------REGISTRATION

classes = (
    SplatbusProperties,
    InitSplatbusOperator,
    ApplySplatbusOperator,
    SCENE_PT_splatbus,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.splatbus_setup = bpy.props.PointerProperty(type=SplatbusProperties)


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.splatbus_setup


if __name__ == "__main__":
    register()
