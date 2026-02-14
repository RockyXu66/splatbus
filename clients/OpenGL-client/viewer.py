import enum
import math
import sys
from collections import defaultdict
from dataclasses import dataclass

import moderngl
import moderngl_window as mglw
import numpy as np
import torch
from imgui_bundle import hello_imgui, imgui, implot
from moderngl_window.integrations.imgui_bundle import ModernglWindowRenderer
from pyrr import Matrix44, Quaternion, Vector3
from rich.console import Console
from splatbus import GaussianSplattingIPCClient

console = Console()


def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2 + 1e-10))


class Controller(enum.Enum):
    KEYBOARD_MOUSE = enum.auto()
    ORBIT = enum.auto()


@dataclass
class HistData:
    # data: np.ndarray = np.zeros(1000, dtype=np.float32)
    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0
    hist_flags = implot.HistogramFlags_.density


class RadianceView(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "RadianceViewer"
    # window_size = (1352, 1014)
    window_size = (532,948)
    aspect_ratio = None
    resizable = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.client = GaussianSplattingIPCClient(
            host="127.0.0.1", ipc_port=6001, msg_port=6000
        )
        try:
            self.client.connect()
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Gaussian Splatting IPC Server: {e}")
        self.width, self.height = self.window_size
        self.time_range = (0, 10.0)  # self.user_context["time_range"]
        self.scene_bounds = 20  # self.user_context["scene_bounds"]

        self.controller_choice = Controller.ORBIT
        self.panning_dir = 1

        # ======= Controls
        # R, t = self.viewpoint_cam.R, self.viewpoint_cam.T
        R = np.eye(3, dtype=np.float32)
        t = np.zeros(3, dtype=np.float32)
        self.right = R[:, 0]
        self.up = R[:, 1]
        self.forward = R[:, 2]
        self.R_init = R.copy()
        self.canonical_pose = np.eye(4, dtype=np.float32)
        self.canonical_pose[:3, :3] = R
        self.canonical_pose[:3, 3] = t.squeeze()
        # Initialize position and orientation from given R, t
        self.position = Vector3(t.squeeze())
        # Extract yaw/pitch from rotation matrix
        self.yaw = 0.0
        self.pitch = 0.0
        # self.yaw = np.arctan2(R[1, 2], R[2, 2])
        # self.pitch = np.arcsin(-R[2, 2])
        self.slam_alignement = None
        self.slam_pose = np.eye(4, dtype=np.float32)
        self.slam_pose_init = None
        self.slam_trans_app = None
        self.R_slam = None
        self.slam_pose_prev = None
        self.paused = False
        # self.canonical_t = t
        self.speed = 0.1
        self.sensitivity = 0.005
        self.keys = set()
        self.time_speed = 1
        self.scaling_factor = 5
        self.release_action_keys = [
            self.wnd.keys.SPACE,
            self.wnd.keys.NUMBER_0,
        ]

        ######## GUI and plotting ############
        self.collection_stream = torch.cuda.Stream()
        self.plot_data = {}
        # self.reset_plot_data()
        imgui.create_context()
        implot.create_context()
        self.wnd.ctx.error
        self.imgui = ModernglWindowRenderer(self.wnd)
        ##################################
        # Fullscreen quad
        vertices = np.array(
            [
                -1.0,
                -1.0,
                0.0,
                0.0,
                1.0,
                -1.0,
                1.0,
                0.0,
                -1.0,
                1.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            dtype="f4",
        )
        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.vao = self.ctx.simple_vertex_array(
            self.create_shader(), self.vbo, "in_pos", "in_uv"
        )
        # Texture
        self.texture = self.ctx.texture((self.width, self.height), 3)
        self.texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        # PBO for async upload
        self.pbo = self.ctx.buffer(reserve=self.width * self.height * 3)
        self.pbo.orphan()  # hint that data will be frequently updated
        self._frames, self._export_n_frames = [], 500
        self._export_vid = False

    def reset_plot_data(self):
        self.plot_data = {
            "hist": {
                "custom_data": defaultdict(HistData),
            },
            "hist_config": {"n_bins": 1000},
            "active_guassians": self.gaussians.xyz.shape[0],
        }

    def create_shader(self):
        return self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_pos;
                in vec2 in_uv;
                out vec2 v_uv;
                void main() {
                    gl_Position = vec4(in_pos, 0.0, 1.0);
                    v_uv = vec2(in_uv.x, 1.0 - in_uv.y);  // flip vertically
                    //v_uv = in_uv;
                }
            """,
            fragment_shader="""
                #version 330
                uniform sampler2D tex;
                in vec2 v_uv;
                out vec4 f_color;
                void main() {
                    f_color = texture(tex, v_uv);
                }
            """,
        )

    # --- Mouse drag handler ---
    def on_mouse_position_event(self, x, y, dx, dy):
        self.imgui.mouse_position_event(x, y, dx, dy)

    def on_mouse_drag_event(self, x, y, dx, dy):
        if (
            self.controller_choice == Controller.KEYBOARD_MOUSE
            and not imgui.get_io().want_capture_mouse
        ):
            self.yaw += dx * self.sensitivity
            self.pitch += dy * self.sensitivity
        # self.pitch = np.clip(self.pitch, -np.pi / 2 + 1e-3, np.pi / 2 - 1e-3)
        self.imgui.mouse_drag_event(x, y, dx, dy)

    def on_mouse_scroll_event(self, x_offset, y_offset):
        self.imgui.mouse_scroll_event(x_offset, y_offset)

    def on_mouse_press_event(self, x, y, button):
        self.imgui.mouse_press_event(x, y, button)

    def on_mouse_release_event(self, x: int, y: int, button: int):
        self.imgui.mouse_release_event(x, y, button)

    # --- Key press/release handler ---
    def on_key_event(self, key, action, modifiers):
        if not imgui.get_io().want_capture_keyboard:
            if (
                key in self.release_action_keys
                and action == self.wnd.keys.ACTION_RELEASE
            ):
                if key in self.keys:
                    self.keys.discard(key)
                else:
                    self.keys.add(key)
            elif (
                key not in self.release_action_keys
                and action == self.wnd.keys.ACTION_PRESS
            ):
                self.keys.add(key)
            elif (
                key not in self.release_action_keys
                and action == self.wnd.keys.ACTION_RELEASE
            ):
                self.keys.discard(key)
        self.imgui.key_event(key, action, modifiers)

    # --- Update translation ---
    def update_controller(self):
        if self.wnd.keys.SPACE in self.keys:
            self.paused = not self.paused
            self.keys.discard(self.wnd.keys.SPACE)
        if self.wnd.keys.NUMBER_0 in self.keys:
            self.controller_choice = (
                Controller.ORBIT
                if self.controller_choice != Controller.ORBIT
                else Controller.KEYBOARD_MOUSE
            )
            self.keys.discard(self.wnd.keys.NUMBER_0)
            print("Switched to ", self.controller_choice)
            self.position = Vector3(self.canonical_pose[:3, 3].squeeze())
            # Extract yaw/pitch from rotation matrix
            self.yaw = 0.0
            self.pitch = 0.0
        if self.controller_choice == Controller.KEYBOARD_MOUSE:
            # # Local axes from yaw/pitch
            forward = -np.array(
                [
                    np.cos(self.pitch) * np.sin(self.yaw),
                    np.sin(self.pitch),
                    np.cos(self.pitch) * np.cos(self.yaw),
                ]
            )
            right = np.array(
                [np.sin(self.yaw - np.pi / 2), 0.0, np.cos(self.yaw - np.pi / 2)]
            )
            up = np.cross(right, forward)

            if self.wnd.keys.W in self.keys:
                self.position += forward * self.speed
            if self.wnd.keys.S in self.keys:
                self.position -= forward * self.speed
            if self.wnd.keys.A in self.keys:
                self.position -= right * self.speed
            if self.wnd.keys.D in self.keys:
                self.position += right * self.speed
            if self.wnd.keys.Q in self.keys:
                self.position += up * self.speed
            if self.wnd.keys.E in self.keys:
                self.position -= up * self.speed
        elif self.controller_choice == Controller.ORBIT:
            forward = -np.array(
                [
                    np.cos(self.pitch) * np.sin(self.yaw),
                    np.sin(self.pitch),
                    np.cos(self.pitch) * np.cos(self.yaw),
                ]
            )
            right = np.array(
                [np.sin(self.yaw - np.pi / 2), 0.0, np.cos(self.yaw - np.pi / 2)]
            )
            up = np.cross(right, forward)
            strife = torch.norm(
                (torch.from_numpy(self.position.xyz) - self.canonical_pose[:3, 3])
            ).item()
            if strife >= self.scene_bounds:
                self.panning_dir *= -1
            self.position += self.panning_dir * right * self.speed * 0.3
            self.position += (
                self.panning_dir
                * up
                * self.speed
                * np.sin(strife / self.scene_bounds * 2 * np.pi)
                * 0.3
            )
            self.yaw -= self.panning_dir * self.sensitivity

    # --- Get R and t ---
    def get_rt(self):
        if self.controller_choice in [Controller.KEYBOARD_MOUSE, Controller.ORBIT]:
            # R = Matrix44.from_eulers([self.pitch, self.yaw, 0.0])[:3, :3]
            yaw_delta = self.yaw
            pitch_delta = self.pitch

            # R_yaw = Matrix44.from_axis_rotation(Vector3(self.up), yaw_delta)
            # R_pitch = Matrix44.from_axis_rotation(Vector3(self.right), pitch_delta)
            # R_keyboard = R_yaw @ R_pitch
            q_yaw = Quaternion.from_axis_rotation(Vector3(self.up), yaw_delta)
            q_pitch = Quaternion.from_axis_rotation(-Vector3(self.right), pitch_delta)

            # combine: yaw first, then pitch
            q_total = (
                q_yaw * q_pitch
            )  # quaternion multiplication applies pitch after yaw

            # convert to rotation matrix
            R_keyboard = Matrix44.from_quaternion(q_total)[:3, :3]
            R = R_keyboard @ self.R_init
            t = self.position
            # print(t, self.canonical_pose[:3, 3])
            trans = np.array([0.0, 0.0, 0.0])
        elif self.controller_choice == Controller.SLAM:
            R = self.canonical_pose[:3, :3] @ self.slam_pose[:3, :3].T
            t = (self.scaling_factor * self.slam_pose[:3, 3]) + self.canonical_pose[
                :3, 3
            ]
            trans = np.array([0.0, 0.0, 0.0])

        else:
            raise ValueError("Unknown controller choice")
        return R, t, trans

    def _move_collection_to_cpu(self):
        def recurse(d):
            if isinstance(d, dict):
                return {k: recurse(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [recurse(v) for v in d]
            elif isinstance(d, torch.Tensor):
                return d.detach().to("cpu", non_blocking=True)
            elif isinstance(d, HistData) and isinstance(d.data, torch.Tensor):
                d.data = d.data.detach().to("cpu", non_blocking=True)
                return d
            else:
                return d

        self.plot_data = recurse(self.plot_data)

    def _convert_collection_to_numpy(self):
        def recurse(d):
            if isinstance(d, dict):
                return {k: recurse(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [recurse(v) for v in d]
            elif isinstance(d, torch.Tensor):
                return d.numpy()
            elif isinstance(d, HistData) and isinstance(d.data, torch.Tensor):
                d.data = d.data.numpy()
                return d
            else:
                return d

        self.plot_data = recurse(self.plot_data)

    def _update_timestamp(self, frame_time: float):
        return
        if not self.paused:
            self.viewpoint_cam.timestamp += frame_time * self.time_speed
            if self.viewpoint_cam.timestamp > self.time_range[1]:
                self.viewpoint_cam.timestamp = self.time_range[0]

    def on_render(self, time: float, frame_time: float):
        self.update_controller()
        self._update_timestamp(frame_time)
        # with torch.cuda.stream(self.collection_stream):
        #     self._collect_data_cuda()
        #     self._move_collection_to_cpu()
        with torch.no_grad():
            R, t, trans = self.get_rt()
            # self.viewpoint_cam.set_rt(R, t)
            self.client.send_camera_pose(
                position={"x": 0.1 * t, "y": 0.0, "z": 1.5},
                rotation={"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            )
            # image = viewer_rendering.render_frame(
            #     self.user_context, self.viewpoint_cam.timestamp, self.viewpoint_cam
            # )
            image = self.client.receive()
            if "color" not in image:
                image = {
                    "color": torch.zeros(3, self.height, self.width, dtype=torch.uint8)
                }

        # Map PBO and write data
        self.pbo.write(
            image["color"].cpu().numpy().tobytes()
        )  # TODO: windowtensor thing
        # Bind PBO to texture
        self.texture.write(self.pbo)
        # Render quad
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.texture.use()
        self.vao.render(moderngl.TRIANGLE_STRIP)
        # self.collection_stream.synchronize()
        # self._convert_collection_to_numpy()
        # self.render_ui()

    def _collect_data_cuda(self):
        raise NotImplementedError("This should be made a SplatBus call.")

    def _render_motion_model_ui(self):
        raise NotImplementedError("This should be made more generic.")
        if implot.begin_plot(
            "Average motion model temporal function",
            size=hello_imgui.em_to_vec2(20, 20),
        ):
            implot.setup_axes(
                "t",
                "phi(t)",
                implot.AxisFlags_.auto_fit,
                implot.AxisFlags_.auto_fit,
            )
            for dof in self.plot_data["temporal_func"].keys():
                implot.plot_line(
                    f"phi_{dof}(t)",
                    self.plot_data["temporal_func"][dof]["x"],
                    self.plot_data["temporal_func"][dof]["y"],
                )
            implot.end_plot()
        # imgui.same_line()
        # if implot.begin_plot()

    def _render_statistics_ui(self):
        if imgui.begin_tab_bar("Histograms"):
            for category_name, histograms in self.plot_data["hist"].items():
                cat_title = category_name.replace("_", " ").title()
                active, _ = imgui.begin_tab_item(cat_title)
                if active:
                    if implot.begin_plot(
                        cat_title,
                        size=hello_imgui.em_to_vec2(40, 20),
                    ):
                        implot.setup_axes(
                            "",
                            "",
                            implot.AxisFlags_.auto_fit,
                            implot.AxisFlags_.auto_fit,
                        )
                        implot.set_next_fill_style(implot.AUTO_COL, 0.5)
                        for hist_name, hist_data in histograms.items():
                            hist_title = hist_name.replace("_", " ").title()
                            implot.plot_histogram(
                                hist_title,
                                hist_data.data,
                                bins=self.plot_data["hist_config"]["n_bins"],
                                bar_scale=1.0,
                                range=implot.Range(),
                                flags=hist_data.hist_flags,
                            )
                        implot.end_plot()
                        imgui.same_line()
                        if imgui.begin_table("Stats", 4):
                            for hist_name, hist_data in histograms.items():
                                hist_title = hist_name.replace("_", " ").title()
                                imgui.table_next_row()
                                imgui.table_set_column_index(0)
                                imgui.text(hist_title)
                                imgui.table_set_column_index(1)
                                imgui.text(f"Mean: {hist_data.mean:.4f}")
                                imgui.table_set_column_index(2)
                                imgui.text(f"Min: {hist_data.min:.4f}")
                                imgui.table_set_column_index(3)
                                imgui.text(f"Max: {hist_data.max:.4f}")
                            imgui.end_table()
                    imgui.end_tab_item()
            imgui.end_tab_bar()

    def render_ui(self):
        def millify(n):
            millnames = ["", " K", " M", " B", " T"]
            n = float(n)
            millidx = max(
                0,
                min(
                    len(millnames) - 1,
                    int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3)),
                ),
            )

            return "{:.2f}{}".format(n / 10 ** (3 * millidx), millnames[millidx])

        imgui.new_frame()
        if imgui.begin("Inspection", True):
            # imgui.text(f"Timestamp: {self.viewpoint_cam.timestamp:.2f}s")
            imgui.text(
                f"Active Gaussians: {millify(self.plot_data['active_guassians'])}/{millify(self.gaussians.xyz.shape[0])}"
            )
            imgui.text(f"Average duration: {self.plot_data['avg_duration']:.2f}s")
            _, self.paused = imgui.checkbox("Pause", self.paused)
            imgui.same_line()
            _, self.time_speed = imgui.slider_float(
                "Time speed", self.time_speed, 0.01, 10.0
            )
            imgui.same_line()
            if imgui.button("Reset"):
                self.time_speed = 1

            if imgui.collapsing_header("Motion model"):
                self._render_motion_model_ui()
            if imgui.collapsing_header("Model statistics"):
                self._render_statistics_ui()
        imgui.end()
        imgui.render()
        self.imgui.render(imgui.get_draw_data())

    def on_resize(self, width: int, height: int):
        self.imgui.resize(width, height)


if __name__ == "__main__":
    mglw.setup_basic_logging(20)  # INFO level
    # window_cls = mglw.get_local_window_cls("glfw")  # or 'glfw', 'sdl2'
    # window = window_cls(
    #     title="Radiance View",
    #     size=(1280, 720),
    #     fullscreen=False,
    #     resizable=True,
    #     gl_version=(3, 3),
    #     vsync=True,
    # )
    # window.config = RadianceView(sys.argv[1:], ctx=window.ctx, wnd=window)
    # window.run()
    mglw.run_window_config(RadianceView)
