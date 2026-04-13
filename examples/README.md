## Integration Examples

This folder contains example `splatbus_renderer.py` scripts for integrating
SplatBus into various Gaussian Splatting projects.

### Available Examples


| Example        | Script                                                                         | Original Project                                                               |
| -------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| 3DGS           | `[3dgs/splatbus_renderer.py](3dgs/splatbus_renderer.py)`                       | [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) |
| 4DGS           | `[4dgs/splatbus_renderer.py](4dgs/splatbus_renderer.py)`                       | [4D Gaussian Splatting](https://github.com/hustvl/4DGaussians)                 |
| GaussianAvatar | `[gaussian-avatar/splatbus_renderer.py](gaussian-avatar/splatbus_renderer.py)` | [mmlphuman](https://github.com/1231234zhan/mmlphuman)                          |


### Usage

1. Clone and install the original project following its setup instructions (datasets, training, etc.).

2. Install SplatBus in your Python environment:

```bash
conda activate YOUR_ENVIRONMENT
cd /path/to/splatbus/splatbus
pip install -e .
```

3. Copy the example script into the project root:

```bash
cp /path/to/splatbus/examples/3dgs/splatbus_renderer.py /path/to/gaussian-splatting/
```

4. Run the renderer (server):

<details>
<summary>3DGS</summary>

```bash
cd /path/to/gaussian-splatting
python splatbus_renderer.py -m <path_to_model> --skip_test --fps 30
```
</details>

<details>
<summary>4DGS</summary>

```bash
cd /path/to/4d-gaussian-splatting
python splatbus_renderer.py --config configs/YOUR_CONFIG.yaml --skip_train --fps 30
```
</details>

<details>
<summary>GaussianAvatar</summary>

```bash
cd /path/to/mmlphuman
python splatbus_renderer.py --config ./config/CONFIG_FILE.yaml --model_dir <MODEL_DIR> --out_dir <IMAGE_OUT_DIR> --cam_path <CAM_PATH> --pose_path <POSE_PATH> --fps 30 --test
```

**Note:** GaussianAvatar requires a small modification to [`scene/gaussian_model.py` (line 560)](https://github.com/1231234zhan/mmlphuman/blob/6668509/scene/gaussian_model.py#L560) in the mmlphuman project to enable depth rendering. Replace the `render` method with:

```python
def render(self, cam, override_color=None, scaling_modifier=1.0, background=None, with_depth=False):
    sh = self.get_sh      # can be faster
    covars = self.get_covariance(scaling_modifier)
    if override_color is None:
        cam_pos = torch.linalg.inv_ex(cam['w2c'])[0][:3,3]
        override_color = self.get_color(cam_pos)
    if with_depth:
        render_mode = "RGB+ED"
    else:
        render_mode = "RGB"
    image, alpha, info = rasterization(
        means=self.get_xyz,
        quats=None,
        scales=None,
        opacities=self.get_opacity,
        colors=override_color,
        viewmats=cam['w2c'][None],  # [1, 4, 4]
        Ks=cam['K'][None],  # [1, 3, 3]
        width=cam['width'],
        height=cam['height'],
        packed=False,
        near_plane=0.1,
        backgrounds=background[None],  # [1, 3]
        covars=covars,
        render_mode=render_mode,
    )
    return image[0], alpha[0], info
```
</details>



5. Run your viewer client (Unity/Blender/OpenGL) to view the rendering results.

