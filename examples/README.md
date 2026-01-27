## Integration Diffs

This folder contains `*.diff` files that show the examples to
integrate SplatBus into some Gaussian Splatting implementations.

### How to apply a diff

Follow the same steps for any project; replace placeholders with the project
you want.

```bash
# 1) Clone the original project at the commit listed in the diff header
git clone <ORIGINAL_GIT_URL>
cd <ORIGINAL_REPO_DIR>
git checkout <COMMIT_SHA>

# 2) Apply the patch from the ORIGINAL repo root
cp /path/to/splatbus/examples/<DIFF_FILE>.diff .
git apply --3way <DIFF_FILE>.diff
```

```bash
# Example for 3DGS
git clone git@github.com:graphdeco-inria/gaussian-splatting.git --recursive
cd gaussian-splatting
git checkout 54c035f

cp /path/to/splatbus/examples/3DGS.diff .
git apply --3way 3DGS.diff
```

### How to use the diff (after applying)

1. Install the Python environment according to the original codebase.

2. Follow the original project’s setup instructions (datasets, training, etc.) to train the GS model.

3. Install SplatBus in your Python environment:

```bash
conda activate YOUR_ENVIRONMENT
cd /path/to/splatbus/splatbus
pip install -e .
```

4. Run the renderer (server) script that was modified by the diff.

<details>
<summary>3DGS example command</summary>

```bash
cd /path/to/gaussian-splatting
python splatbus_renderer.py -m <path_to_model> --skip_test --fps 10
```
</details>
<details>
<summary>GaussianAvatar example command</summary>

```bash
cd /path/to/mmlphuman
python splatbus_renderer.py --config ./config/CONFIG_FILE.yaml --model_dir <MODEL_DIR> --out_dir <IMAGE_OUT_DIR> --cam_path <CAM_PATH> --pose_path <POSE_PATH> --fps 30 --test
```
</details>

4. Run your viewer client (Unity/Blender/OpenGL) to view the rendering results.

### Integration Examples

| Diff file | Original project | Commit SHA | Available |
| --- | --- | --- | --- |
| `3DGS.diff` | [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) | 54c035f | ✅ |
| `GaussianAvatar.diff` | [mmlphuman](https://github.com/1231234zhan/mmlphuman) | 6668509 | ✅ |
| `4DGS.diff` | [4D Gaussian Splatting](https://github.com/hustvl/4DGaussians) | 63725f2 | TODO |