import os
import shutil
import subprocess
import sys
import tomllib

MANIFEST = "blender_manifest.toml"
SRC = "./src"
BUILD = "./build"
DIST = "./dist"
WHEELS = "./wheels"


def write_toml(data, path):
    """Write a dict as valid TOML (handles nested structures)."""
    def fmt_val(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, str):
            return f'"{v}"'
        if isinstance(v, list):
            items = []
            for i in v:
                items.append(fmt_val(i))
            if len(items) == 0:
                return "[]"
            if len(items) == 1 and len(str(items[0])) < 60:
                return f"[ {items[0]} ]"
            inner = ",\n  ".join(items)
            return f"[\n  {inner},\n]"
        return str(v)

    lines = []
    for k, v in data.items():
        if v is None:
            continue
        lines.append(f'{k} = {fmt_val(v)}')

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    with open(MANIFEST, "rb") as f:
        manifest = tomllib.load(f)

    if os.path.exists(BUILD):
        shutil.rmtree(BUILD)
    if os.path.exists(DIST):
        shutil.rmtree(DIST)

    os.makedirs(f"{BUILD}/wheels")
    os.makedirs(DIST)

    # Copy source
    for item in os.listdir(SRC):
        s = os.path.join(SRC, item)
        d = os.path.join(BUILD, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

    # Copy extra files
    for f in ["LICENSE", "README.md"]:
        if os.path.isfile(f):
            shutil.copy2(f, BUILD)

    # Copy wheel files
    for w in os.listdir(WHEELS):
        if w.endswith(".whl"):
            shutil.copy2(os.path.join(WHEELS, w), os.path.join(BUILD, "wheels", w))

    # Build output manifest with auto-discovered wheels
    # Drop 'dependencies' — those are Blender extension IDs, not pip packages.
    # Everything we need is bundled as wheels, so no extension dependencies exist.
    out = {k: v for k, v in manifest.items() if k not in ("build", "dependencies")}
    wheel_files = sorted(
        f"./wheels/{w}" for w in os.listdir(WHEELS) if w.endswith(".whl")
    )
    out["wheels"] = wheel_files

    write_toml(out, os.path.join(BUILD, "blender_manifest.toml"))

    # Build extension
    blender = sys.argv[1] if len(sys.argv) > 1 else "blender"
    addon_id = manifest.get("id", "splatbus")
    addon_version = manifest.get("version", "0.0.1")
    output_path = os.path.abspath(f"{DIST}/{addon_id}-{addon_version}.zip")
    subprocess.run(
        [
            blender,
            "--command", "extension", "build",
            "--source-dir", BUILD,
            "--output-filepath", output_path,
            "--verbose",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
