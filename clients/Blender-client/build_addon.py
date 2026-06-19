import os
import platform
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


def host_platform():
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "linux" and machine in ("x86_64", "amd64"):
        return "linux-x64"
    if system == "darwin" and machine == "arm64":
        return "macos-arm64"
    if system == "darwin" and machine in ("x86_64", "amd64"):
        return "macos-x64"
    if system == "windows" and machine in ("x86_64", "amd64"):
        return "windows-x64"
    return f"{system}-{machine}"


def wheel_platforms(filename: str):
    """Return the list of Blender platform names this wheel supports."""
    name = filename.lower()
    if "-any" in name or name.endswith("-none-any.whl") or "py3-none-any" in name:
        return ["linux-x64", "macos-arm64", "macos-x64", "windows-x64"]
    result = []
    if "manylinux" in name or "linux_x86_64" in name:
        result.append("linux-x64")
    if "macosx" in name:
        if "arm64" in name:
            result.append("macos-arm64")
        else:
            result.append("macos-x64")
    if "win_amd64" in name:
        result.append("windows-x64")
    return result


def parse_args():
    args = sys.argv[1:]
    blender = "blender"
    platform_override = None

    i = 0
    while i < len(args):
        if args[i] == "--platform" and i + 1 < len(args):
            platform_override = args[i + 1]
            i += 2
        elif not args[i].startswith("--"):
            blender = args[i]
            i += 1
        else:
            i += 1

    return blender, platform_override


def main():
    blender, platform_override = parse_args()

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

    # Determine platforms and filter wheels
    all_platforms = list(
        dict.fromkeys(
            p for w in os.listdir(WHEELS) if w.endswith(".whl")
            for p in wheel_platforms(w)
        )
    )

    if platform_override:
        target_platforms = [platform_override]
    else:
        target_platforms = all_platforms

    for w in os.listdir(WHEELS):
        if not w.endswith(".whl"):
            continue
        if not any(p in wheel_platforms(w) for p in target_platforms):
            continue
        shutil.copy2(os.path.join(WHEELS, w), os.path.join(BUILD, "wheels", w))

    # Build output manifest with auto-discovered wheels
    out = {k: v for k, v in manifest.items() if k not in ("build", "dependencies")}
    out["platforms"] = sorted(
        dict.fromkeys(
            p for w in os.listdir(BUILD + "/wheels") if w.endswith(".whl")
            for p in wheel_platforms(w)
        )
    )
    wheel_files = sorted(
        f"./wheels/{w}" for w in os.listdir(BUILD + "/wheels") if w.endswith(".whl")
    )
    out["wheels"] = wheel_files

    write_toml(out, os.path.join(BUILD, "blender_manifest.toml"))

    # Build extension
    addon_id = manifest.get("id", "splatbus")
    addon_version = manifest.get("version", "0.0.1")
    if platform_override:
        addon_version = f"{addon_version}+{platform_override}"
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

    whls_in_zip = len(wheel_files)
    print(f"\nBuilt {os.path.basename(output_path)} with {whls_in_zip} wheel(s)")
    print(f"Platforms: {', '.join(out['platforms'])}")


if __name__ == "__main__":
    main()
