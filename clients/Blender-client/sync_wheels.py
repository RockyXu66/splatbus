"""Download splatbus and all its dependency wheels for every target platform.

Uses `uv lock` + `uv export` for resolution, then `pip download` (via uvx)
to fetch the actual wheels per platform. Pure-Python wheels are kept across
all platforms (downloaded once), while platform‑specific wheels are fetched
separately for each target.
"""

import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import packaging.markers
import packaging.requirements

HERE = Path(__file__).parent
SPLATBUS_SRC = HERE / ".." / ".." / "splatbus"
WHEELS = HERE / "wheels"
PYTHON_VERSION = "3.11"

# Target platform → pip platform tags + marker evaluation context
PLATFORMS = {
    "linux-x64": {
        "pip_platforms": [
            "manylinux_2_27_x86_64",
            "manylinux_2_28_x86_64",
            "manylinux2014_x86_64",
            "linux_x86_64",
        ],
        "markers": {
            "sys_platform": "linux",
            "platform_machine": "x86_64",
            "platform_system": "Linux",
            "python_full_version": PYTHON_VERSION,
            "os_name": "posix",
        },
    },
    "macos-arm64": {
        "pip_platforms": [
            "macosx_11_0_arm64",
            "macosx_12_0_arm64",
            "macosx_13_0_arm64",
            "macosx_14_0_arm64",
        ],
        "markers": {
            "sys_platform": "darwin",
            "platform_machine": "arm64",
            "platform_system": "Darwin",
            "python_full_version": PYTHON_VERSION,
            "os_name": "posix",
        },
    },
    "windows-x64": {
        "pip_platforms": [
            "win_amd64",
        ],
        "markers": {
            "sys_platform": "win32",
            "platform_machine": "AMD64",
            "platform_system": "Windows",
            "python_full_version": PYTHON_VERSION,
            "os_name": "nt",
        },
    },
}

LINK_RE = re.compile(r"^    # via")


def is_package_downloaded(pkg_spec: str) -> bool:
    try:
        req = packaging.requirements.Requirement(pkg_spec)
        name_normalized = req.name.replace("-", "_").lower()
        version = ""
        for spec in req.specifier:
            if spec.operator == "==":
                version = spec.value.replace("-", "_").lower()
                break
                
        for f in WHEELS.glob("*.whl"):
            f_name = f.name.lower().replace("-", "_")
            if f_name.startswith(name_normalized + "_") or f_name.startswith(name_normalized + "-"):
                if not version or version in f_name:
                    return True
    except Exception:
        pass
    return False


def _lock_is_current() -> bool:
    uv_lock = SPLATBUS_SRC / "uv.lock"
    pyproject = SPLATBUS_SRC / "pyproject.toml"
    if not uv_lock.exists():
        return False
    if pyproject.exists() and uv_lock.stat().st_mtime < pyproject.stat().st_mtime:
        return False
    return True


def _everything_downloaded() -> bool:
    existing = list(WHEELS.glob("*.whl"))
    if not existing:
        return False
    prefixes = {"loguru", "numpy", "splatbus"}
    for prefix in prefixes:
        if not any(w.name.lower().startswith(prefix) for w in existing):
            return False

    # Check splatbus wheel isn't stale vs source
    wheel = next(w for w in existing if w.name.lower().startswith("splatbus"))
    src_mtime = max(
        p.stat().st_mtime for p in SPLATBUS_SRC.rglob("*.py")
        if p.is_file() and ".venv" not in p.parts and "__pycache__" not in p.parts
    ) if SPLATBUS_SRC.exists() else 0
    if wheel.stat().st_mtime < src_mtime:
        return False

    return True


def main():
    WHEELS.mkdir(parents=True, exist_ok=True)

    # Fast path: skip everything if lock is current and all wheels exist
    if _lock_is_current() and _everything_downloaded():
        print("All wheels are already up-to-date, nothing to do.")
        return

    if not _lock_is_current():
        print("Locking splatbus dependencies …")
        subprocess.run(
            ["uv", "lock"],
            cwd=SPLATBUS_SRC,
            check=True,
            capture_output=True,
        )

    reqs_file = HERE / "splatbus-requirements.txt"
    print(f"Exporting requirements to {reqs_file} …")
    subprocess.run(
        ["uv", "export", "--frozen", "--no-hashes", f"--output-file={reqs_file}"],
        cwd=SPLATBUS_SRC,
        check=True,
        capture_output=True,
    )

    raw = reqs_file.read_text()
    entries = parse_requirements(raw)

    # Separate pure-Python (no marker or only python_version) from
    # platform-specific entries so we can download pure wheels once.
    pure_entries: list[str] = []
    plat_entries: dict[str, list[str]] = {k: [] for k in PLATFORMS}

    for pkg_spec, marker_str in entries:
        req = packaging.requirements.Requirement(pkg_spec)
        if not req.name:
            continue

        # Skip torch and all of its heavy rendering/transitive dependencies
        name_lower = req.name.lower()
        if (
            name_lower == "torch"
            or name_lower.startswith("nvidia-")
            or name_lower.startswith("intel-")
            or name_lower.startswith("triton")
            or name_lower in {
                "sympy", "networkx", "jinja2", "fsspec", "mpmath",
                "markupsafe", "filelock", "typing-extensions"
            }
        ):
            continue

        if not marker_str:
            pure_entries.append(pkg_spec)
            continue

        marker = packaging.markers.Marker(marker_str)

        # Download a pure-Python entry for every platform as long as it
        # matches at least one target (e.g. `python_full_version >= '3.11'`).
        for plat_name, plat_cfg in PLATFORMS.items():
            if marker.evaluate(plat_cfg["markers"]):
                plat_entries[plat_name].append(pkg_spec)


    # Download pure-Python wheels once (use the first platform's tag)
    missing_pure = [pkg for pkg in pure_entries if not is_package_downloaded(pkg)]
    first_plat = next(iter(PLATFORMS.values()))
    if missing_pure:
        print(f"Downloading {len(missing_pure)} pure-Python packages …")
        _pip_download(missing_pure, first_plat["pip_platforms"], label="pure")
    else:
        print("All pure-Python packages are already downloaded.")

    # Download platform-specific wheels per platform
    for plat_name, plat_cfg in PLATFORMS.items():
        pkgs = plat_entries[plat_name]
        missing_plat = [pkg for pkg in pkgs if not is_package_downloaded(pkg)]
        if not missing_plat:
            print(f"  All platform-specific packages for {plat_name} are already downloaded.")
            continue
        print(f"Downloading {len(missing_plat)} packages for {plat_name} …")
        _pip_download(missing_plat, plat_cfg["pip_platforms"], label=plat_name)

    # Build and copy splatbus wheel (skipped if _everything_downloaded already covered it)
    print("Building splatbus wheel …")
    subprocess.run(["uv", "build", "--directory", SPLATBUS_SRC], check=True)
    for whl in (SPLATBUS_SRC / "dist").glob("splatbus-*.whl"):
        shutil.copy2(str(whl), str(WHEELS))

    reqs_file.unlink(missing_ok=True)

    # Report what we got
    whls = sorted(WHEELS.glob("*.whl"))
    print(f"\nDone — {len(whls)} wheel(s) in {WHEELS}:")
    for w in whls:
        print(f"  {w.name}")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _pip_download(pkg_specs: list[str], platforms: list[str], label: str = ""):
    """Run ``pip download --no-deps`` for the given packages."""
    req_file = Path(f"/tmp/splatbus-deps-{label or hash(tuple(pkg_specs))}.txt")
    req_file.write_text("\n".join(pkg_specs) + "\n")

    cmd = [
        "uvx", "-p", PYTHON_VERSION, "--with", "pip",
        "pip", "download",
        "--no-deps",
        "--only-binary", ":all:",
        "--python-version", "311",
        "--implementation", "cp",
        "--abi", "cp311",
        "--retries", "3",
        "-d", str(WHEELS),
    ]
    for p in platforms:
        cmd.extend(["--platform", p])
    cmd.extend(["-r", str(req_file)])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    req_file.unlink(missing_ok=True)

    if result.returncode != 0:
        for line in result.stderr.splitlines():
            if any(kw in line for kw in ("ERROR", "WARNING", "not found", "Could not find")):
                print(f"    ⚠ {line.strip()}")


def parse_requirements(text: str):
    """Turn ``uv export`` output into ``(pkg_spec, marker_str)`` tuples."""
    entries: list[tuple[str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or LINK_RE.match(line):
            continue
        # Skip editable install references and the project itself
        if line.startswith("-e ") or line == ".":
            continue
        if ";" in line:
            spec, _, marker = line.partition(";")
            entries.append((spec.strip(), marker.strip()))
        else:
            entries.append((line.strip(), ""))
    return entries


if __name__ == "__main__":
    main()
