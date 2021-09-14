"""
Launching logic for bundle packaging
"""
import os.path
import sys
from pathlib import Path
from typing import Optional
import subprocess

try:
    from importlib import metadata as importlib_metadata
except ImportError:
    import importlib_metadata  # noqa


USERSPACE_LOCATIONS = {
    # TODO: check paths for linux / windows
    "linux": [Path(os.path.expanduser("~/.local/napari/"))],
    "darwin": [Path(os.path.expanduser("~/.local/napari/"))],
    "windows": [],
}


def running_as_bundled_app() -> bool:
    """Infer whether we are running as a briefcase bundle"""
    # https://github.com/beeware/briefcase/issues/412
    # https://github.com/beeware/briefcase/pull/425
    # note that a module may not have a __package__ attribute
    # From 0.4.12 we add a sentinel file next to the bundled sys.executable
    if (Path(sys.executable).parent / ".napari_is_bundled").exists():
        return True
    try:
        app_module = sys.modules['__main__'].__package__
    except AttributeError:
        return False
    try:
        metadata = importlib_metadata.metadata(app_module)
    except importlib_metadata.PackageNotFoundError:
        return False

    return 'Briefcase-Version' in metadata


def bundle_bin_dir() -> Optional[str]:
    """Return path to briefcase app_packages/bin if it exists."""
    bin = os.path.join(os.path.dirname(sys.exec_prefix), 'app_packages', 'bin')
    if os.path.isdir(bin):
        return bin


def install_to_userspace(target):
    extension = "exe" if sys.platform.startswith("win") else "sh"
    bundle = Path(sys.executable).parents[1] / "Resources" / f"bundle.{extension}"
    assert bundle.exists(), f"Cannot locate {bundle}!"

    prefix = first_writable_location()
    # TODO: this looks like a bug in constructor we need to fix
    # urls is a file actually?! but just creating something that exists is enough
    # ... ¯\_(ツ)_/¯
    (prefix / "pkgs" / "urls").mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["bash", str(bundle), "-bfp", str(prefix)])
    return prefix


def first_writable_location():
    for location in USERSPACE_LOCATIONS[sys.platform]:
        if not location.exists():
            try:
                location.mkdir(parents=True)
            except:
                continue

        try:
            (location / ".canary").touch()
        except:
            continue
        else:
            return location
        finally:
            (location / ".canary").unlink()

    raise ValueError("Could not find a suitable userspace location to install to.")


def ensure_installed():
    if not running_as_bundled_app():
        return

    locations = USERSPACE_LOCATIONS[sys.platform]

    for location in locations:
        if location.exists() and (location / "condabin" / "conda").exists():
            break
    else:
        # no installation detected!
        location = install_to_userspace(locations)

    # patch environment
    sys.path.insert(
        2,
        str(
            location
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
        ),
    )
