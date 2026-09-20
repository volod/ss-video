"""Manage Frigate camera configuration from the video package.

Usage:
  ./scripts/ssv/ssv-camera.sh --name front_door --rtsp rtsp://user:pass@192.168.1.100:554/stream1
  ./scripts/ssv/ssv-camera.sh --name usb_cam --usb /dev/video0
  ./scripts/ssv/ssv-camera.sh --name usb_cam --usb /dev/video0 --restart
  ./scripts/ssv/ssv-camera.sh --list
"""

import argparse
import os
import subprocess
import sys

from ss_kit.paths import data_dir, discover_project_root
from ss_kit.settings import load_layered_env

ROOT = discover_project_root()
load_layered_env(ROOT, app_env=os.getenv("APP_ENV", "prod"))
FRIGATE_CONFIG = data_dir(ROOT) / "coop" / "frigate" / "config.yml"


def load_config() -> dict:
    """Load Frigate config as a dict."""
    try:
        import yaml
    except ImportError:
        print("ERROR: PyYAML required. Run: pip install pyyaml", file=sys.stderr)
        sys.exit(1)

    if not FRIGATE_CONFIG.exists():
        print(f"ERROR: Config not found: {FRIGATE_CONFIG}", file=sys.stderr)
        sys.exit(1)

    with open(FRIGATE_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_config(config: dict) -> None:
    """Save Frigate config."""
    try:
        import yaml
    except ImportError:
        print("ERROR: PyYAML required. Run: pip install pyyaml", file=sys.stderr)
        sys.exit(1)

    with open(FRIGATE_CONFIG, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def list_cameras(config: dict) -> None:
    """Print configured cameras."""
    cameras = config.get("cameras", {})
    if not cameras:
        print("No cameras configured.")
        return
    for name, cam in cameras.items():
        enabled = cam.get("enabled", True)
        path = "?"
        if "ffmpeg" in cam and "inputs" in cam["ffmpeg"]:
            inputs = cam["ffmpeg"]["inputs"]
            if inputs:
                path = inputs[0].get("path", "?")
        print(f"  {name}: {path} (enabled={enabled})")


def add_rtsp_camera(config: dict, name: str, path: str, width: int, height: int, fps: int) -> None:
    """Add an RTSP camera config."""
    config.setdefault("cameras", {})
    config["cameras"][name] = {
        "enabled": True,
        "ffmpeg": {
            "inputs": [
                {
                    "path": path,
                    "roles": ["detect", "record"],
                }
            ]
        },
        "detect": {
            "width": width,
            "height": height,
            "fps": fps,
        },
    }


def add_usb_camera(
    config: dict,
    name: str,
    device: str,
    width: int,
    height: int,
    fps: int,
) -> None:
    """Add a USB (V4L2) camera config."""
    config.setdefault("cameras", {})
    config["cameras"][name] = {
        "enabled": True,
        "ffmpeg": {
            "inputs": [
                {
                    "path": device,
                    "input_args": "-f v4l2",
                    "roles": ["detect", "record"],
                }
            ]
        },
        "detect": {
            "width": width,
            "height": height,
            "fps": fps,
        },
    }


def restart_frigate() -> bool:
    """Restart Frigate container. Returns True on success."""
    compose_file = ROOT / "docker" / "core" / "docker-compose.yml"
    try:
        subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(compose_file),
                "--profile",
                "frigate",
                "restart",
                "frigate",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        return True
    except subprocess.CalledProcessError as exc:
        print(f"WARNING: Restart failed: {exc}", file=sys.stderr)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add camera to Frigate config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--list", action="store_true", help="List configured cameras")
    parser.add_argument("--name", type=str, help="Camera name (alphanumeric, underscores)")
    parser.add_argument("--rtsp", type=str, help="RTSP URL (e.g. rtsp://user:pass@ip:554/stream)")
    parser.add_argument("--usb", type=str, help="USB device path (e.g. /dev/video0)")
    parser.add_argument(
        "--width", type=int, default=640, help="Detect width (default: 640, use 1280 for RTSP)"
    )
    parser.add_argument(
        "--height", type=int, default=480, help="Detect height (default: 480, use 720 for RTSP)"
    )
    parser.add_argument("--fps", type=int, default=5, help="Detect FPS (default: 5)")
    parser.add_argument("--restart", action="store_true", help="Restart Frigate after adding")

    args = parser.parse_args()
    config = load_config()

    if args.list:
        list_cameras(config)
        return

    if not args.name:
        parser.error("--name required when adding camera")

    if args.rtsp:
        if args.name in config.get("cameras", {}):
            print(f"ERROR: Camera '{args.name}' already exists", file=sys.stderr)
            sys.exit(1)
        add_rtsp_camera(config, args.name, args.rtsp, args.width, args.height, args.fps)
        print(f"Added RTSP camera: {args.name}")
    elif args.usb:
        if args.name in config.get("cameras", {}):
            print(f"ERROR: Camera '{args.name}' already exists", file=sys.stderr)
            sys.exit(1)
        add_usb_camera(config, args.name, args.usb, args.width, args.height, args.fps)
        print(f"Added USB camera: {args.name} -> {args.usb}")
    else:
        parser.error("Specify --rtsp or --usb")

    save_config(config)
    print(f"Config saved: {FRIGATE_CONFIG}")

    if args.restart:
        print("Restarting Frigate...")
        if restart_frigate():
            print("OK: Frigate restarted")
        else:
            print(
                "Run manually: docker compose -f docker/core/docker-compose.yml --profile frigate restart frigate"
            )


if __name__ == "__main__":
    main()
