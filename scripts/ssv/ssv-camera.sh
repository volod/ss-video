#!/usr/bin/env bash
# Manage Frigate camera entries from the project shell script surface.
#
# Usage:
#   ./scripts/ssv/ssv-camera.sh --name front_door --rtsp rtsp://user:pass@192.168.1.100:554/stream1
#   ./scripts/ssv/ssv-camera.sh --name usb_cam --usb /dev/video0 --restart
#   ./scripts/ssv/ssv-camera.sh --list

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../shared/common.sh"
project_run_python_module selfsuvis.camera_cli "$@"
