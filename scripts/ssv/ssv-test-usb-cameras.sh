#!/usr/bin/env bash
# Test USB cameras attached to the computer.
# Lists V4L2 devices and optionally tests capture with ffmpeg.
# Requires: v4l-utils (v4l2-ctl), optionally ffmpeg for capture test.
#
# Usage:
#   ./scripts/ss-sens/ss-sens-test-usb-cameras.sh           # List devices only
#   ./scripts/ss-sens/ss-sens-test-usb-cameras.sh --test    # List + test capture

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../shared/common.sh"
TEST_CAPTURE=false

if [[ "${1:-}" == "--test" ]]; then
  TEST_CAPTURE=true
fi

echo "=== USB Camera Detection ==="
echo ""

# Check for v4l2-ctl (preferred) or fall back to sysfs
HAS_V4L2_CTL=false
if command -v v4l2-ctl &>/dev/null; then
  HAS_V4L2_CTL=true
fi

if [[ "$HAS_V4L2_CTL" == true ]]; then
  # List V4L2 devices
  echo "V4L2 devices (v4l2-ctl --list-devices):"
  echo "----------------------------------------"
  if ! v4l2-ctl --list-devices 2>/dev/null; then
    echo "No V4L2 devices found or permission denied."
    echo "Try: sudo v4l2-ctl --list-devices"
    echo "Or add your user to the 'video' group: sudo usermod -aG video $USER"
  fi
else
  echo "v4l2-ctl not found. Install v4l-utils for full device info:"
  echo "  Ubuntu/Debian: sudo apt-get install v4l-utils"
  echo "  Fedora:        sudo dnf install v4l-utils"
  echo "  Arch:          sudo pacman -S v4l-utils"
  echo ""
  echo "Checking /dev/video* and /sys/class/video4linux..."
  echo "----------------------------------------"
  if [[ -d /sys/class/video4linux ]]; then
    for d in /sys/class/video4linux/video*; do
      if [[ -d "$d" ]]; then
        name=$(cat "$d/name" 2>/dev/null || echo "unknown")
        dev="/dev/$(basename "$d")"
        echo "  $dev: $name"
      fi
    done
  fi
  if [[ ! -d /sys/class/video4linux ]] || [[ -z "$(ls -A /sys/class/video4linux 2>/dev/null)" ]]; then
    echo "No video devices found."
  fi
fi

echo ""
echo "Device nodes in /dev/v4l/:"
echo "----------------------------------------"
if [[ -d /dev/v4l ]]; then
  ls -la /dev/v4l/ 2>/dev/null || true
  echo ""
  if [[ -d /dev/v4l/by-id ]]; then
    echo "Stable device paths (by-id) for Docker passthrough:"
    ls -la /dev/v4l/by-id/ 2>/dev/null || true
  fi
else
  echo "/dev/v4l not found"
fi

echo ""
echo "Video4Linux devices:"
echo "----------------------------------------"
for dev in /dev/video*; do
  if [[ -e "$dev" ]]; then
    name="unknown"
    if [[ "$HAS_V4L2_CTL" == true ]] && v4l2-ctl -d "$dev" --info &>/dev/null; then
      name=$(v4l2-ctl -d "$dev" --info 2>/dev/null | grep "Card type" | cut -d: -f2 | xargs || echo "unknown")
    elif [[ -f "/sys/class/video4linux/$(basename $dev)/name" ]]; then
      name=$(cat "/sys/class/video4linux/$(basename $dev)/name" 2>/dev/null || echo "unknown")
    fi
    echo "  $dev: $name"
  fi
done

# Optional: test capture with ffmpeg
if [[ "$TEST_CAPTURE" == true ]]; then
  echo ""
  echo "=== Capture Test ==="
  if ! command -v ffmpeg &>/dev/null; then
    echo "ffmpeg not found. Install ffmpeg to run capture test."
  else
    FIRST_VIDEO=""
    for dev in /dev/video0 /dev/video1 /dev/video2; do
      if [[ -e "$dev" ]]; then
        FIRST_VIDEO="$dev"
        break
      fi
    done

    if [[ -z "$FIRST_VIDEO" ]]; then
      echo "No /dev/video* device found for capture test."
    else
      echo "Testing capture from $FIRST_VIDEO (2 seconds)..."
      if ffmpeg -y -f v4l2 -i "$FIRST_VIDEO" -t 2 -frames:v 5 -f null - 2>&1 | tail -20; then
        echo ""
        echo "OK: Capture test passed for $FIRST_VIDEO"
      else
        echo ""
        echo "WARNING: Capture test failed. Camera may be in use or incompatible."
      fi
    fi
  fi
fi

echo ""
echo "Done. For Frigate USB camera setup, see docs/ss-sens/sensor-integration.md"
