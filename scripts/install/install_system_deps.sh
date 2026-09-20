#!/usr/bin/env bash
# Install system libraries and tools required to run the video semantic search
# solution locally on Linux (ffmpeg, OpenCV runtime deps, optional Python).
# Usage: sudo ./scripts/install_system_deps.sh [--with-python]
set -euo pipefail

WITH_PYTHON=false
for arg in "$@"; do
  case "$arg" in
    --with-python) WITH_PYTHON=true ;;
    -h|--help)
      echo "Usage: $0 [--with-python]"
      echo ""
      echo "Installs system packages needed to run the project locally (no Docker):"
      echo "  - ffmpeg (video decoding)"
      echo "  - OpenCV runtime libs (libgl1, libglib2.0-0, libsm, libxext, libxrender)"
      echo "  - PortAudio (libportaudio2) — required by sounddevice for audio playback/capture"
      echo ""
      echo "Options:"
      echo "  --with-python  Also install Python 3 and python3-venv / python3-pip."
      echo ""
      echo "After running this script, create a venv and install Python deps:"
      echo "  make venv"
      exit 0
      ;;
  esac
done

if [[ "$EUID" -ne 0 ]]; then
  echo "Error: this script must be run as root." >&2
  echo "  sudo $0 $*" >&2
  exit 1
fi

echo "=============================================="
echo "  System dependencies for Video Semantic Search"
echo "=============================================="
echo "This script installs ffmpeg and OpenCV-related libraries so you can run"
echo "the app locally (without Docker). Run with sudo."
echo ""

# Detect distro
if [[ -f /etc/os-release ]]; then
  # shellcheck source=/dev/null
  source /etc/os-release
  echo "Detected OS: $ID ${VERSION_ID:-}"
else
  echo "Cannot detect OS: /etc/os-release not found." >&2
  exit 1
fi
echo ""

# Map nvidia-smi "CUDA Version" (driver max) → nvcc toolkit version needed by torch.
# Same mapping as install_requirements.sh :: map_cuda_to_torch_index, but returns
# "MAJOR.MINOR" instead of "cuXXX" (apt package names use dashes: cuda-nvcc-12-6).
_map_driver_cuda_to_nvcc_ver() {
  local driver_cuda="$1"
  case "$driver_cuda" in
    13.*|12.9*|12.8*|12.7*|12.6*|12.5*) echo "12.6" ;;
    12.4*)                                 echo "12.4" ;;
    12.3*|12.2*|12.1*|12.0*)              echo "12.1" ;;
    11.8*|11.7*|11.6*)                     echo "11.8" ;;
    *)                                     echo ""     ;;
  esac
}

# Add the official NVIDIA CUDA apt repository (downloads cuda-keyring, runs apt-get update).
# Must be run as root.  Non-fatal.
_add_nvidia_cuda_apt_repo() {
  dpkg -s cuda-keyring >/dev/null 2>&1 && {
    echo "  NVIDIA CUDA apt repo already present."
    return 0
  }
  local os_id os_ver
  # shellcheck source=/dev/null
  . /etc/os-release
  os_id="${ID:-ubuntu}"
  os_ver="${VERSION_ID:-22.04}"
  local distro="${os_id}${os_ver//./}"
  local arch
  arch=$(dpkg --print-architecture 2>/dev/null || uname -m)
  local nvidia_arch
  case "$arch" in
    amd64|x86_64)  nvidia_arch="x86_64" ;;
    arm64|aarch64) nvidia_arch="sbsa"   ;;
    *)             nvidia_arch="x86_64" ;;
  esac
  local url="https://developer.download.nvidia.com/compute/cuda/repos/${distro}/${nvidia_arch}/cuda-keyring_1.1-1_all.deb"
  local deb="/tmp/cuda-keyring_1.1-1_all.deb"
  echo "  Downloading NVIDIA CUDA keyring from ${url} ..."
  if command -v wget >/dev/null 2>&1; then
    wget -q -O "$deb" "$url" || { echo "  wget failed."; return 1; }
  elif command -v curl >/dev/null 2>&1; then
    curl -fsSL -o "$deb" "$url" || { echo "  curl failed."; return 1; }
  else
    echo "  wget/curl not found — cannot download keyring."; return 1
  fi
  dpkg -i "$deb" && rm -f "$deb" || { rm -f "$deb"; echo "  dpkg -i failed."; return 1; }
  apt-get update -qq 2>/dev/null || true
  echo "  NVIDIA CUDA apt repo added."
}

# Detect GPU, determine required nvcc version, install it, and register with
# update-alternatives so it becomes the system default.  Non-fatal.
_maybe_install_cuda_nvcc_debian() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi
  local driver_cuda
  driver_cuda=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)
  [[ -z "$driver_cuda" ]] && return 0

  local nvcc_ver
  nvcc_ver=$(_map_driver_cuda_to_nvcc_ver "$driver_cuda")
  if [[ -z "$nvcc_ver" ]]; then
    echo "  GPU detected (driver CUDA ${driver_cuda}) but no known nvcc mapping — skipping."
    return 0
  fi
  local major="${nvcc_ver%%.*}" minor="${nvcc_ver##*.}"
  local pkg="cuda-nvcc-${major}-${minor}"

  # Already installed?
  local current_nvcc
  current_nvcc=$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1)
  if [[ "$current_nvcc" == "$nvcc_ver" ]] || [[ -x "/usr/local/cuda-${nvcc_ver}/bin/nvcc" ]]; then
    echo "  nvcc ${nvcc_ver} already installed — OK."
    # Ensure it is the default alternative.
    local nvcc_path="/usr/local/cuda-${nvcc_ver}/bin/nvcc"
    [[ -x "$nvcc_path" ]] || nvcc_path=$(command -v nvcc)
    local priority=$(( major * 10 + minor ))
    update-alternatives --install /usr/bin/nvcc nvcc "$nvcc_path" "$priority" 2>/dev/null || true
    update-alternatives --set nvcc "$nvcc_path" 2>/dev/null || true
    return 0
  fi

  echo "  GPU detected (driver CUDA ${driver_cuda}) — torch requires nvcc ${nvcc_ver}."
  echo "  Installing ${pkg} ..."
  if ! apt-get install -y --no-install-recommends "${pkg}" 2>/dev/null; then
    echo "  ${pkg} not in apt sources — adding NVIDIA CUDA apt repository ..."
    if _add_nvidia_cuda_apt_repo; then
      apt-get install -y --no-install-recommends "${pkg}" 2>/dev/null || {
        echo "  WARNING: ${pkg} still unavailable after adding NVIDIA apt repo."
        echo "  Manual fix: https://developer.nvidia.com/cuda-downloads"
        return 0
      }
    else
      echo "  WARNING: could not add NVIDIA apt repo. Manual fix:"
      echo "    https://developer.nvidia.com/cuda-downloads"
      echo "    sudo apt-get install -y ${pkg} && make venv"
      return 0
    fi
  fi

  echo "  ${pkg} installed."
  # Register nvcc with update-alternatives so it becomes the system default.
  local nvcc_path="/usr/local/cuda-${nvcc_ver}/bin/nvcc"
  [[ -x "$nvcc_path" ]] || nvcc_path=$(command -v nvcc 2>/dev/null || echo "")
  if [[ -n "$nvcc_path" ]]; then
    local priority=$(( major * 10 + minor ))
    update-alternatives --install /usr/bin/nvcc nvcc "$nvcc_path" "$priority" 2>/dev/null || true
    update-alternatives --set nvcc "$nvcc_path" 2>/dev/null && \
      echo "  update-alternatives: nvcc default → ${nvcc_path}" || true
  fi
}

install_debian() {
  echo "Installing packages with apt (Debian/Ubuntu)..."
  apt-get update
  echo "  Installing ffmpeg, OpenCV runtime libraries, PortAudio, and build tools..."
  apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libportaudio2 \
    ninja-build
  if [[ "$WITH_PYTHON" == true ]]; then
    echo "  Installing Python 3, venv, and pip..."
    apt-get install -y --no-install-recommends \
      python3 \
      python3-venv \
      python3-pip
  fi
  echo ""
  echo "  Detecting GPU and installing matching CUDA compiler (nvcc)..."
  _maybe_install_cuda_nvcc_debian
  echo "  Done."
}

install_fedora() {
  echo "Installing packages with dnf (Fedora/RHEL)..."
  echo "  Installing ffmpeg, OpenCV runtime libraries, PortAudio, and build tools..."
  dnf install -y \
    ffmpeg \
    mesa-libGL \
    glib2 \
    libSM \
    libXext \
    libXrender \
    portaudio-devel \
    ninja-build
  if [[ "$WITH_PYTHON" == true ]]; then
    echo "  Installing Python 3 and pip..."
    dnf install -y \
      python3 \
      python3-pip
  fi
  echo "  Done."
}

install_arch() {
  echo "Installing packages with pacman (Arch)..."
  echo "  Installing ffmpeg, OpenCV runtime libraries, PortAudio, and build tools..."
  pacman -Sy --noconfirm --needed \
    ffmpeg \
    mesa \
    glib2 \
    libsm \
    libxext \
    libxrender \
    portaudio \
    ninja
  if [[ "$WITH_PYTHON" == true ]]; then
    echo "  Installing Python and pip..."
    pacman -Sy --noconfirm --needed \
      python \
      python-pip
  fi
  echo "  Done."
}

case "${ID:-}" in
  debian|ubuntu)
    install_debian
    ;;
  fedora|rhel|centos|rocky|almalinux)
    install_fedora
    ;;
  arch|artix)
    install_arch
    ;;
  *)
    case "${ID_LIKE:-}" in
      *debian*|*ubuntu*)
        install_debian
        ;;
      *fedora*|*rhel*)
        install_fedora
        ;;
      *)
        echo "Unsupported OS: ID=$ID ID_LIKE=$ID_LIKE" >&2
        exit 1
        ;;
    esac
    ;;
esac

echo ""
echo "=============================================="
echo "  System dependencies installed"
echo "=============================================="
if [[ "$WITH_PYTHON" == true ]]; then
  echo "Python was installed. Next steps:"
  echo "  make venv"
else
  echo "Next steps: ensure Python 3.10+ is installed, then create a venv and install deps:"
  echo "  make venv"
fi
echo ""
echo "To run the full stack with Docker instead: make up"
