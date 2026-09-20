#!/usr/bin/env bash
# Install NVIDIA Container Toolkit so Docker can use the GPU.
# Requires: Docker installed, NVIDIA driver installed on the host.
# Usage: sudo ./scripts/install_nvidia_docker.sh
set -euo pipefail

echo "=============================================="
echo "  NVIDIA Container Toolkit installer"
echo "=============================================="
echo "This script installs the toolkit so Docker containers can use your NVIDIA GPU."
echo ""

if [[ $(id -u) -ne 0 ]]; then
  echo "This script must be run as root." >&2
  echo "Run: sudo $0" >&2
  exit 1
fi

echo "Checking for NVIDIA driver..."
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. Install the NVIDIA driver first (e.g. sudo apt install nvidia-driver-535)." >&2
  exit 1
fi
echo "  Found nvidia-smi. Proceeding."
echo ""

if [[ -f /etc/os-release ]]; then
  # shellcheck source=/dev/null
  source /etc/os-release
fi

install_ubuntu_debian() {
  echo "Detected Debian/Ubuntu. Installing NVIDIA Container Toolkit..."
  apt-get update
  apt-get install -y curl gpg
  echo "  Adding NVIDIA package repository..."
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update
  echo "  Installing nvidia-container-toolkit package..."
  apt-get install -y nvidia-container-toolkit
  echo "  Configuring Docker to use the NVIDIA runtime..."
  nvidia-ctk runtime configure --runtime=docker
  echo "  Restarting Docker daemon..."
  systemctl restart docker 2>/dev/null || service docker restart 2>/dev/null || true
  echo "  Done."
}

install_fedora() {
  echo "Detected Fedora/RHEL. Installing NVIDIA Container Toolkit..."
  dnf install -y curl
  echo "  Adding NVIDIA package repository..."
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | tee /etc/yum.repos.d/nvidia-container-toolkit.repo
  echo "  Installing nvidia-container-toolkit package..."
  dnf install -y nvidia-container-toolkit
  echo "  Configuring Docker to use the NVIDIA runtime..."
  nvidia-ctk runtime configure --runtime=docker
  echo "  Restarting Docker daemon..."
  systemctl restart docker
  echo "  Done."
}

case "${ID:-}" in
  ubuntu|debian)
    install_ubuntu_debian
    ;;
  fedora|rhel|centos|rocky|almalinux)
    install_fedora
    ;;
  *)
    case "${ID_LIKE:-}" in
      *debian*|*ubuntu*)
        install_ubuntu_debian
        ;;
      *fedora*)
        install_fedora
        ;;
      *)
        echo "Unsupported OS: ID=$ID ID_LIKE=$ID_LIKE" >&2
        echo "See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html" >&2
        exit 1
        ;;
    esac
    ;;
esac

echo ""
echo "=============================================="
echo "  Installation complete"
echo "=============================================="
echo "To verify GPU access from Docker, run:"
echo "  docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi"
echo ""
echo "Then run  make test  or  make up  to use the stack with GPU."
