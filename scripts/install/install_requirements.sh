#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PY_HELPER="${REPO_ROOT}/src/selfsuvis/scripts/shell_helpers.py"

DEPENDENCY_GROUPS=${1:-vision,dev}
VENV_PATH=${2:-.venv}
PYTHON_BIN="$VENV_PATH/bin/python"

if [[ ! -d "$VENV_PATH" ]]; then
  echo "Virtual environment not found at $VENV_PATH" >&2
  exit 1
fi

# Self-heal: if UV_CACHE_DIR is set and contains a stale format from an older uv
# version, uv pip install fails with a misleading EACCES (os error 13) on cache
# initialisation. Detect and clear before any install attempt.
if [[ -n "${UV_CACHE_DIR:-}" && -d "$UV_CACHE_DIR" ]]; then
  if ! uv pip install --python "$VENV_PATH" pip --dry-run >/dev/null 2>&1; then
    echo "[setup] UV cache at $UV_CACHE_DIR failed init check — clearing stale cache."
    uv cache clean 2>/dev/null || rm -rf "$UV_CACHE_DIR"
  fi
fi

# Ensure pip is available in the venv (uv-created venvs do not include it by default)
uv pip install --python "$VENV_PATH" pip

PACKAGE_SPEC="."
if [[ -n "$DEPENDENCY_GROUPS" ]]; then
  PACKAGE_SPEC=".[${DEPENDENCY_GROUPS}]"
fi

# Detect if the uv cache lives on a different filesystem than the venv.
# Cross-device hardlinks are not allowed, so we fall back to copy mode.
_detect_uv_link_mode() {
  local venv_dir="$1"
  local cache_dir="${UV_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/uv}"

  # Walk up to an existing ancestor if the cache dir itself doesn't exist yet.
  while [[ -n "$cache_dir" && ! -d "$cache_dir" ]]; do
    cache_dir="$(dirname "$cache_dir")"
  done

  local venv_dev cache_dev
  venv_dev=$(stat -c '%d' "$(dirname "$venv_dir")" 2>/dev/null || echo "")
  cache_dev=$(stat -c '%d' "$cache_dir" 2>/dev/null || echo "")

  if [[ -n "$venv_dev" && -n "$cache_dev" && "$venv_dev" != "$cache_dev" ]]; then
    echo "copy"
  fi
}

_UV_LINK_MODE=$(_detect_uv_link_mode "$VENV_PATH")
if [[ -n "$_UV_LINK_MODE" ]]; then
  echo "Cross-device uv cache detected (cache and .venv are on different disks) → using --link-mode=copy"
  export UV_LINK_MODE=copy
fi

detect_gpu_compute_capability() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
      | head -1 | tr -d ' '
  fi
}

# Build a TORCH_CUDA_ARCH_LIST value from the detected GPU compute capability.
# Single-arch+PTX compiles faster and is guaranteed safe on the current machine.
# Falls back to a broad safe list (up to sm_90) when no GPU is detected, which
# avoids compute_120 failures on nvcc versions that don't yet support Blackwell.
_arch_list_for_gpu() {
  local cc="$1"
  if [[ -n "$cc" ]]; then
    echo "${cc}+PTX"
  else
    echo "7.5;8.0;8.6;8.9;9.0+PTX"
  fi
}

# Returns the minimum torch CUDA index required for the given compute capability.
# e.g. "12.0" → "cu128" (Blackwell needs CUDA 12.8+ kernels).
# Returns "" if no special minimum applies (existing driver-based mapping suffices).
_min_torch_index_for_compute_cap() {
  local cc="$1"
  local major="${cc%%.*}"
  case "$major" in
    12) echo "cu128" ;;
    13) echo "cu130" ;;
    *) echo "" ;;
  esac
}

# True (exit 0) if cuda index $1 is strictly less than $2 (e.g. cu121 < cu128).
_cuda_index_lt() {
  local a="${1#cu}" b="${2#cu}"
  [[ "${a:-0}" -lt "${b:-0}" ]]
}

detect_cuda_version() {
  local cuda_version=""

  # Parse regular nvidia-smi text output (--query-gpu=cuda_version is not a valid field)
  if command -v nvidia-smi >/dev/null 2>&1; then
    cuda_version=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n1)
  fi

  # Fallback: nvcc reports the installed toolkit version
  if [[ -z "$cuda_version" ]] && command -v nvcc >/dev/null 2>&1; then
    cuda_version=$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n1)
  fi

  # Fallback: /usr/local/cuda version files
  if [[ -z "$cuda_version" ]] && [[ -d /usr/local/cuda ]]; then
    if [[ -f /usr/local/cuda/version.json ]]; then
      cuda_version=$("$PYTHON_BIN" "$PY_HELPER" cuda-version-from-json --path /usr/local/cuda/version.json 2>/dev/null)
    fi
    if [[ -z "$cuda_version" ]] && [[ -f /usr/local/cuda/version.txt ]]; then
      cuda_version=$(sed -n 's/.*CUDA Version \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' /usr/local/cuda/version.txt | head -n1)
    fi
  fi

  echo "$cuda_version"
}

map_cuda_to_torch_index() {
  local cuda_version="$1"
  case "$cuda_version" in
    13.*) echo "cu126" ;;
    12.9*) echo "cu126" ;;
    12.8*) echo "cu126" ;;
    12.7*) echo "cu126" ;;
    12.6*) echo "cu126" ;;
    12.5*) echo "cu126" ;;
    12.4*) echo "cu124" ;;
    12.3*) echo "cu121" ;;
    12.2*) echo "cu121" ;;
    12.1*) echo "cu121" ;;
    12.0*) echo "cu121" ;;
    11.8*) echo "cu118" ;;
    11.7*) echo "cu118" ;;
    11.6*) echo "cu118" ;;
    *) echo "" ;;
  esac
}

# Install the package plus extras from pyproject.toml, which is the single
# source of truth for Python dependencies.
uv pip install --python "$VENV_PATH" -e "$PACKAGE_SPEC"

# Published ss-fusion workspace members. Root pyproject pins these git
# subdirectory URLs; the explicit --no-deps installs keep console scripts
# (ssv, ssv-models) registered even when a caller uses --no-deps on the root.
SS_FUSION_GIT="git+https://github.com/volod/ss-fusion.git@v0.2.0"
uv pip install --python "$VENV_PATH" --no-deps \
  "ss-perception @ ${SS_FUSION_GIT}#subdirectory=packages/ss-perception"
uv pip install --python "$VENV_PATH" --no-deps \
  "ss-mapping @ ${SS_FUSION_GIT}#subdirectory=packages/ss-mapping"
uv pip install --python "$VENV_PATH" --no-deps \
  "ss-fusion @ ${SS_FUSION_GIT}#subdirectory=packages/ss-fusion"
uv pip install --python "$VENV_PATH" --no-deps \
  "fusion-rt @ ${SS_FUSION_GIT}#subdirectory=apps/fusion-rt"

CUDA_VERSION=$(detect_cuda_version)

# Allow caller to force a specific CUDA index (e.g. FORCE_CUDA=cu126 or FORCE_CUDA=1 for auto-detect)
if [[ -n "${FORCE_CUDA:-}" ]]; then
  if [[ "$FORCE_CUDA" == "1" ]]; then
    TORCH_CUDA_INDEX=$(map_cuda_to_torch_index "$CUDA_VERSION")
  else
    TORCH_CUDA_INDEX="$FORCE_CUDA"
  fi
else
  TORCH_CUDA_INDEX=$(map_cuda_to_torch_index "$CUDA_VERSION")
fi

# Override torch index if the GPU's compute capability requires newer CUDA kernels
# than the driver-version mapping above selected.
# Example: CUDA 13.2 driver → cu126 by default, but Blackwell (sm_120) needs cu128.
_GPU_CC=$(detect_gpu_compute_capability)
if [[ -n "$_GPU_CC" ]]; then
  _MIN_FOR_CC=$(_min_torch_index_for_compute_cap "$_GPU_CC")
  if [[ -n "$_MIN_FOR_CC" ]]; then
    if [[ -z "$TORCH_CUDA_INDEX" ]] || _cuda_index_lt "$TORCH_CUDA_INDEX" "$_MIN_FOR_CC"; then
      echo "GPU compute capability ${_GPU_CC} (sm_${_GPU_CC/./}): needs ${_MIN_FOR_CC}+ torch kernels"
      echo "  upgrading TORCH_CUDA_INDEX from '${TORCH_CUDA_INDEX:-none}' → ${_MIN_FOR_CC}"
      TORCH_CUDA_INDEX="$_MIN_FOR_CC"
    fi
  fi
fi

if [[ -n "$TORCH_CUDA_INDEX" ]]; then
  echo "CUDA $CUDA_VERSION detected → installing torch with $TORCH_CUDA_INDEX wheels"
  if ! uv pip install --python "$VENV_PATH" --upgrade --index-url "https://download.pytorch.org/whl/$TORCH_CUDA_INDEX" torch torchvision; then
    echo "CUDA wheel install failed, falling back to CPU"
    uv pip install --python "$VENV_PATH" --upgrade --index-url https://download.pytorch.org/whl/cpu torch torchvision
  fi
else
  echo "No CUDA detected → installing CPU-only torch"
  uv pip install --python "$VENV_PATH" --upgrade --index-url https://download.pytorch.org/whl/cpu torch torchvision
fi

# flash-attn must be installed AFTER torch (its setup.py imports torch at build time).
# --no-build-isolation lets it find the torch headers already in the active environment.
# Skipped on CPU-only machines; non-fatal on failure (models fall back to sdpa attention).

# Compute safe ninja job count for flash-attn CUDA compilation.
#
# Each parallel CUDA compilation unit (translation unit) can peak at 10-14 GiB of
# RAM — flash-attn's backward-pass kernels are especially large.  Running too many
# jobs exhausts RAM and causes the kernel to swap or OOM-kill the compiler.
#
# Heuristic:
#   1. Read total and available RAM from /proc/meminfo.
#   2. Reserve RAM_RESERVE_FRAC of *total* RAM as a hard floor (protects the OS,
#      running models, and the desktop from being crowded out).
#   3. Divide usable available RAM by RAM_PER_JOB_GB to get memory-capped jobs.
#   4. Cap at (nproc - 2) / 2 — each ninja job forks ~1 compiler thread, so N
#      jobs saturate ~2N cores; reserve 2 cores for the OS.
#   5. Always emit at least 1 job.
#
# Tune RAM_PER_JOB_GB if your GPU arch produces larger TUs.  flash-attn backward
# kernels peak well above the average; 12 GiB per job is conservative enough to
# avoid swapping even at peak compile-time working set.  Use a lower value only
# if profiling confirms lower peak RSS during compilation:
#   RTX 3090 / 4090 (Ampere/Ada, large kernels) → 10-14 GiB per job
#   RTX 3070 / 4070 / A-series mid-range        →  8-12 GiB per job
_compute_flash_attn_jobs() {
  local ram_per_job_gb=${FLASH_ATTN_RAM_PER_JOB_GB:-12}
  local reserve_frac="0.20"   # keep 20 % of total RAM free as a hard floor

  # /proc/meminfo values are in kB
  local total_kb avail_kb
  total_kb=$(awk '/^MemTotal:/  { print $2 }' /proc/meminfo 2>/dev/null || echo "8388608")
  avail_kb=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo 2>/dev/null || echo "4194304")

  local cpu_cores
  cpu_cores=$(nproc 2>/dev/null || echo "4")

  "$PYTHON_BIN" "$PY_HELPER" compute-flash-attn-jobs \
    --total-kb "$total_kb" \
    --avail-kb "$avail_kb" \
    --cpu-cores "$cpu_cores" \
    --ram-per-job-gb "$ram_per_job_gb" \
    --reserve-frac "$reserve_frac"
}

# -- nvcc / torch alignment helpers --------------------------------------------
#
# flash-attn's setup.py hard-checks that `nvcc --version` == torch.version.cuda.
# These helpers ensure that invariant holds before attempting the build by:
#   1. looking for a matching nvcc in /usr/local/cuda-*/
#   2. trying  sudo apt-get install cuda-nvcc-MAJOR-MINOR
#   3. if that fails: adding the NVIDIA CUDA apt repo, then retrying
#   4. falling back through supported (nvcc, torch) pairs — reinstalling torch
#      to match whatever nvcc version can be installed
#   5. registering the active nvcc with update-alternatives so it is the default

# Return the CUDA home directory for the given "MAJOR.MINOR" nvcc version, or "".
_find_cuda_home() {
  local ver="$1" major="${1%%.*}" minor="${1##*.}"
  for d in \
      "/usr/local/cuda-${ver}" \
      "/usr/local/cuda-${major}.${minor}" \
      "/usr/local/cuda-${major}-${minor}"; do
    if [[ -x "$d/bin/nvcc" ]]; then echo "$d"; return 0; fi
  done
  return 1
}

# Register nvcc at PATH $1 (e.g. /usr/local/cuda-12.6/bin/nvcc) version $2 ("12.6")
# as the system default via update-alternatives.  Non-fatal.
_set_nvcc_alternative() {
  local nvcc_path="$1" ver="$2"
  command -v update-alternatives >/dev/null 2>&1 || return 0
  local major="${ver%%.*}" minor="${ver##*.}"
  local priority=$(( major * 10 + minor ))
  sudo update-alternatives --install /usr/bin/nvcc nvcc "$nvcc_path" "$priority" 2>/dev/null || true
  sudo update-alternatives --set nvcc "$nvcc_path" 2>/dev/null && \
    echo "  update-alternatives: nvcc default → ${nvcc_path} (priority ${priority})" || true
}

# Add the official NVIDIA CUDA apt repository for the current Debian/Ubuntu distro.
# Downloads the cuda-keyring .deb from developer.download.nvidia.com, installs it,
# and runs apt-get update.  Returns 0 on success, 1 if anything fails.
_add_nvidia_cuda_apt_repo() {
  command -v apt-get >/dev/null 2>&1 || return 1
  [[ -f /etc/os-release ]] || return 1

  # If the cuda-keyring package is already installed the repo is already present.
  dpkg -s cuda-keyring >/dev/null 2>&1 && {
    echo "  NVIDIA CUDA apt repo already present (cuda-keyring installed)."
    return 0
  }

  local os_id os_ver
  # shellcheck source=/dev/null
  . /etc/os-release
  os_id="${ID:-ubuntu}"
  os_ver="${VERSION_ID:-22.04}"

  # Build the distro token used in NVIDIA's URL (e.g. ubuntu2204, debian12).
  local distro="${os_id}${os_ver//./}"

  # Architecture token (NVIDIA uses x86_64 / sbsa / ppc64le).
  local arch
  arch=$(dpkg --print-architecture 2>/dev/null || uname -m)
  local nvidia_arch
  case "$arch" in
    amd64|x86_64)  nvidia_arch="x86_64" ;;
    arm64|aarch64) nvidia_arch="sbsa"   ;;
    ppc64le)       nvidia_arch="ppc64le" ;;
    *)             nvidia_arch="x86_64" ;;
  esac

  local base_url="https://developer.download.nvidia.com/compute/cuda/repos/${distro}/${nvidia_arch}"
  local keyring_deb="/tmp/cuda-keyring_1.1-1_all.deb"

  echo "  Adding NVIDIA CUDA apt repo: ${base_url} ..."

  # Download the keyring package.
  if command -v wget >/dev/null 2>&1; then
    wget -q -O "$keyring_deb" "${base_url}/cuda-keyring_1.1-1_all.deb" || {
      echo "  wget failed — distro '${distro}' may not be supported by NVIDIA's repo."
      return 1
    }
  elif command -v curl >/dev/null 2>&1; then
    curl -fsSL -o "$keyring_deb" "${base_url}/cuda-keyring_1.1-1_all.deb" || {
      echo "  curl failed — distro '${distro}' may not be supported by NVIDIA's repo."
      return 1
    }
  else
    echo "  wget and curl not found — cannot download CUDA keyring."
    return 1
  fi

  sudo dpkg -i "$keyring_deb" 2>/dev/null && rm -f "$keyring_deb" || {
    rm -f "$keyring_deb"
    echo "  dpkg -i cuda-keyring failed."
    return 1
  }

  echo "  Running apt-get update ..."
  sudo apt-get update -qq 2>/dev/null || true
  echo "  NVIDIA CUDA apt repo added."
  return 0
}

# Try  sudo apt-get install cuda-nvcc-MAJOR-MINOR.  Returns 0 on success.
_apt_install_nvcc() {
  local major="$1" minor="$2"
  command -v apt-get >/dev/null 2>&1 || return 1
  sudo apt-get install -y --no-install-recommends "cuda-nvcc-${major}-${minor}" 2>/dev/null
}

# Reinstall torch + torchvision from the given CUDA index (e.g. "cu124").
_reinstall_torch() {
  local idx="$1"
  uv pip install --python "$VENV_PATH" --upgrade \
    --index-url "https://download.pytorch.org/whl/${idx}" \
    torch torchvision
}

# Ensure nvcc version == torch.version.cuda before building flash-attn.
#
# Strategy (in order):
#   A. system nvcc already matches → done
#   B. matching toolkit exists in /usr/local/cuda-*  → set CUDA_HOME + alternative
#   C. apt-install ideal nvcc; if that fails, add NVIDIA apt repo then retry
#   D. fallback chain: try each supported (nvcc, torch-index) pair;
#      add NVIDIA apt repo once if needed; reinstall torch to match
#
# Always exits 0.  Sets TORCH_CUDA_INDEX to the active index after any reinstall.
_align_nvcc_torch() {
  local torch_ver
  torch_ver=$("$PYTHON_BIN" -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "")
  [[ -z "$torch_ver" ]] && return 0

  local t_major="${torch_ver%%.*}" t_minor="${torch_ver##*.}"
  local sys_nvcc
  sys_nvcc=$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1)

  # A — already aligned
  if [[ "$sys_nvcc" == "$torch_ver" ]]; then
    echo "nvcc ${sys_nvcc} == torch.version.cuda — OK."
    return 0
  fi

  # B — matching toolkit already installed elsewhere; set CUDA_HOME + alternative
  local found
  if found=$(_find_cuda_home "$torch_ver") 2>/dev/null; then
    echo "CUDA_HOME → ${found}  (nvcc ${torch_ver} matches torch)"
    export CUDA_HOME="$found"; export PATH="$found/bin:$PATH"
    _set_nvcc_alternative "$found/bin/nvcc" "$torch_ver"
    return 0
  fi

  echo "nvcc ${sys_nvcc:-none} ≠ torch.version.cuda ${torch_ver} — attempting to fix ..."

  # Helper: install nvcc, set CUDA_HOME + alternative, return 0 on success.
  _install_and_activate_nvcc() {
    local ver="$1" major="${1%%.*}" minor="${1##*.}"
    _apt_install_nvcc "$major" "$minor" || return 1
    local home
    if home=$(_find_cuda_home "$ver") 2>/dev/null; then
      export CUDA_HOME="$home"; export PATH="$home/bin:$PATH"
      _set_nvcc_alternative "$home/bin/nvcc" "$ver"
    else
      # apt may have placed nvcc on PATH directly (e.g. /usr/bin/nvcc symlink)
      _set_nvcc_alternative "$(command -v nvcc)" "$ver" 2>/dev/null || true
    fi
    echo "cuda-nvcc-${major}-${minor} installed — nvcc ${ver} ready."
    return 0
  }

  # C — install ideal nvcc; add NVIDIA apt repo if first attempt fails
  _repo_added=false
  if _install_and_activate_nvcc "$torch_ver"; then
    return 0
  fi
  echo "  First apt attempt failed — adding NVIDIA CUDA apt repository ..."
  if _add_nvidia_cuda_apt_repo; then
    _repo_added=true
    if _install_and_activate_nvcc "$torch_ver"; then
      return 0
    fi
  fi
  echo "  cuda-nvcc-${t_major}-${t_minor} not available. Trying fallback nvcc versions ..."

  # D — fallback: try other (nvcc, torch-index) pairs highest-to-lowest
  local pairs=("12.6:cu126" "12.4:cu124" "12.1:cu121" "11.8:cu118")
  for pair in "${pairs[@]}"; do
    local fb_ver="${pair%%:*}" fb_idx="${pair##*:}"
    local fb_major="${fb_ver%%.*}" fb_minor="${fb_ver##*.}"
    [[ "$fb_ver" == "$torch_ver" ]] && continue  # already tried in C

    # Ensure this nvcc version is available.
    if ! found=$(_find_cuda_home "$fb_ver") 2>/dev/null; then
      if ! _apt_install_nvcc "$fb_major" "$fb_minor"; then
        # Repo not yet added — add it once and retry.
        if [[ "$_repo_added" == false ]]; then
          _add_nvidia_cuda_apt_repo && _repo_added=true
          _apt_install_nvcc "$fb_major" "$fb_minor" || continue
        else
          continue
        fi
      fi
      found=$(_find_cuda_home "$fb_ver") 2>/dev/null || true
    fi

    echo "Found nvcc ${fb_ver} — reinstalling torch with ${fb_idx} ..."
    if _reinstall_torch "$fb_idx"; then
      TORCH_CUDA_INDEX="$fb_idx"
      if [[ -n "$found" ]]; then
        export CUDA_HOME="$found"; export PATH="$found/bin:$PATH"
        _set_nvcc_alternative "$found/bin/nvcc" "$fb_ver"
      fi
      echo "torch reinstalled (${fb_idx}); nvcc ${fb_ver} will build flash-attn."
      return 0
    fi
  done

  echo "WARNING: could not install any supported nvcc version."
  echo "  flash-attn build will be attempted but may fail."
  echo "  Manual fix: https://developer.nvidia.com/cuda-downloads"
  echo "  Then: sudo apt-get install cuda-nvcc-${t_major}-${t_minor} && make venv"
}

# -- GPU compute-capability detection ------------------------------------------
_GPU_CC=$(detect_gpu_compute_capability)
_CC_MAJOR="${_GPU_CC%%.*}"
_FORCE_SOURCE_BUILD=false

# Align nvcc with torch.version.cuda (sets CUDA_HOME / PATH as needed).
_align_nvcc_torch

# For new GPU architectures (sm_12x Blackwell and later) that have no prebuilt
# flash-attn wheel, force a source build and include all common arch targets so
# the resulting wheel also runs on more powerful machines (sm_80/sm_90).
if [[ -n "$_CC_MAJOR" ]] && [[ "$_CC_MAJOR" -ge 12 ]]; then
  _INSTALLED_SO=$(find "${VENV_PATH}" -name "flash_attn_2_cuda*.so" 2>/dev/null | head -1)
  _NEEDS_REBUILD=true
  if [[ -n "$_INSTALLED_SO" ]]; then
    # Use `strings` rather than cuobjdump: older cuobjdump versions (e.g. CUDA 12.0)
    # cannot parse Blackwell (sm_12x) ELF sections and silently omit them, causing
    # a false-negative arch check that triggers an unnecessary source rebuild.
    if strings "$_INSTALLED_SO" 2>/dev/null | grep -q "sm_${_CC_MAJOR}"; then
      _NEEDS_REBUILD=false
      echo "flash-attn: sm_${_CC_MAJOR}x kernels already in installed binary — skipping rebuild"
    fi
  fi
  if $_NEEDS_REBUILD; then
    echo "flash-attn: building from source for sm_${_CC_MAJOR}0 (no prebuilt wheel available)"
    _FORCE_SOURCE_BUILD=true
    # Covers: Turing(7.5) Ampere-DC(8.0) Ampere-consumer(8.6) Ada/4060(8.9) Hopper(9.0)
    # +PTX on the last entry allows JIT forward-compatibility for newer archs.
    #
    # IMPORTANT: Do not include "12.0" here unless your nvcc supports it.
    # CUDA 12.6 nvcc will fail with: "Unsupported gpu architecture 'compute_120'".
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$(_arch_list_for_gpu "$_GPU_CC")}"
    find ~/.cache/pip/wheels -name "flash_attn*.whl" -delete 2>/dev/null || true
    find ~/.cache/uv         -name "flash_attn*.whl" -delete 2>/dev/null || true
  fi
fi

if "$PYTHON_BIN" -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  # ninja parallelises CUDA kernel compilation — without it flash-attn takes 10-30 min.
  # Check system ninja first, then fall back to the pip-bundled wheel.
  if command -v ninja >/dev/null 2>&1; then
    echo "ninja found at $(command -v ninja) — flash-attn compilation will be fast."
  else
    echo "ninja not found in PATH — installing via pip (speeds up flash-attn build) …"
    if "$PYTHON_BIN" -m pip install ninja -q; then
      echo "ninja (pip) installed."
    else
      echo "WARNING: ninja install failed. flash-attn will compile without it (slow, ~30 min)."
      echo "  To fix: sudo apt-get install -y ninja-build"
    fi
  fi

  # Determine safe parallelism.
  # read returns 1 on EOF-without-newline; || true prevents set -e from aborting.
  read -r FLASH_JOBS _total _avail _usable _mem_cap _cpu_cap < <(_compute_flash_attn_jobs) || true
  FLASH_JOBS="${FLASH_JOBS:-1}"
  echo ""
  echo "-- flash-attn compilation resource budget ---------------------------------"
  echo "  RAM total / available / usable : ${_total:-?} GiB / ${_avail:-?} GiB / ${_usable:-?} GiB"
  echo "  RAM per job (FLASH_ATTN_RAM_PER_JOB_GB) : ${FLASH_ATTN_RAM_PER_JOB_GB:-12} GiB"
  echo "  CPU cores   : $(nproc)  →  limit: ${_cpu_cap:-?} ((cores-2)/2, each job forks ~1 extra thread)"
  echo "  Memory jobs : ${_mem_cap:-?}"
  echo "  → MAX_JOBS  : ${FLASH_JOBS}  (min of memory and CPU limits)"
  echo "  (Set FLASH_ATTN_RAM_PER_JOB_GB=N to override per-job RAM estimate)"
  echo "----------------------------------------------------------------------------"
  echo ""

  # -- wheel cache lookup -------------------------------------------------------
  # Shared cache at .data/wheels/.  Cache key:
  #   torch{ver}_cu{cuda}_{sm}_nvcc{nvcc}
  # Exact match first; then a loose match on cu/sm/nvcc (any torch minor) so a
  # previously built wheel can be reused here when torch versions align or are
  # ABI-compatible (flash-attn 2.x is stable across torch 2.x minors).
  _FA_WHEEL_CACHE="${REPO_ROOT}/.data/wheels"
  _SYS_NVCC_VER=$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1 || echo "")
  _NVCC_COMPACT="${_SYS_NVCC_VER//.}"
  _FA_CACHE_KEY=$("$PYTHON_BIN" -c "
import torch, sys
v = torch.__version__.split('+')[0]
c = (torch.version.cuda or 'cpu').replace('.','')
caps = sorted({torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())})
sm = '_'.join('sm'+''.join(str(x) for x in cap) for cap in caps)
print('torch{}_cu{}_{}_nvcc{}'.format(v, c, sm, sys.argv[1]))
" "$_NVCC_COMPACT" 2>/dev/null || echo "")
  _FA_CACHE_DIR="${_FA_WHEEL_CACHE}/flash-attn_${_FA_CACHE_KEY}"
  _FA_CACHED_WHEEL=""

  if [[ -n "$_FA_CACHE_KEY" ]] && ls "${_FA_CACHE_DIR}"/flash_attn*.whl 1>/dev/null 2>&1; then
    _FA_CACHED_WHEEL=$(ls "${_FA_CACHE_DIR}"/flash_attn*.whl | head -1)
    echo "flash-attn: exact cached wheel found → $(basename "$_FA_CACHED_WHEEL")"
  elif [[ -n "$_NVCC_COMPACT" ]]; then
    _cu_tag=$("$PYTHON_BIN" -c "import torch; print((torch.version.cuda or 'cpu').replace('.','')); " 2>/dev/null || echo "")
    _sm_tag=$("$PYTHON_BIN" -c "import torch; caps=sorted({torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())}); print('_'.join('sm'+''.join(str(x) for x in cap) for cap in caps))" 2>/dev/null || echo "")
    if [[ -n "$_cu_tag" ]] && [[ -n "$_sm_tag" ]]; then
      for _fa_cand in "${_FA_WHEEL_CACHE}"/flash-attn_*_cu${_cu_tag}_${_sm_tag}_nvcc${_NVCC_COMPACT}; do
        if [[ -d "$_fa_cand" ]] && ls "${_fa_cand}"/flash_attn*.whl 1>/dev/null 2>&1; then
          _FA_CACHED_WHEEL=$(ls "${_fa_cand}"/flash_attn*.whl | head -1)
          echo "flash-attn: compatible cached wheel (cu/sm/nvcc match) → $(basename "$_FA_CACHED_WHEEL")"
          break
        fi
      done
    fi
  fi

  if [[ -n "$_FA_CACHED_WHEEL" ]]; then
    if "$PYTHON_BIN" -m pip install "$_FA_CACHED_WHEEL" --no-deps --quiet; then
      echo "flash-attn installed (from cache)."
    else
      echo "WARNING: cached wheel install failed — will build from source."
      _FA_CACHED_WHEEL=""
    fi
  fi

  if [[ -z "$_FA_CACHED_WHEEL" ]]; then
    echo "Installing flash-attn with MAX_JOBS=${FLASH_JOBS} …"
    # torch/torchvision are already installed above from the selected PyTorch index.
    # Do not let pip re-resolve them here, or it may pull a newer torch from PyPI
    # that no longer matches the pinned torchvision CUDA wheel.
    _flash_attn_flags="--no-build-isolation --no-deps"
    if $_FORCE_SOURCE_BUILD; then
      # --no-binary  : build from source (skip any prebuilt wheel for wrong arch)
      # --no-cache-dir   : prevent pip from serving the old cached wheel
      _flash_attn_flags="$_flash_attn_flags --no-binary flash-attn --no-cache-dir"
    fi

    if [[ -n "$_FA_CACHE_KEY" ]]; then
      mkdir -p "$_FA_CACHE_DIR"
      _FA_BUILD_TMP=$(mktemp -d)
      # shellcheck disable=SC2086
      if MAX_JOBS="$FLASH_JOBS" "$PYTHON_BIN" -m pip wheel flash-attn \
          --wheel-dir "$_FA_BUILD_TMP" \
          $_flash_attn_flags -q; then
        _built_whl=$(ls "${_FA_BUILD_TMP}"/flash_attn*.whl 2>/dev/null | head -1)
        if [[ -n "$_built_whl" ]]; then
          cp "$_built_whl" "$_FA_CACHE_DIR/"
          echo "Wheel cached at ${_FA_CACHE_DIR}"
          "$PYTHON_BIN" -m pip install "${_FA_CACHE_DIR}"/flash_attn*.whl --no-deps --quiet \
            && echo "flash-attn installed."
        fi
      else
        echo "WARNING: flash-attn build failed. Run later with:"
        echo "  python -m selfsuvis.scripts.prepare_models --flash-attn"
        echo "Models will use sdpa (PyTorch built-in SDPA) until then."
      fi
      rm -rf "$_FA_BUILD_TMP"
    else
      # shellcheck disable=SC2086
      if MAX_JOBS="$FLASH_JOBS" "$PYTHON_BIN" -m pip install flash-attn $_flash_attn_flags -q; then
        echo "flash-attn installed."
      else
        echo "WARNING: flash-attn build failed. Run later with:"
        echo "  python -m selfsuvis.scripts.prepare_models --flash-attn"
        echo "Models will use sdpa (PyTorch built-in SDPA) until then."
      fi
    fi
  fi
else
  echo "No CUDA GPU detected — skipping flash-attn (CPU-only mode)."
fi

# -- xformers source build ------------------------------------------------------
#
# The prebuilt xformers wheel on PyPI is compiled for a fixed set of CUDA
# architectures (typically sm_75, sm_80, sm_90).  GPUs not in that set fall
# back to PTX JIT, which is slower and — for Blackwell (sm_12x) — some
# attention kernels are explicitly gated to sm_<=9.0 and simply fail.
#
# This section detects whether the installed wheel covers the current GPU and,
# if not, rebuilds xformers from source with a comprehensive arch list that
# covers all common consumer and data-centre GPUs:
#   7.5  Turing       RTX 2000 / T4
#   8.0  Ampere-DC    A100 / A30
#   8.6  Ampere       RTX 3000 series
#   8.9  Ada          RTX 4000 series (4060, 4070, 4080, 4090)
#   9.0  Hopper       H100 / H800
#  +PTX  forward-compatibility JIT for future architectures
#
# The build uses the same FLASH_JOBS parallelism computed for flash-attn.
# Expected build time: 20–60 min depending on machine.  Progress is printed.

# Returns 0 (true) if the installed xformers was compiled for GPU CC $1 (e.g. "12.0").
_xformers_arch_supported() {
  local gpu_cc="$1"
  local major="${gpu_cc%%.*}" minor="${gpu_cc##*.}"
  "$PYTHON_BIN" -c "
import subprocess, re, sys
major, minor = int('$major'), int('$minor')
try:
    r = subprocess.run([sys.executable, '-m', 'xformers.info'],
                       capture_output=True, text=True, timeout=30)
except Exception:
    sys.exit(1)
for line in r.stdout.splitlines():
    if 'TORCH_CUDA_ARCH_LIST' in line:
        val = line.split(':', 1)[-1].strip()
        if not val or val.lower() == 'none':
            sys.exit(1)
        # Parse tokens like '8.6', '9.0a', '9.0+PTX', '80' etc.
        for token in re.split(r'[\s;,]+', val):
            token = re.sub(r'[a+].*', '', token)   # strip 'a' / '+PTX'
            parts = token.split('.')
            try:
                t_maj = int(parts[0])
                t_min = int(parts[1]) if len(parts) > 1 else 0
                if t_maj == major and t_min == minor:
                    sys.exit(0)
            except ValueError:
                pass
        break
sys.exit(1)
" 2>/dev/null
}

_XFORMERS_NEEDS_REBUILD=false
if [[ -n "$_GPU_CC" ]]; then
  if ! _xformers_arch_supported "$_GPU_CC"; then
    _XFORMERS_NEEDS_REBUILD=true
  fi
fi

if "$PYTHON_BIN" -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null && \
   $_XFORMERS_NEEDS_REBUILD; then
  # Reuse the same arch list set by the flash-attn section (or set it now for
  # machines that only need an xformers rebuild without flash-attn).
  _XFORMERS_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$(_arch_list_for_gpu "$_GPU_CC")}"
  _XFORMERS_JOBS="${FLASH_JOBS:-2}"

  echo ""
  echo "========================================================================"
  echo "  xformers: GPU sm_${_GPU_CC/./} not in prebuilt wheel — building from source"
  echo "  TORCH_CUDA_ARCH_LIST=${_XFORMERS_ARCH_LIST}"
  echo "  MAX_JOBS=${_XFORMERS_JOBS}"
  echo "  Expected build time: 20–60 min.  Output is shown — do not interrupt."
  echo "  To force a manual rebuild at any time:  make venv-rebuild-xformers"
  echo "========================================================================"
  echo ""

  # Clear any cached prebuilt wheel so pip always compiles from source.
  find ~/.cache/pip/wheels -name "xformers*.whl" -delete 2>/dev/null || true
  find ~/.cache/uv         -name "xformers*.whl" -delete 2>/dev/null || true

  if TORCH_CUDA_ARCH_LIST="$_XFORMERS_ARCH_LIST" \
     MAX_JOBS="$_XFORMERS_JOBS" \
     "$PYTHON_BIN" -m pip install xformers \
       --no-build-isolation \
       --no-deps \
       --no-binary xformers \
       --force-reinstall \
       --no-cache-dir; then
    echo ""
    echo "xformers rebuilt for sm_${_GPU_CC/./} — all attention kernels now native."
  else
    echo ""
    echo "WARNING: xformers source build failed."
    echo "  Models will fall back to PyTorch SDPA (correct but slower)."
    echo "  To retry manually:"
    echo "    TORCH_CUDA_ARCH_LIST='${_XFORMERS_ARCH_LIST}' \\"
    echo "    MAX_JOBS=${_XFORMERS_JOBS} \\"
    echo "    .venv/bin/python -m pip install xformers \\"
    echo "      --no-build-isolation --no-binary xformers --force-reinstall --no-cache-dir"
    echo "  Ensure nvcc $(\"$PYTHON_BIN\" -c 'import torch; print(torch.version.cuda)' 2>/dev/null) is on PATH."
    echo "  Or run:  make venv-rebuild-xformers"
  fi
elif ! $_XFORMERS_NEEDS_REBUILD && [[ -n "$_GPU_CC" ]]; then
  echo "xformers: sm_${_GPU_CC/./} already in installed wheel — skipping rebuild."
fi

# -- TensorRT + onnxruntime-gpu -------------------------------------------------
# onnxruntime-gpu replaces the CPU-only onnxruntime installed from pyproject.toml.
# It adds CUDAExecutionProvider and TensorrtExecutionProvider, giving the fastest
# ONNX inference on NVIDIA GPUs.  TensorRT must also be present at runtime for
# TensorrtExecutionProvider to activate; we install it from NVIDIA's PyPI index.
#
# TensorRT CUDA compatibility:
#   CUDA 12.x → TensorRT 10.x  (tensorrt>=10.0 from pypi.nvidia.com)
#   CUDA 11.8 → TensorRT 8.x   (tensorrt<9; use apt or older wheel)
#
# Non-fatal: falls back to CPU onnxruntime if GPU install fails.
if [[ -n "$TORCH_CUDA_INDEX" ]]; then
  echo ""
  echo "-- onnxruntime-gpu + TensorRT -----------------------------------------------"

  # Use uv (not pip) to handle the onnxruntime / onnxruntime-gpu package conflict.
  # uv replaces whichever variant is currently installed without leaving the venv
  # in a broken state.  No explicit uninstall is needed — uv's resolver handles it.
  echo "Installing onnxruntime-gpu (CUDAExecutionProvider) …"
  if uv pip install --python "$VENV_PATH" "onnxruntime-gpu>=1.18.0"; then
    echo "  onnxruntime-gpu installed."
  else
    echo "  WARNING: onnxruntime-gpu install failed — keeping CPU onnxruntime."
    uv pip install --python "$VENV_PATH" "onnxruntime>=1.18.0" || true
  fi

  # onnxruntime-gpu and onnxruntime use the same Python package name at import
  # time but are distinct pip packages.  If the import is broken (e.g. conflict
  # left a gap) reinstall the CPU fallback so the runtime is always importable.
  if ! "$PYTHON_BIN" -c "import onnxruntime" 2>/dev/null; then
    echo "  onnxruntime not importable after GPU install — reinstalling CPU fallback."
    uv pip install --python "$VENV_PATH" "onnxruntime>=1.18.0" || true
  fi

  # TensorRT wheels are published by NVIDIA at https://pypi.nvidia.com.
  # Version selection:
  #   CUDA 12.x → tensorrt>=10 (cu12 variant)
  #   CUDA 11.8 → tensorrt<9   (cu11 bindings are not on pypi.nvidia.com; use apt)
  echo "Installing TensorRT Python package (enables TensorrtExecutionProvider) …"
  if [[ "$TORCH_CUDA_INDEX" == cu118 ]]; then
    echo "  CUDA 11.8 detected — TensorRT 10+ is not available for CUDA 11."
    echo "  Install TensorRT 8.x via apt from the NVIDIA TensorRT apt repo, then"
    echo "  the TensorrtExecutionProvider will activate automatically."
    echo "  See: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html"
  else
    # CUDA 12.x — use NVIDIA's PyPI index (pypi.nvidia.com publishes cu12 wheels).
    if "$PYTHON_BIN" -m pip install \
        "tensorrt>=10.0" \
        "tensorrt-cu12-bindings" \
        "tensorrt-cu12-libs" \
        --extra-index-url https://pypi.nvidia.com \
        -q; then
      echo "  TensorRT installed (TensorrtExecutionProvider will be active)."
    else
      echo "  WARNING: TensorRT pip install failed. Possible fixes:"
      echo "    pip install tensorrt --extra-index-url https://pypi.nvidia.com"
      echo "    sudo apt-get install -y tensorrt  (requires nvidia-tensorrt apt repo)"
      echo "    See: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html"
      echo "  Without TensorRT, onnxruntime-gpu will use CUDAExecutionProvider only."
    fi
  fi
else
  echo ""
  echo "No CUDA detected — keeping CPU-only onnxruntime (no TensorRT)."
fi
