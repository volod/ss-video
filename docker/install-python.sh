#!/bin/sh
set -eu
# Install this repository inside Docker builders.
# ss-fusion workspace members and ss-common come from published git tags
# (pip cannot parse uv path pins).
#
# Usage:
#   sh install-python.sh --runtime     fusion-rt (no torch, no video extra)
#   sh install-python.sh '.[vision]'   video API/worker (CUDA extra from TORCH_CUDA_INDEX)
#   sh install-python.sh '.[dev]'      root package plus pytest (still pulls transformers)
#
# Heavy C++/CUDA compiles use MAX_JOBS from ss-kit. Host `make venv` and
# vision image builds install flash-attn from $DATA_DIR/wheels/flash-attn_*/
# when the ABI key matches. Compose bind-mounts that directory at /tmp/wheels.
SPEC="${1:-.}"
WHEELS_DIR="${WHEELS_DIR:-/tmp/wheels}"
TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-cpu}"
SS_FUSION_GIT="git+https://github.com/volod/ss-fusion.git@v0.1.0"

install_ss_fusion() {
  pip install --no-deps \
    "ss-perception @ ${SS_FUSION_GIT}#subdirectory=packages/ss-perception" \
    "ss-mapping @ ${SS_FUSION_GIT}#subdirectory=packages/ss-mapping" \
    "ss-fusion @ ${SS_FUSION_GIT}#subdirectory=packages/ss-fusion" \
    "fusion-rt @ ${SS_FUSION_GIT}#subdirectory=apps/fusion-rt"
}

pip install --prefer-binary \
  "ss-common @ git+https://github.com/volod/ss-common.git@v0.1.0"

if [ "$SPEC" = "--runtime" ]; then
  echo "fusion-rt runtime install (no torch, no video extra)"
  install_ss_fusion
  pip install --prefer-binary \
    "fastapi>=0.115" "uvicorn[standard]>=0.27" "asyncpg>=0.29" "redis>=5.0" \
    "aiomqtt>=2.3" "sse-starlette>=1.6" "httpx>=0.27" "python-multipart>=0.0.9" \
    "numpy>=1.26" "pillow>=10.2" "pydantic>=2.8" "python-dotenv>=1.0" \
    "PyYAML>=6.0" "jinja2>=3.1" "psutil>=5.9" "scipy>=1.10"
  pip install --prefer-binary "ss-common[web,mqtt]"
  exit 0
fi

MAX_JOBS=$(python -m ss_kit max-jobs)
export MAX_JOBS
export CMAKE_BUILD_PARALLEL_LEVEL="$MAX_JOBS"
export NINJAFLAGS="-j${MAX_JOBS}"
echo "MAX_JOBS=$MAX_JOBS (ss-kit max-jobs)"

pip install --prefer-binary ninja

# Root pyproject lists transformers (pulls torch). Pin torch from the host CUDA
# extra first so pip does not fetch an unrelated PyPI CUDA 13 build.
if [ "$TORCH_CUDA_INDEX" = "cpu" ]; then
  echo "Installing CPU torch from download.pytorch.org/whl/cpu"
  pip install --prefer-binary \
    --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision
else
  echo "Installing torch from download.pytorch.org/whl/${TORCH_CUDA_INDEX}"
  pip install --prefer-binary \
    --index-url "https://download.pytorch.org/whl/${TORCH_CUDA_INDEX}" \
    torch torchvision
fi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
TORCH_PIN=$(python -c "import torch; print(torch.__version__.split('+')[0])")
printf 'torch==%s\n' "$TORCH_PIN" >/tmp/torch-pin.txt
echo "Pinning torch==$TORCH_PIN for the remaining pip install"

install_ss_fusion

pip install --prefer-binary -c /tmp/torch-pin.txt "$SPEC"
pip install --prefer-binary -c /tmp/torch-pin.txt "ss-common[web,mqtt]"

# Reuse a host-built flash-attn wheel only when torch in this image matches
# the wheel cache key. A mismatch is skipped; models fall back to SDPA.
if [ -d "$WHEELS_DIR" ]; then
  TORCH_KEY=$(python -c "
import torch
v = torch.__version__.split('+')[0]
c = (torch.version.cuda or 'cpu').replace('.', '')
print('torch{}_cu{}'.format(v, c))
" 2>/dev/null || true)
  if [ -n "$TORCH_KEY" ]; then
    WHL=$(find "$WHEELS_DIR" -path "*flash-attn_${TORCH_KEY}*" -name 'flash_attn*.whl' 2>/dev/null | head -n 1 || true)
    if [ -n "$WHL" ]; then
      echo "flash-attn: installing prebuilt wheel $WHL"
      pip install --no-deps "$WHL"
    else
      echo "flash-attn: no wheel matching $TORCH_KEY under $WHEELS_DIR (skip)"
    fi
  else
    echo "flash-attn: torch not installed in this image (skip wheel)"
  fi
fi
