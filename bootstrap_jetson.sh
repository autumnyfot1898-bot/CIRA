#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_ENV="${SOFTCAS_JETSON_ENV:-$TASK_ROOT/.venv-jetson}"
# JetPack supplies TensorRT/CUDA. Keep the matching NVIDIA PyTorch/torchvision
# pair instead of installing desktop PyPI wheels on aarch64.
python3 -c 'import sys; assert sys.version_info >= (3,10), "SoftCAS requires Python 3.10 or newer; use a JetPack-matched environment/container"'
python3 -m venv --system-site-packages "$TASK_ENV"
TASK_PYTHON="$TASK_ENV/bin/python"
if [[ -n "${TORCH_INSTALL:-}" ]]; then
  "$TASK_PYTHON" -m pip install --no-cache-dir "$TORCH_INSTALL"
fi
if [[ -n "${TORCHVISION_INSTALL:-}" ]]; then
  "$TASK_PYTHON" -m pip install --no-cache-dir --no-deps "$TORCHVISION_INSTALL"
fi
"$TASK_PYTHON" -c 'import platform,torch,torchvision; assert platform.machine() in ("aarch64","arm64"); assert torch.cuda.is_available(), "Install the NVIDIA JetPack-matched GPU PyTorch wheel/container first"'
"$TASK_PYTHON" -m pip install 'numpy>=1.26,<2' 'scipy>=1.11,<1.16' 'Pillow>=10' 'psutil>=5.9' PyYAML tqdm requests matplotlib pandas 'ultralytics-thop>=2.0' 'onnx>=1.16,<2'
"$TASK_PYTHON" -m pip install --no-deps ultralytics==8.4.0
"$TASK_PYTHON" -m pip install --no-deps -e "$TASK_ROOT"
echo "Prepared Jetson environment; OpenCV/TensorRT remain the JetPack/system packages."
