#!/usr/bin/env bash
# Install Unsloth from GitHub with the right torch/CUDA extra (avoids pip backtracking on mirrors).
# Run AFTER: pip install torch (matching your CUDA).
#
# Usage:
#   ./scripts/install_unsloth.sh
#   ./scripts/install_unsloth.sh cu124-torch260   # explicit extra if auto-detect fails

set -euo pipefail

EXTRA="${1:-}"

if [[ -z "$EXTRA" ]]; then
  echo "Detecting torch/CUDA for unsloth extra..."
  EXTRA=$(python - <<'PY'
import re
import torch
from packaging.version import Version as V

v = V(re.match(r"[0-9\.]{3,}", torch.__version__).group(0))
cuda = str(torch.version.cuda)
if cuda not in ("11.8", "12.1", "12.4", "12.6", "12.8", "13.0"):
    raise SystemExit(f"Unsupported CUDA {cuda}")
if v < V("2.5.0"):
    tag = "cu{}{}-torch240".format(cuda.replace(".", ""), "")
elif v < V("2.6.0"):
    tag = "cu{}{}-torch250".format(cuda.replace(".", ""), "")
elif v < V("2.7.0"):
    tag = "cu{}{}-torch260".format(cuda.replace(".", ""), "")
else:
    tag = "cu{}{}-torch270".format(cuda.replace(".", ""), "")
print(tag)
PY
)
fi

echo "Installing unsloth[${EXTRA}] from GitHub (official source)..."
pip install --upgrade pip

# Prefer PyPI.org for dependency resolution; drop Yandex-only backtracking if you use a custom index.
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"

pip install --no-deps "git+https://github.com/unslothai/unsloth-zoo.git"
pip install --no-build-isolation "unsloth[${EXTRA}] @ git+https://github.com/unslothai/unsloth.git"

echo "Done. Verify: python -c 'from unsloth import FastLanguageModel; print(\"ok\")'"
