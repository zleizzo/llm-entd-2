#!/usr/bin/env bash
# Fresh RunPod pod (PyTorch CUDA image) -> ready to run scripts/run_mvp.sh.
#   git clone https://github.com/zleizzo/llm-entd-2.git /workspace/entd2 && cd /workspace/entd2 && bash scripts/setup_pod.sh
set -euo pipefail
cd "$(dirname "$0")/.."
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
pip install -q -e .
export HF_HOME=${HF_HOME:-/workspace/hf}
mkdir -p "$HF_HOME" data runs
python tests/test_knn.py && python tests/test_projection_metrics.py
df -h . | tail -1
echo "ok. Next: nohup bash scripts/run_mvp.sh > mvp.log 2>&1 &"
