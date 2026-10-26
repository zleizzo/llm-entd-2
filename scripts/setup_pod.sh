#!/usr/bin/env bash
# Fresh RunPod pod (PyTorch CUDA image) -> ready to run scripts/run_mvp.sh. Clone onto the LOCAL container disk
# (/root), not an NFS-backed /workspace volume.
#   git clone https://github.com/zleizzo/llm-entd-2.git /root/entd2 && cd /root/entd2 && bash scripts/setup_pod.sh
set -euo pipefail
cd "$(dirname "$0")/.."
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
pip install -q -e .
export HF_HOME=${HF_HOME:-/root/hf}
mkdir -p "$HF_HOME" data runs
python tests/test_knn.py && python tests/test_projection_metrics.py
df -h . | tail -1
echo "ok. Next: STOP_POD=1 LOG=/root/mvp.log nohup bash scripts/run_mvp.sh > /root/mvp.log 2>&1 &"
