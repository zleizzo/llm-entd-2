#!/usr/bin/env bash
# Tier 1 (TinyStories) on a single 80GB GPU pod. Stage the store on local NVMe for the search phase.
#   Pod: A100/H100 80GB, >=250 GB container/local disk, optional 250 GB network volume at /workspace/vol
#   Setup: pip install -r requirements.txt   (torch with CUDA from the pod image)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
CFG=configs/tinystories_mvp.yaml
export HF_HOME=${HF_HOME:-/workspace/hf}
$PY -m entd.train $CFG                       # ~1 h on A100 (2 epochs, 427M tokens)
$PY -m entd.embed $CFG                       # ~15 min; writes 121 GB store + 47 GB h-subset
$PY -m entd.eval $CFG --stages queries,search,proj,radius,fidelity,anchors
$PY -m entd.eval $CFG --stages rerank        # ~20-30 min (1000 queries/set x 1024 candidates)
$PY -m entd.eval $CFG --stages baselines,report
# Persist: rsync runs/tinystories_mvp/results /workspace/vol/   (the store itself can be regenerated)

# ---- persist results and optionally stop the pod ----------------------------------------------
# Results are small (a few hundred MB); copy them off the container disk if a network volume is mounted.
if [[ -d /workspace/vol ]]; then
  mkdir -p /workspace/vol/entd2 && cp -r runs/tinystories_mvp/results /workspace/vol/entd2/ && cp mvp.log /workspace/vol/entd2/ 2>/dev/null || true
fi
# STOP_POD=1 stops the pod (container disk is kept and still billed at storage rates; delete it from the
# console when done). Needs RUNPOD_API_KEY exported before launching; RUNPOD_POD_ID is set by RunPod.
if [[ "${STOP_POD:-0}" == "1" ]]; then
  runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null 2>&1 || true
  runpodctl stop pod "$RUNPOD_POD_ID"
fi
