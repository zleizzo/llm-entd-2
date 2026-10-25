#!/usr/bin/env bash
# Large Option A: GPT-2 medium (354M) on 5B FineWeb-Edu tokens, full coverage.
#   Train:   8xH100 pod (~1 h incl. tokenization), ~$35-50
#   Embed:   1xH100, ~3 h, 650 GB store on local NVMe (>=1 TB local disk), persist to a network volume
#   Search:  1xH100 from local NVMe; prefiltered brute force, ~5 min/pass
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
CFG=configs/fineweb_large.yaml
export HF_HOME=${HF_HOME:-/workspace/hf}
STAGE=${1:-all}
if [[ $STAGE == all || $STAGE == train ]]; then
  $PY -c "from entd.config import load_config; from entd.data import prepare; prepare(load_config('$CFG'))"
  torchrun --standalone --nproc_per_node ${NGPU:-8} -m entd.train $CFG
fi
if [[ $STAGE == all || $STAGE == embed ]]; then
  $PY -m entd.embed $CFG
fi
if [[ $STAGE == all || $STAGE == eval ]]; then
  $PY -m entd.eval $CFG --stages queries,search,proj,radius,fidelity,anchors
  $PY -m entd.eval $CFG --stages rerank
  $PY -m entd.eval $CFG --stages baselines,report
fi
