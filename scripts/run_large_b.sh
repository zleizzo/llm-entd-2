#!/usr/bin/env bash
# Large Option B: Pythia ladder (160M / 410M / 1.4B, deduped) over the same ~5B-token slice of the
# preshuffled Pile. No training. Requires:
#   git clone https://github.com/EleutherAI/pythia /workspace/pythia
#   huggingface-cli download EleutherAI/pile-deduped-pythia-preshuffled --include "document-000*" --local-dir /workspace/pile
#   (only the shards covering start_sample..start_sample+n_samples are needed; see the pythia README)
#   a pre-tokenized Pile validation stream at /workspace/pile/val.bin (GPT-NeoX tokenizer, uint16)
#   pip install transformers
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
for M in pythia-160m-deduped pythia-410m-deduped pythia-1.4b-deduped; do
  CFG=/tmp/${M}.yaml
  sed -e "s#hf_name: .*#hf_name: EleutherAI/${M}#" -e "s#^name: .*#name: ${M}_pile5b#" configs/pythia_large.yaml > $CFG
  $PY -m entd.embed $CFG
  $PY -m entd.eval $CFG --stages queries,search,proj,radius,fidelity,anchors
  $PY -m entd.eval $CFG --stages rerank
  $PY -m entd.eval $CFG --stages baselines,report
done
