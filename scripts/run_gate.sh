#!/usr/bin/env bash
# $5 coverage gate for Large Option B: embed ~100M Pile tokens with Pythia-160M and measure how the
# nearest-neighbor distance scales with store size (1M ... 100M positions). Extrapolate N^(-1/d_eff)
# to 5B vs 300B positions to decide whether 1.7% Pile coverage keeps neighbors inside the radii of
# interest found in the MVP.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
cat > /tmp/pythia_gate.yaml <<'YAML'
name: pythia_160m_gate
data:
  kind: pythia_pile
  block_size: 2048
  untrained_frac: 0.0
  pythia:
    bin_prefix: /workspace/pile/document
    mmap_module: /workspace/pythia/utils/mmap_dataset.py
    start_sample: 20480000
    n_samples: 49000          # ~100M tokens
    val_bin: /workspace/pile/val.bin
model: {kind: hf, hf_name: EleutherAI/pythia-160m-deduped}
embed: {code: pca, d: 128, store_dtype: float16, h_subset_frac: 0.0, exact_sample: 20000, pca_sample_windows: 512, batch_windows: 8}
eval:
  n_queries_heldout: 2000
  n_queries_train: 2000
  K: 1024
  coverage_fracs: [0.3, 0.1, 0.03, 0.01]
  rerank_queries: 0
  fidelity_d_sweep: [32, 64, 128]
YAML
$PY -m entd.embed /tmp/pythia_gate.yaml
$PY -m entd.eval /tmp/pythia_gate.yaml --stages queries,search,proj,fidelity,report
echo "Look at summary_proj_variants.csv: d2_k (k=1,10,100) vs coverage fraction; fit log d2 ~ -(1/d_eff) log N."
