# entd: next-token-distribution embeddings of a pretraining corpus

Embed every position of a language model's pretraining corpus as the **final model's next-token
distribution** at that position (computed with the exact context the position had during training), then
ask: for a new input, do the actual next tokens of the nearest corpus positions in that space follow the
model's own prediction?

This is a nearest-neighbor estimator of **canonical (full-vector) calibration**, binned in the model's own
output space, plus a memorization probe (in-train vs held-out queries, trained vs untrained store).

## Pipeline

| stage | module | what it does |
|---|---|---|
| data | `entd/data.py` | corpus → uint16 token streams; explicit window table (T+1 tokens per window); trained / untrained-store split; persisted step schedule |
| train | `entd/train.py` | nanoGPT-style GPT over fixed windows in the persisted order (DDP via torchrun) |
| embed | `entd/embed.py` | final model over every window → `sqrt(softmax)` → linear code (exact / PCA / random) → memmap store (`z`, `‖z‖²`, next token, argmax, entropy); hidden-state subset; exact-distribution sample |
| search | `entd/knn.py` | brute force streamed over the store with an elementwise threshold prefilter; exact top-K + radius counts; radius histograms over a sub-store |
| re-rank | `entd/rerank.py` | exact distributions for the top candidates (re-run their windows); geometry / calibration decomposition; consistency-resampling null; log-score statistic `T` |
| baselines | `entd/suffix.py`, `entd/eval.py` | unigram; same-argmax and argmax×entropy buckets; longest-suffix n-gram via a suffix array; hidden-state kNN (kNN-LM keys); coverage ablation; untrained store |
| analysis | `entd/metrics.py`, `entd/report.py` | JS/TV/Hellinger, naive and exact nulls, λ-interpolation, window-bootstrap CIs, plots and `summary.md` |

Distance convention: `d = ‖√p − √q‖₂` (so `d² = 2·H²`, Bhattacharyya `= 1 − d²/2`).

## Quick start (tier 0, laptop)

```bash
uv venv --python python3 .venv && uv pip install --python .venv/bin/python -e .
.venv/bin/python tests/test_knn.py && .venv/bin/python tests/test_projection_metrics.py
scripts/run_tier0.sh            # train → embed → eval → runs/shakespeare_char/results/report/summary.md
```

## RunPod: running the MVP on a fresh pod

Pod: one A100 or H100 80GB, PyTorch CUDA image, **>= 450 GB container disk** (two models x (121 GB store +
47 GB h-subset) + data + headroom), plus a small (50 GB) **network volume at /workspace** that receives the
persisted artifacts. Work on the container disk (local NVMe): on many hosts the /workspace pod volume is
NFS-backed at ~100 MB/s writes, which makes the embed stage several times slower.

```bash
git clone https://github.com/zleizzo/llm-entd-2.git /root/entd2 && cd /root/entd2
bash scripts/setup_pod.sh                       # installs the package, runs the unit tests
LOG=/root/mvp.log nohup bash scripts/run_mvp.sh > /root/mvp.log 2>&1 &
tail -f /root/mvp.log
```

`run_mvp.sh` trains for 2 epochs (~1 h; `entd.train` saves a model-only `ckpt_epoch<k>.pt` at every epoch
boundary), and runs embed + the full eval suite on two models: the final one (`configs/tinystories_mvp.yaml`,
`runs/tinystories_mvp/`) and the epoch-1 checkpoint (`configs/tinystories_mvp_ep1.yaml`, which `extends`
the main config and sets `model.checkpoint`; `runs/tinystories_mvp_ep1/`). Both share the data, window
tables, and trained/untrained split; the epoch-1 eval derives its `last_seen_step` from the main run's
schedule truncated at that checkpoint's step.

With `PARALLEL=1` (default) the epoch-1 pipeline starts as soon as `ckpt_epoch1.pt` appears, overlapping the
second training epoch, and the final-model pipeline starts when training ends, so the two pipelines share
the GPU (`eval.chunk` is kept at 131072 so two search processes fit in 80 GB). Each pipeline logs to
`runs/<name>/pipeline.log`; `PARALLEL=0` runs everything in sequence. After training and after each pipeline
the small artifacts (`results/`, checkpoints, `train_log.jsonl`, `pipeline.log`, `projection.pt`,
`schedule.npy`, the main log) are copied to `PERSIST_DIR` (default `/workspace/entd2`), also on failure.

To stop the pod automatically when the script exits (success or failure):

```bash
export RUNPOD_API_KEY=...        # from the RunPod console, Settings -> API Keys
STOP_POD=1 LOG=/root/mvp.log nohup bash scripts/run_mvp.sh > /root/mvp.log 2>&1 &
```

`run_mvp.sh` caps `OMP_NUM_THREADS` at 8: pods expose all host cores (128) under a ~16-vCPU quota, and torch's
default 128 intra-op threads make small CPU ops take hundreds of ms.

Stopping wipes the container disk (the stores), so only what was persisted to the network volume survives;
delete the pod from the console afterwards. The stores can be regenerated from the persisted checkpoints.

Stages can be re-run individually, e.g. `python -m entd.eval configs/tinystories_mvp.yaml --stages rerank,report`.
Everything under `data/` and `runs/` is regenerable; only `runs/tinystories_mvp/results/` needs persisting.

## Experiments

| version | config | model | corpus / coverage | store | cost |
|---|---|---|---|---|---|
| tier 0 | `configs/shakespeare_char.yaml` | 10M char-GPT | 1M chars, 100 % | 260 MB exact (V=65) | laptop, < 1 h |
| tier 1 (MVP) | `configs/tinystories_mvp.yaml` | ~50M GPT (25M non-emb) | TinyStories 0.47B tokens, 100 % | 121 GB fp16 128-d + 47 GB h-subset | 1×A100/H100, $15–40 |
| Large A | `configs/fineweb_large.yaml` | GPT-2 medium 354M, self-trained | FineWeb-Edu 5B tokens, 100 % | 650 GB int8 | $150–250 |
| Large B | `configs/pythia_large.yaml` | Pythia 160M / 410M / 1.4B | Pile 5B-token slice, ~1.7 % | 3 × 650 GB | $120–220 |

Scripts: `scripts/run_tier0.sh`, `run_mvp.sh`, `run_gate.sh` (the $5 Pythia coverage test that decides A vs B),
`run_large_a.sh`, `run_large_b.sh`. The Large configs and the FineWeb-Edu / Pythia data paths are written but
have not been run.

## Reading the results

`results/report/summary.md` and the plots:

- `divergence_vs_k.png` – JS between the neighbor histogram `e_k` and the model's prediction `p_q` vs k, held-out
  and in-train, with the *naive* null (histogram of k draws from `p_q`). Projected space, full store.
- `rerank_decomposition.png` – exact re-rank: `d(e_k, p_q)` split into geometry `d(p̄_k, p_q)` (how far the
  neighbors' own predictions are from the query's) and calibration `d(e_k, p̄_k)` (do neighbor outcomes follow the
  neighbors' predictions), with the proper null `y_i* ~ p_i`; the global z of
  `T_k = mean[−log p_q(y_i)] − mean[H(p_i) + KL(p_i‖p_q)]`, which is 0 under local calibration.
- `memorization_lambda.png` – NLL of the true next token under the model vs under the neighbor histogram
  (train vs held-out), and the kNN-LM-style λ curve.
- `coverage.png` – store variants: full, untrained (never trained on), 10 % / 1 % / 0.1 % coverage.
- `baselines.png`, `radius.png`, `strata.png`, `exact_vs_projected.png`.

Caveats that the analysis is built around: neighbors are selected by the model's own prediction, so the right
null is per-neighbor (`y_i ~ p_i`), not `y_i ~ p_q`; the trained store's outcomes were fit on, so the untrained
store is the clean calibration estimate; duplicated text makes in-train self-leakage common, so neighbors sharing
a ≥ L-token suffix with the query are flagged and counted; `p = softmax(W_U h)` is a bijection on the model's
output manifold, so distribution-kNN and hidden-state-kNN carry the same information under different metrics.
