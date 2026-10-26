#!/usr/bin/env bash
# Tier 1 (TinyStories) on one 80GB GPU pod: train 2 epochs, then embed + the full eval suite for BOTH the
# epoch-1 checkpoint (configs/tinystories_mvp_ep1.yaml) and the final 2-epoch model (configs/tinystories_mvp.yaml).
#
# PARALLEL=1 (default): the epoch-1 pipeline starts as soon as ckpt_epoch1.pt appears (overlapping the second
#   training epoch) and the final-model pipeline starts when training ends; the two share the GPU. Each pipeline
#   logs to runs/<name>/pipeline.log. GPU memory: training ~20 GB, embed ~15 GB, search ~20 GB per process at
#   eval.chunk 131072, so everything fits on 80 GB. PARALLEL=0 runs train, then each pipeline in turn.
#
#   Pod: A100/H100 80GB. Run from the repo on the pod's LOCAL disk (container disk, >= 450 GB: two 170 GB
#        stores + data), NOT on an NFS-backed /workspace volume (~100 MB/s writes). A small network volume at
#        /workspace keeps the persisted artifacts (see PERSIST_DIR) across pod stop/delete.
#   Env: MAIN (training config), CFGS (configs to embed/eval; the one named *_ep1 waits for ckpt_epoch1.pt),
#        PERSIST_DIR (default /workspace/entd2), LOG (path of the main log file to persist), HF_HOME,
#        STOP_POD=1 + RUNPOD_API_KEY (+ RUNPOD_POD_ID, set by RunPod) to stop the pod when everything is done.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
MAIN=${MAIN:-configs/tinystories_mvp.yaml}
CFGS=(${CFGS:-configs/tinystories_mvp.yaml configs/tinystories_mvp_ep1.yaml})
PARALLEL=${PARALLEL:-1}
PERSIST_DIR=${PERSIST_DIR:-/workspace/entd2}
LOG=${LOG:-mvp.log}
export HF_HOME=${HF_HOME:-/root/hf}

cfg_name() { $PY -c "import sys; from entd.config import load_config; print(load_config(sys.argv[1])['name'])" "$1"; }
MAIN_DIR=runs/$(cfg_name "$MAIN")
EP1_CKPT=$MAIN_DIR/ckpt_epoch1.pt

# Copy the small artifacts (results, checkpoints, logs; never the stores) of every run under runs/ to PERSIST_DIR.
persist() {
  [[ -d "$(dirname "$PERSIST_DIR")" ]] || { echo "[persist] $(dirname "$PERSIST_DIR") missing, skipping"; return 0; }
  for rd in runs/*/; do
    dst="$PERSIST_DIR/$rd"; mkdir -p "$dst"
    for src in "$rd"results "$rd"ckpt.pt "$rd"ckpt_epoch*.pt "$rd"train_log.jsonl "$rd"pipeline.log "$rd"projection.pt \
               "$rd"schedule.npy "$rd"last_seen_step.npy "$rd"split_train.npy; do
      [[ -e "$src" ]] || continue            # (an unmatched glob stays literal and fails this test)
      f=$(basename "$src")
      if [[ -d "$src" ]]; then mkdir -p "$dst/$f" && cp -r "$src/." "$dst/$f/"; else cp "$src" "$dst/"; fi
    done
  done
  [[ -e "$LOG" ]] && cp "$LOG" "$PERSIST_DIR/" || true
  echo "[persist] synced to $PERSIST_DIR at $(date -u +%FT%TZ)"
}
trap persist EXIT   # also on failure, so partial results survive

# embed + full eval for one config, logged to runs/<name>/pipeline.log; persist afterwards.
run_model() {
  local cfg=$1 name; name=$(cfg_name "$cfg"); mkdir -p "runs/$name"
  echo "[run_model] $cfg -> runs/$name/pipeline.log ($(date -u +%FT%TZ))"
  (
    set -euo pipefail
    $PY -m entd.embed "$cfg"                      # ~15 min on local NVMe; 121 GB store + 47 GB h-subset
    $PY -m entd.eval "$cfg" --stages queries,search,proj,radius,fidelity,anchors
    $PY -m entd.eval "$cfg" --stages rerank       # ~20-30 min (1000 queries/set x 1024 candidates)
    $PY -m entd.eval "$cfg" --stages baselines,report
  ) > "runs/$name/pipeline.log" 2>&1
  local rc=$?
  echo "[run_model] $cfg finished rc=$rc ($(date -u +%FT%TZ))"
  persist
  return $rc
}

EP1_CFG=""; MAIN_CFGS=()
for c in "${CFGS[@]}"; do if [[ "$(cfg_name "$c")" == *_ep1 ]]; then EP1_CFG=$c; else MAIN_CFGS+=("$c"); fi; done

if [[ "$PARALLEL" == "1" ]]; then
  $PY -m entd.train "$MAIN" & train_pid=$!     # ~1 h on A100: 2 epochs; writes ckpt_epoch1.pt, ckpt_epoch2.pt, ckpt.pt
  ep1_pid=""
  if [[ -n "$EP1_CFG" ]]; then
    while kill -0 "$train_pid" 2>/dev/null && [[ ! -f "$EP1_CKPT" ]]; do sleep 30; done
    if [[ -f "$EP1_CKPT" ]]; then run_model "$EP1_CFG" & ep1_pid=$!; else echo "[run_mvp] training ended without $EP1_CKPT"; fi
  fi
  wait "$train_pid"          # set -e: a training failure aborts here (the ep1 pipeline, if any, keeps its own log)
  persist
  fail=0; pids=()
  for c in "${MAIN_CFGS[@]}"; do run_model "$c" & pids+=($!); done
  for p in ${ep1_pid:+"$ep1_pid"} "${pids[@]}"; do wait "$p" || fail=1; done
  [[ $fail == 0 ]] || { echo "[run_mvp] a pipeline failed, see runs/*/pipeline.log"; exit 1; }
else
  $PY -m entd.train "$MAIN"
  persist
  for c in "${CFGS[@]}"; do run_model "$c"; done
fi
echo "[run_mvp] all done ($(date -u +%FT%TZ))"

# STOP_POD=1: stop the pod when done. The container disk (stores) is wiped on stop; everything persisted above
# lives on the network volume. Delete the pod from the console afterwards.
if [[ "${STOP_POD:-0}" == "1" && -n "${RUNPOD_POD_ID:-}" ]]; then
  echo "[stop] stopping pod $RUNPOD_POD_ID"
  curl -sf -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop" >/dev/null \
    || { runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null 2>&1 || true; runpodctl stop pod "$RUNPOD_POD_ID"; }
fi
