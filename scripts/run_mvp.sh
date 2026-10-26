#!/usr/bin/env bash
# Tier 1 (TinyStories) on one 80GB GPU pod: train 2 epochs, then embed + the full eval suite for BOTH the
# epoch-1 checkpoint (configs/tinystories_mvp_ep1.yaml) and the final 2-epoch model (configs/tinystories_mvp.yaml).
#   Pod: A100/H100 80GB. Run from the repo on the pod's LOCAL disk (container disk, >= 450 GB: two 170 GB
#        stores + data), NOT on an NFS-backed /workspace volume (~100 MB/s writes). A small network volume at
#        /workspace keeps the persisted artifacts (see PERSIST_DIR) across pod stop/delete.
#   Env: PERSIST_DIR (default /workspace/entd2), LOG (path of the log file to persist), HF_HOME,
#        STOP_POD=1 + RUNPOD_API_KEY (+ RUNPOD_POD_ID, set by RunPod) to stop the pod when everything is done.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
MAIN=configs/tinystories_mvp.yaml
CFGS=(${CFGS:-configs/tinystories_mvp.yaml configs/tinystories_mvp_ep1.yaml})
PERSIST_DIR=${PERSIST_DIR:-/workspace/entd2}
LOG=${LOG:-mvp.log}
export HF_HOME=${HF_HOME:-/root/hf}

# Copy the small artifacts (results, checkpoints, logs; never the stores) of every run under runs/ to PERSIST_DIR.
persist() {
  [[ -d "$(dirname "$PERSIST_DIR")" ]] || { echo "[persist] $(dirname "$PERSIST_DIR") missing, skipping"; return 0; }
  for rd in runs/*/; do
    dst="$PERSIST_DIR/$rd"; mkdir -p "$dst"
    for src in "$rd"results "$rd"ckpt.pt "$rd"ckpt_epoch*.pt "$rd"train_log.jsonl "$rd"projection.pt \
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

$PY -m entd.train $MAIN          # ~1 h on A100: 2 epochs over 427M tokens; writes ckpt_epoch1.pt, ckpt_epoch2.pt, ckpt.pt
persist
for CFG in "${CFGS[@]}"; do
  echo "===== $CFG ====="
  $PY -m entd.embed $CFG                      # ~15 min on local NVMe; 121 GB store + 47 GB h-subset
  $PY -m entd.eval $CFG --stages queries,search,proj,radius,fidelity,anchors
  $PY -m entd.eval $CFG --stages rerank       # ~20-30 min (1000 queries/set x 1024 candidates)
  $PY -m entd.eval $CFG --stages baselines,report
  persist
done

# STOP_POD=1: stop the pod when done. The container disk (stores) is wiped on stop; everything persisted above
# lives on the network volume. Delete the pod from the console afterwards.
if [[ "${STOP_POD:-0}" == "1" && -n "${RUNPOD_POD_ID:-}" ]]; then
  echo "[stop] stopping pod $RUNPOD_POD_ID"
  curl -sf -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop" >/dev/null \
    || { runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null 2>&1 || true; runpodctl stop pod "$RUNPOD_POD_ID"; }
fi
