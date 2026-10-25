#!/usr/bin/env bash
# Tier 0: laptop smoke test end to end (char Shakespeare, exact distributions).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
CFG=configs/shakespeare_char.yaml
$PY -m entd.train $CFG
$PY -m entd.embed $CFG
$PY -m entd.eval $CFG
echo "report: runs/shakespeare_char/results/report/summary.md"
