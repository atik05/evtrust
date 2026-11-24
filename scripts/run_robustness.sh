#!/usr/bin/env bash
# Robustness analyses and the ACN-Sim site-level study (Sections III-B to III-D of the paper and their
# Supplementary tables). Run after the main pipeline:  ./reproduce.sh acn && ./reproduce.sh paloalto
#
#   bash scripts/run_robustness.sh            # smoke test on a small subset, then the full run
#
# 1. installs acnportal 0.3.3 and adacharge (third_party/wheels), cvxpy, pandapower and simbench if they are missing
# 2. checks the included cold-start predictions of the paper (data/manuscript_shards) against their SHA-256 sums
# 3. smoke test on a small subset of both networks; stops if anything fails
# 4. full run: ACN stages on the GPU, then ACN-Sim on the CPU in parallel with the Palo Alto stages
# Outputs go to runs/robustness/full/<network>/ (tables, RESULTS.md, robustness.log). A rerun resumes:
# finished stages, model fits and simulation runs are not repeated.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"
CFG="${EVTRUST_CONFIG:-configs/robustness.yaml}"
OUT="$ROOT/$($PY -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['output_dir'])" "$CFG")"
export PYTHONPATH="$ROOT:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
T="${EVTRUST_THREADS:-8}"     # cap CPU threads; idle OpenMP threads should not spin on a shared machine
export OMP_NUM_THREADS="$T" MKL_NUM_THREADS="$T" OPENBLAS_NUM_THREADS="$T" NUMEXPR_NUM_THREADS="$T" OMP_WAIT_POLICY=PASSIVE
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

$PY -c "import pandas, numpy, sklearn, catboost, torch, yaml" || { say "missing core packages: activate the evtrust environment"; exit 1; }
$PY -c "import tabulate, pytz, requests" 2>/dev/null || $PY -m pip install -q tabulate pytz requests
$PY -c "import acnportal" 2>/dev/null || $PY -m pip install -q --no-deps third_party/wheels/acnportal-0.3.3-py3-none-any.whl
$PY -m robustness.acnportal_compat
$PY -c "import cvxpy" 2>/dev/null || $PY -m pip install -q cvxpy
$PY -c "import adacharge" 2>/dev/null || $PY -m pip install -q --no-deps third_party/wheels/adacharge-0.1.0-py3-none-any.whl
$PY -c "import acnportal, adacharge, cvxpy" || { say "ACN-Sim packages could not be installed"; exit 1; }
$PY -c "import pandapower, simbench" 2>/dev/null || $PY -m pip install -q "pandapower==3.5.6" "simbench==1.6.3" numba
$PY -c "import pandapower, simbench" || { say "power-flow packages (pandapower, simbench) could not be installed"; exit 1; }

say "checking the included cold-start predictions"
( cd data/manuscript_shards && sha256sum -c SHA256SUMS --quiet ) || { say "data/manuscript_shards differ from SHA256SUMS"; exit 1; }

if [[ ! -f "$OUT/smoke/.passed" ]]; then
  say "SMOKE TEST (small subset, both networks)"
  $PY -m robustness.run --config "$CFG" --network acn --smoke && $PY -m robustness.run --config "$CFG" --network paloalto --smoke \
    || { say "SMOKE TEST FAILED: see $OUT/smoke/*/robustness.log"; exit 1; }
  touch "$OUT/smoke/.passed"
fi

say "FULL RUN: ACN"
$PY -m robustness.run --config "$CFG" --network acn --skip acnsim,report || { say "ACN preflight failed: see $OUT/full/acn/robustness.log"; exit 1; }
say "FULL RUN: ACN-Sim (CPU, background) and Palo Alto (GPU)"
mkdir -p "$OUT/full"
$PY -m robustness.run --config "$CFG" --network acn --only acnsim > "$OUT/full/acnsim_console.log" 2>&1 &
SIM=$!
$PY -m robustness.run --config "$CFG" --network paloalto
wait $SIM
$PY -m robustness.run --config "$CFG" --network acn --only report
say "DONE: tables in $OUT/full/acn and $OUT/full/paloalto"
