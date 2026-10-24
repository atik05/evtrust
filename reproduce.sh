#!/usr/bin/env bash
# Reproduce every number, table and figure of the paper.
#
#   ./reproduce.sh acn        ACN-Data: download, prepare, audit, all analyses      (needs ACN_API_TOKEN)
#   ./reproduce.sh paloalto   Palo Alto external validation                          (needs PALOALTO_CSV)
#   ./reproduce.sh paper      numbers, tables and figures into paper/build/
#   ./reproduce.sh all        all of the above, in order
#
# Every stage writes a marker in runs/<network>/.done/, so an interrupted run resumes where it stopped.
# Delete runs/<network>/.done/<stage> to repeat one stage.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PY="${PYTHON:-python}"

stage() {  # stage <network> <name> <command...>
  local net="$1" name="$2"; shift 2
  local dir="$ROOT/runs/$net"; mkdir -p "$dir/.done" "$dir/logs"
  if [[ -f "$dir/.done/$name" ]]; then echo "[$net] $name: done"; return; fi
  echo "[$net] $name ..."
  ( cd "$dir" && "$@" ) 2>&1 | tee -a "$dir/logs/$name.log"
  touch "$dir/.done/$name"
}

cli() { "$PY" -m evtrust.cli "$@"; }

run_acn() {
  local C="$ROOT/configs/acn.yaml"
  [[ -n "${ACN_API_TOKEN:-}" || -f "$ROOT/runs/acn/.done/download" ]] || { echo "Set ACN_API_TOKEN (https://ev.caltech.edu/dataset)"; exit 1; }
  stage acn environment   "$PY" "$ROOT/scripts/check_environment.py"
  stage acn versions      bash -c "$PY -m pip freeze > software_versions.txt && $PY $ROOT/scripts/make_lock.py software_versions.txt > requirements-lock.txt"
  stage acn download      cli download         --config "$C"
  stage acn prepare       cli prepare          --config "$C"
  stage acn audit         cli audit            --config "$C"
  stage acn freeze        cli freeze-splits    --config "$C"
  stage acn regression    cli run-regression   --config "$C"
  stage acn ablation      cli run-ablation     --config "$C"
  stage acn occ_panel     cli build-occupancy  --config "$C"
  stage acn occupancy     cli run-occupancy    --config "$C"
  stage acn analyze       cli analyze          --config "$C"
  stage acn validate      "$PY" "$ROOT/scripts/validate_outputs.py"
  stage acn remedies      "$PY" "$ROOT/scripts/run_contributions.py"  --config "$C"
  stage acn paper         "$PY" "$ROOT/scripts/run_paper_analyses.py" --config "$C"
  stage acn calibration "$PY" "$ROOT/scripts/calibration_robustness.py" --results results --sessions data/processed/sessions.parquet --network ACN --out results/paper/calibration_robustness.csv
  stage acn benchmark     "$PY" "$ROOT/scripts/run_benchmark.py"      --config "$C"
  stage acn squared_error "$PY" "$ROOT/scripts/run_squared_error.py"  --config "$C"
  stage acn operational   "$PY" "$ROOT/scripts/run_operational.py"    --config "$C"
  stage acn ablation_seed "$PY" "$ROOT/scripts/run_ablation_seeds.py" --config "$C"
  stage acn report        "$PY" "$ROOT/scripts/full_report.py"
}

run_paloalto() {
  local C="$ROOT/configs/paloalto.yaml"
  [[ -n "${PALOALTO_CSV:-}" || -f "$ROOT/runs/paloalto/.done/prepare" ]] || { echo "Set PALOALTO_CSV to the downloaded ChargePoint Data CY20Q4.csv"; exit 1; }
  stage paloalto versions   bash -c "$PY -m pip freeze > software_versions.txt && $PY $ROOT/scripts/make_lock.py software_versions.txt > requirements-lock.txt"
  stage paloalto prepare    "$PY" "$ROOT/scripts/prepare_paloalto.py" --csv "${PALOALTO_CSV:-}" --config "$C"
  stage paloalto audit      cli audit          --config "$C"
  stage paloalto freeze     cli freeze-splits  --config "$C"
  stage paloalto regression cli run-regression --config "$C"
  stage paloalto analyze    cli analyze        --config "$C"
  stage paloalto paper      "$PY" "$ROOT/scripts/run_paper_analyses.py" --config "$C"
  stage paloalto calibration "$PY" "$ROOT/scripts/calibration_robustness.py" --results results --sessions data/processed/sessions.parquet --network PaloAlto --out results/paper/calibration_robustness.csv
  stage paloalto report     "$PY" "$ROOT/scripts/full_report.py"
}

run_paper() {
  local A="$ROOT/runs/acn/results" P="$ROOT/runs/paloalto/results/FULL_REPORT.xlsx" B="$ROOT/paper/build"
  for f in "$A/FULL_REPORT.xlsx" "$P"; do [[ -f "$f" ]] || { echo "missing $f: run the network stages first"; exit 1; }; done
  mkdir -p "$B/figs" && cd "$B"
  "$PY" ../make_results_tex.py "$A/FULL_REPORT.xlsx" "$A/operational"
  "$PY" ../make_external_tex.py "$P"
  "$PY" ../make_figures.py "$A" "$P" figs
  "$PY" ../make_si.py "$A" "$P"
  "$PY" ../make_calibration_tex.py "$A/paper/calibration_robustness.csv" "$ROOT/runs/paloalto/results/paper/calibration_robustness.csv"
  "$PY" ../make_fig1.py
  "$PY" ../make_overview.py .
  echo "paper outputs written to $B"
}

case "${1:-}" in
  acn)      run_acn ;;
  paloalto) run_paloalto ;;
  paper)    run_paper ;;
  all)      run_acn; run_paloalto; run_paper ;;
  *)        sed -n '2,11p' "$0"; exit 1 ;;
esac
