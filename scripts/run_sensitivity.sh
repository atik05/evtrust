#!/usr/bin/env bash
# Sensitivity analyses of the site-level results (Supplementary Tables of the five forecast seeds, the two tariffs,
# time-series hosting and the tuning budget). Run after ./reproduce.sh acn and ./reproduce.sh robustness.
#
#   bash scripts/run_sensitivity.sh          # or: ./reproduce.sh sensitivity
#
#   1. CatBoost deploy forecasts of the five seeds (GPU), then for every seed ACN-Sim (E1, E2 and the E5 morning event
#      on fifteen limits), the information-state replay and both feeders; seed 11 uses the published forecasts
#   2. tuning budget: the eight-trial search of the main study extended to 40 trials, in the background (GPU)
#   3. E3 (adaptive scheduling algorithm) on every week under SCE TOU-EV-4 and TOU-EV-8, five seeds
#   4. time-series hosting: every site simulated at each forecast's own capacity (E6), then the SimBench feeder year
#   5. summary tables in runs/sensitivity/results, laid out as results/ (scripts/summarise_sensitivity.py)
# Every step resumes: finished stages, ACN-Sim runs, tuning searches and refits are skipped on a rerun.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"
export PYTHONPATH="$ROOT:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
OUT="$ROOT/runs/sensitivity"
PUB="$ROOT/runs/robustness/full/acn"                      # the robustness run of ./reproduce.sh robustness
SESS="$ROOT/runs/acn/data/processed/sessions.parquet"
PRED="$ROOT/runs/acn/results/predictions/regression_predictions.parquet"
HP=$(find "$ROOT/runs/acn" -name tuned_hyperparameters.json 2>/dev/null | head -1 || true)
CFG=configs/robustness_seeds.yaml
SEEDS="11 23 37 51 79"
TARIFFS="sce_tou_ev_4_march_2019 sce_tou_ev_8_june_2019"
say() { printf '\n== %s  (%s)\n' "$*" "$(date '+%Y-%m-%d %H:%M')"; }
unset EVTRUST_ACNSIM_INCLUDE EVTRUST_ACNSIM_EXCLUDE EVTRUST_ACNSIM_NO_SUMMARY

for f in "$PUB/deploy_forecasts_catboost_seed11.parquet" "$SESS" "$PRED"; do
  [ -f "$f" ] || { echo "missing: $f (run ./reproduce.sh acn and ./reproduce.sh robustness first)"; exit 1; }
done
$PY -c "import acnportal, adacharge, cvxpy, pandapower, simbench, optuna" || { echo "run scripts/run_robustness.sh first (it installs the simulation packages); pip install optuna"; exit 1; }
mkdir -p "$OUT"

percfg() {  # percfg <template> <output_dir> <seed> <python statements on c>
  $PY - "$1" "$2" "$3" "$4" <<'EOF'
import sys, yaml
src, out, seed, extra = sys.argv[1:5]
c = yaml.safe_load(open(src))
c['output_dir'] = out; c['acnsim']['seed'] = int(seed)
exec(extra)
yaml.safe_dump(c, open(f'{out}/config.yaml', 'w'), sort_keys=False)
EOF
}

say "1. five forecast seeds: deploy forecasts, ACN-Sim, information-state replay and feeders"
$PY -m robustness.run --config "$CFG" --network acn --only anonymous
for s in $SEEDS; do
  D="$OUT/seed$s/full/acn"; mkdir -p "$D/.done" "$D/acnsim/runs"
  if [ "$s" = 11 ]; then
    cp "$PUB/deploy_forecasts_catboost_seed11.parquet" "$D/"
    [ -d "$PUB/acnsim/runs" ] && for f in "$PUB/acnsim/runs/"*; do [ -f "$D/acnsim/runs/$(basename "$f")" ] || cp "$f" "$D/acnsim/runs/"; done
  else
    cp "$OUT/base/full/acn/deploy_forecasts_catboost_seed$s.parquet" "$D/"
  fi
  touch "$D/.done/anonymous"
  percfg "$CFG" "$OUT/seed$s" "$s" "c['networks']['acn']['stages'] = ['acnsim', 'infostate', 'infostate_sim', 'feeder']"
  say "   seed $s"
  EVTRUST_ACNSIM_EXCLUDE='__E3__|__E4__|__edf__|__request__|__E5__.*__tau0\.1$' \
    $PY -m robustness.run --config "$OUT/seed$s/config.yaml" --network acn --only acnsim,infostate,infostate_sim,feeder
done
$PY -m robustness.seeds_summary --root "$OUT" --seeds $SEEDS --out "$OUT/summary"

say "2. tuning budget (background; logs in $OUT/tuning_*.log)"
tune() {
  exec 9>"$OUT/tuning_$1.lock"; flock 9
  if EVTRUST_CATBOOST_GPU_RAM_PART=0.3 $PY scripts/tuning_budget.py --config configs/acn.yaml --sessions "$SESS" --published "$PRED" \
      ${HP:+--published-params "$HP"} --learners "$1" --trials 40 --check 8 --out "$OUT/tuning/$1" >> "$OUT/tuning_$1.log" 2>&1
  then echo 0 > "$OUT/tuning_$1.exit"; else echo 1 > "$OUT/tuning_$1.exit"; fi
}
rm -f "$OUT"/tuning_*.exit
tune catboost & P1=$!
tune torch_ft & P2=$!

say "3. E3 under TOU-EV-4 and TOU-EV-8, every week, five seeds"
for t in $TARIFFS; do
  for s in $SEEDS; do
    D="$OUT/e3_$t/seed$s/full/acn"; mkdir -p "$D/.done" "$D/acnsim/runs"
    [ -f "$D/deploy_forecasts_catboost_seed$s.parquet" ] || cp "$OUT/seed$s/full/acn/deploy_forecasts_catboost_seed$s.parquet" "$D/"
    touch "$D/.done/anonymous"
    percfg "$OUT/seed$s/config.yaml" "$OUT/e3_$t/seed$s" "$s" \
      "c['acnsim']['tariff'] = '$t'; c['acnsim']['asa_weeks'] = 'all'; c['networks']['acn']['stages'] = ['acnsim', 'e3_weeks']"
    say "   $t, seed $s"
    EVTRUST_ACNSIM_INCLUDE='__E3__' $PY -m robustness.run --config "$OUT/e3_$t/seed$s/config.yaml" --network acn --only acnsim,e3_weeks
  done
done

say "4. time-series hosting, five seeds"
for s in $SEEDS; do
  D="$OUT/hosting/seed$s/full/acn"; mkdir -p "$D/.done" "$D/acnsim/runs"
  [ -f "$D/deploy_forecasts_catboost_seed$s.parquet" ] || cp "$OUT/seed$s/full/acn/deploy_forecasts_catboost_seed$s.parquet" "$D/"
  touch "$D/.done/anonymous"
  percfg "$OUT/seed$s/config.yaml" "$OUT/hosting/seed$s" "$s" \
    "c['acnsim']['e6_capacity_needed'] = '$OUT/seed$s/full/acn/acnsim_capacity_needed.csv'; c.setdefault('feeder', {})['ts_orders'] = 10; c['networks']['acn']['stages'] = ['acnsim', 'feeder_ts']"
  say "   seed $s"
  EVTRUST_ACNSIM_INCLUDE='__E6__' EVTRUST_ACNSIM_NO_SUMMARY=1 $PY -m robustness.run --config "$OUT/hosting/seed$s/config.yaml" --network acn --only acnsim
  $PY -m robustness.run --config "$OUT/hosting/seed$s/config.yaml" --network acn --only feeder_ts
done

say "5. tuning budget: waiting, then merging"
wait $P1 || true; wait $P2 || true
for l in catboost torch_ft; do
  [ "$(cat "$OUT/tuning_$l.exit" 2>/dev/null || echo 1)" = 0 ] || { echo "tuning of $l failed: see $OUT/tuning_$l.log; rerun to resume"; exit 1; }
done
$PY scripts/tuning_budget.py --merge "$OUT/tuning/catboost" "$OUT/tuning/torch_ft" --out "$OUT/tuning/all" \
    --published "$PRED" --trials 40 --check 8 --sessions "$SESS"

say "6. summary tables (runs/sensitivity/results, same layout as results/)"
R="$OUT/results"; A="$R/robustness/acn"; mkdir -p "$R/tuning" "$A/seeds" "$A/tariffs" "$A/hosting_timeseries"
cp "$OUT"/tuning/all/*.csv "$R/tuning/"
for l in catboost torch_ft; do cp "$OUT/tuning/$l/tuning_budget_settings.json" "$R/tuning/tuning_budget_settings_$l.json"; done
cp "$OUT/summary/seeds_summary.csv" "$OUT/summary/seeds_summary.md" "$A/seeds/"
for s in $SEEDS; do
  mkdir -p "$A/seeds/seed$s" "$A/hosting_timeseries/seed$s"
  cp "$OUT/seed$s/full/acn/"*.csv "$A/seeds/seed$s/"
  cp "$OUT/hosting/seed$s/full/acn/"feeder_ts_* "$A/hosting_timeseries/seed$s/"
  for t in $TARIFFS; do
    mkdir -p "$A/tariffs/$t/seed$s"; cp "$OUT/e3_$t/seed$s/full/acn/"acnsim_runs.csv "$OUT/e3_$t/seed$s/full/acn/"acnsim_e3_*.csv "$A/tariffs/$t/seed$s/"
  done
done
$PY scripts/summarise_sensitivity.py --results "$R"
$PY scripts/verify_results.py --results "$R" || echo "some values differ from the paper (training on a GPU is not bit-for-bit deterministic): see above"
say "done: tables in $R (the published ones are in results/)"
