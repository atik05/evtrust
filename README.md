# evtrust

Code and reproduction package for the paper

> **The first-visit problem in electric vehicle smart charging: driver history, shortfall risk, and dependable grid flexibility.** Md. Atiqur Rahman, Mohammad Alamgir Hossain, Md Al Amin, Md. Ibrahim Abdullah, Ahsan-Ul-Ambia and Md. Shahidul Islam. Manuscript submitted.

The package evaluates forecasts of EV charging session duration and energy, made at plug-in, under six evaluation protocols (random, chronological, driver-disjoint and site-disjoint among them), decomposes the error gap for unseen drivers on identical sessions, measures how much accuracy depends on information available only after plug-in, and translates forecast error into smart-charging flexibility and site capacity. The analyses run on the three ACN-Data workplace sites and, as an external validation, on the public charging network of the City of Palo Alto.

## Repository layout

```
src/evtrust/       pipeline: data, features, protocols, leakage audit, learners, conformal intervals,
                   cold-start decomposition, information availability, smart-charging impact
robustness/        robustness analyses, the information-state rule, the ACN-Sim site-level study and the feeder study (see below)
scripts/           analysis entry points used by reproduce.sh
configs/           acn.yaml and paloalto.yaml; they differ only in the sites, the data period and the source
figures/           programs that draw the figures of the paper
data/manuscript_shards/  the cold-start predictions of the paper (ACN), input to the robustness analyses
results/robustness/      result tables of the robustness analyses and the ACN-Sim study, as reported
third_party/wheels/      acnportal 0.3.3 and adacharge 0.1.0 (BSD licence, Caltech), used by ACN-Sim
tests/             unit tests
reproduce.sh       single entry point
```

## Requirements

Linux with an NVIDIA GPU. The published results were produced on one workstation: Intel Core Ultra 9 285K (24 cores), 64 GB of DDR5 memory, NVIDIA GeForce RTX 4090 (a 48 GB memory variant), Ubuntu 26.04, CUDA 12.4 and Python 3.12.

```bash
conda env create -f environment.yml      # all dependencies, including pytest
conda activate evtrust
pip install -r requirements-lock.txt     # optional: the exact versions of the published run
```

Without conda, `pip install -e ".[dev,torch,paper]"` installs the package with the test, deep-learning and figure dependencies; the figure export also needs the Cairo library of the system (for example `apt install libcairo2`). `requirements-lock.txt` lists the exact versions of the published run; every run records its own in `runs/<network>/requirements-lock.txt`, written by `scripts/make_lock.py` from `pip freeze`.

## Robustness analyses and the site-level simulation

`robustness/` holds the analyses of Sections III-B to III-E of the paper that build on the cold-start predictions:

- order of the decomposition (two-order shares), driver-identity encoding and training size;
- optimism ladder up to a literature-style random split, and learned cold-start priors;
- first recorded visits and anonymous sessions, censored energy need, charging rate and laxity forecasting, and a conservative flexibility metric that credits deferral only up to the oracle's (`robustness/flexsafe.py`);
- reliability of the conformal shifts (Proposition 1), protocol errors by period, rolling-origin evaluation;
- information-state calibration rule (Proposition 2; `robustness/infostate.py`), on both networks;
- ACN-Sim study (acnportal 0.3.3) on the Caltech and JPL network models: real capacity (E1), transformer capacity sweep (E2), adaptive scheduling under a time-of-use tariff (E3), weekday events (E4, E5) and the information-state rule at real capacity;
- feeder hosting capacity on the IEEE 33-bus and SimBench commercial feeders (pandapower; `robustness/feeder.py`), and its time-series version on the SimBench feeder, in which hubs draw the load simulated for the site at each forecast's own capacity (E6) and the feeder is checked at every 15-minute step of the year, with coincident hubs and with hubs whose sessions do not coincide (`robustness/feeder_ts.py`, with the Newton-Raphson power flow of `robustness/fastpf.py`, checked against pandapower);
- E3 by week under a chosen tariff (`robustness/tariffs.py`; TOU-EV-4 and TOU-EV-8 as distributed with acnportal), and the summary of the site-level results over the five forecast seeds (`robustness/seeds_summary.py`);
- sensitivity of the protocol results to the tuning budget (`scripts/tuning_budget.py`: the eight-trial search of the main study extended to 40 trials, both selections refitted on all five seeds).

The five-seed, tariff, time-series and tuning-budget analyses run together with `./reproduce.sh sensitivity` (`scripts/run_sensitivity.sh`, settings in `configs/robustness_seeds.yaml`), after the robustness stage. Their published tables are in `results/robustness/acn/seeds/`, `tariffs/`, `hosting_timeseries/` and `results/tuning/`; `scripts/summarise_sensitivity.py` writes the summary tables from the per-seed files, and `./reproduce.sh verify` (`scripts/verify_results.py`) recomputes every sensitivity value quoted in the paper from the per-seed files and compares it with the printed value.

```bash
pip install -e ".[sim]"                 # cvxpy, pandapower, simbench; acnportal and adacharge come from third_party/wheels
./reproduce.sh robustness               # or: bash scripts/run_robustness.sh
```

The runner checks the included predictions of the paper (`data/manuscript_shards`, SHA-256 sums), runs a smoke test on a small subset of both networks, then the full analyses; a rerun resumes. Settings are in `configs/robustness.yaml`. Every table is written to `runs/robustness/full/<network>/` with a summary in `RESULTS.md`; the tables of the published run are in `results/robustness/`, and `figures/make_figures.py` draws the site-level panels of Fig. 4 from them.

## Data

The data are public but are not redistributed here.

| Network | Source | Access |
|---|---|---|
| ACN-Data (Caltech, JPL, Office001), 2018 to 2021 | https://ev.caltech.edu/dataset | Free registration gives an API token; set `ACN_API_TOKEN` |
| City of Palo Alto, July 2011 to December 2020 | https://data.paloalto.gov/datasets/194693/electric-vehicle-charging-station-usage-july-2011-dec-2020/ | Download `ChargePoint Data CY20Q4.csv` (the full file listed under *Data collected from*, not the 10,000-row export); MD5 `15360a7df4c0b4f0b89ea7ec03a54e20`; set `PALOALTO_CSV` to its path |

The preparation step records the MD5 of the Palo Alto file and whether it matches the verified copy.

## Reproducing the paper

```bash
export ACN_API_TOKEN=...                 # ACN-Data token
export PALOALTO_CSV=/path/to/ChargePoint\ Data\ CY20Q4.csv
./reproduce.sh all
```

or stage by stage:

```bash
./reproduce.sh acn        # ACN-Data: download, preparation, leakage audit and all analyses   -> runs/acn/
./reproduce.sh paloalto   # Palo Alto external validation                                      -> runs/paloalto/
./reproduce.sh robustness # robustness analyses, ACN-Sim and feeder study (after acn and paloalto) -> runs/robustness/
./reproduce.sh sensitivity # five forecast seeds, two tariffs, time-series hosting, tuning budget  -> runs/sensitivity/
./reproduce.sh verify     # recompute the sensitivity values of the paper from results/
./reproduce.sh figures    # Figures 1 to 4 and Supplementary Figures S1 to S4                  -> figures/build/
```

Each stage records a marker in `runs/<network>/.done/`, so an interrupted run resumes where it stopped. Logs are written to `runs/<network>/logs/`, and the package versions of each run to `runs/<network>/software_versions.txt`. On that workstation the ACN pipeline, including the twelve-learner benchmark, took about 14 hours, and the feature ablation and squared-error refits a few hours more; the Palo Alto pipeline took about 10 hours.

## Where each result comes from

| In the paper | Output | Program |
|---|---|---|
| Main-pipeline numbers in the text; Supplementary Tables S1 to S12, S14, S15, S18 to S20, S23, S25 to S27, S29, S49 to S54 | `runs/acn/results/`: `FULL_REPORT.xlsx` and the CSV files under `metrics/`, `benchmark/`, `paper/` and `operational/` | `reproduce.sh acn` (summary by `scripts/full_report.py`) |
| Table II (ACN and Palo Alto side by side) | `runs/acn/results/`, `runs/paloalto/results/` and `runs/robustness/full/` | `reproduce.sh acn`, `paloalto` and `robustness` |
| Supplementary Tables S44 to S48 | `runs/paloalto/results/` (same layout) | `reproduce.sh paloalto` |
| Calibration over time, Supplementary Table S33 | `runs/<network>/results/paper/calibration_robustness.csv` | `scripts/calibration_robustness.py` |
| Robustness analyses: Supplementary Tables S13, S16, S17, S21, S22, S24, S28, S30 to S32 | `runs/robustness/full/<network>/` (published run: `results/robustness/`) | `reproduce.sh robustness` |
| ACN-Sim: Table III (capacity), Supplementary Tables S37, S39 and S40 | `runs/robustness/full/acn/acnsim_*.csv` (published run: `results/robustness/acn/`) | `reproduce.sh robustness` |
| Information-state calibration: Table II (last block), Supplementary Tables S34 to S36 | `runs/robustness/full/<network>/infostate_*.csv` (published run: `results/robustness/`) | `reproduce.sh robustness` |
| Feeder hosting capacity: Table III (hubs hosted), Supplementary Table S42 | `runs/robustness/full/acn/feeder_*.csv` (published run: `results/robustness/acn/`) | `reproduce.sh robustness` |
| Five forecast seeds and the E5 firm limits (fifteen event limits): Table III (firm event limit), Supplementary Table S41 | `runs/sensitivity/seed*/full/acn/` (published run: `results/robustness/acn/seeds/`, Table III from `seed11/`) | `reproduce.sh sensitivity` (`robustness/seeds_summary.py`) |
| E3 under TOU-EV-4 and TOU-EV-8, every week: Supplementary Table S38 | `runs/sensitivity/e3_*/` (published run: `results/robustness/acn/tariffs/`) | `reproduce.sh sensitivity` (`robustness/tariffs.py`, `scripts/summarise_sensitivity.py`) |
| Time-series hosting capacity: Supplementary Table S43 | `runs/sensitivity/hosting/` (published run: `results/robustness/acn/hosting_timeseries/`) | `reproduce.sh sensitivity` (`robustness/feeder_ts.py`, `scripts/summarise_sensitivity.py`) |
| Tuning budget: Supplementary Table S55 | `runs/sensitivity/tuning/all/` (published run: `results/tuning/`) | `reproduce.sh sensitivity` (`scripts/tuning_budget.py`) |
| Figure 1, evaluation framework | `figures/build/fig1_framework.svg` (and `.pdf`, `.png` with cairosvg) | `figures/make_fig1.py` |
| Figure 2, evaluation protocols | `figures/build/fig2_protocols` (`.pdf`, `.svg`, `.png`) | `figures/make_figures.py` |
| Figure 3, cold-start decomposition and warm-up | `figures/build/fig3_coldstart` | `figures/make_figures.py` |
| Figure 4, per-session and site-level smart charging | `figures/build/fig4_operation` | `figures/make_figures.py` |
| Supplementary Figure S3, information availability | `figures/build/si_fig_availability` | `figures/make_figures.py` |
| Supplementary Figures S1, S2 and S4 | `figures/build/si_cd_energy.pdf`, `si_cd_duration.pdf`, `si_ft.pdf` | `figures/make_si_figures.py` |

Supplementary Tables S4 and S56 (protocol definitions and literature) contain no computed values.

## Randomness

Seeds are fixed (11, 23, 37, 51 and 79) and splits are frozen. Training on a GPU is not bit-for-bit deterministic, so a reproduced value can differ slightly from the published one.

## Tests

```bash
python -m pytest -q        # pytest comes with environment.yml, or with pip install -e ".[dev]"
```

## Licence and citation

The code is released under the MIT licence. The data remain subject to the terms of their providers. Please cite the paper; citation details are in `CITATION.cff`.
