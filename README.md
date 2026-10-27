# evtrust

Code and reproduction package for the paper

> **Trustworthy electric vehicle charging session prediction: leakage-aware evaluation, the cold-start gap and its cost to smart charging.** Md. Atiqur Rahman, Mohammad Alamgir Hossain, Md. Ibrahim Abdullah, Ahsan-Ul-Ambia and Md. Shahidul Islam. Manuscript under review.

The package evaluates forecasts of EV charging session duration and energy, made at plug-in, under six evaluation protocols (random, chronological, driver-disjoint and site-disjoint among them), decomposes the error gap for unseen drivers on identical sessions, measures how much accuracy depends on information available only after plug-in, and translates forecast error into smart-charging flexibility and site capacity. The analyses run on the three ACN-Data workplace sites and, as an external validation, on the public charging network of the City of Palo Alto.

## Repository layout

```
src/evtrust/       pipeline: data, features, protocols, leakage audit, learners, conformal intervals,
                   cold-start decomposition, information availability, smart-charging impact
scripts/           analysis entry points used by reproduce.sh
configs/           acn.yaml and paloalto.yaml; they differ only in the sites, the data period and the source
figures/           programs that draw the figures of the paper
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
./reproduce.sh figures    # Figures 1 to 5 and Supplementary Figures S1 to S4                  -> figures/build/
```

Each stage records a marker in `runs/<network>/.done/`, so an interrupted run resumes where it stopped. Logs are written to `runs/<network>/logs/`, and the package versions of each run to `runs/<network>/software_versions.txt`. On that workstation the ACN pipeline, including the twelve-learner benchmark, took about 14 hours, and the feature ablation and squared-error refits a few hours more; the Palo Alto pipeline took about 10 hours.

## Where each result comes from

| In the paper | Output | Program |
|---|---|---|
| Every number in the text, Tables 3 to 5, Supplementary Tables S1 to S25 | `runs/acn/results/`: `FULL_REPORT.xlsx` and the CSV files under `metrics/`, `benchmark/`, `paper/` and `operational/` | `reproduce.sh acn` (summary by `scripts/full_report.py`) |
| Table 6, Supplementary Tables S26 to S30 | `runs/paloalto/results/` (same layout) | `reproduce.sh paloalto` |
| Calibration over time, Supplementary Table S31 | `runs/<network>/results/paper/calibration_robustness.csv` | `scripts/calibration_robustness.py` |
| Figure 1, evaluation framework | `figures/build/fig1_framework.svg` (and `.pdf`, `.png` with cairosvg) | `figures/make_fig1.py` |
| Figures 2 to 5 | `figures/build/fig2_protocols` to `fig5_operation` (`.pdf`, `.svg`, `.png`) | `figures/make_figures.py` |
| Supplementary Figures S1 to S4 | `figures/build/si_cd_energy.pdf`, `si_cd_duration.pdf`, `si_ft.pdf`, `si_site.pdf` | `figures/make_si_figures.py` |

Tables 1 and 2 (literature and protocol definitions) contain no computed values.

## Randomness

Seeds are fixed (11, 23, 37, 51 and 79) and splits are frozen. Training on a GPU is not bit-for-bit deterministic, so a reproduced value can differ slightly from the published one.

## Tests

```bash
python -m pytest -q        # pytest comes with environment.yml, or with pip install -e ".[dev]"
```

## Licence and citation

The code is released under the MIT licence. The data remain subject to the terms of their providers. Please cite the paper; citation details are in `CITATION.cff`.
