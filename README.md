# evtrust

Code and reproduction package for the paper

> **Trustworthy electric vehicle charging session prediction: leakage-aware evaluation, the cold-start gap and its cost to smart charging.** Md. Atiqur Rahman. Manuscript under review.

The package evaluates forecasts of EV charging session duration and energy, made at plug-in, under six evaluation protocols (random, chronological, driver-disjoint and site-disjoint among them), decomposes the error gap for unseen drivers on identical sessions, measures how much accuracy depends on information available only after plug-in, and translates forecast error into smart-charging flexibility and site capacity. The analyses run on the three ACN-Data workplace sites and, as an external validation, on the public charging network of the City of Palo Alto.

## Repository layout

```
src/evtrust/       pipeline: data, features, protocols, leakage audit, learners, conformal intervals,
                   cold-start decomposition, information availability, smart-charging impact
scripts/           analysis entry points used by reproduce.sh, and the reproduction check
configs/           acn.yaml and paloalto.yaml; they differ only in the sites, the data period and the source
paper/             generators of every number, table and figure in the paper, and the published values (expected/)
tests/             unit tests
reproduce.sh       single entry point
```

## Requirements

Linux with an NVIDIA GPU; the published results were produced on one NVIDIA GeForce RTX 4090 (a 48 GB memory variant) with CUDA 12.4 and Python 3.12.

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
./reproduce.sh paper      # numbers, tables and figures                                        -> paper/build/
./reproduce.sh check      # compare paper/build/ with the published values in paper/expected/
```

Each stage records a marker in `runs/<network>/.done/`, so an interrupted run resumes where it stopped. Logs are written to `runs/<network>/logs/`, and the package versions of each run to `runs/<network>/software_versions.txt`. The ACN stages, which include the twelve-learner benchmark, take the longest, of the order of a day on one RTX 4090; the Palo Alto stages take less.

## List of tables, figures and programs

| In the paper | File in `paper/build/` | Generator |
|---|---|---|
| Every number in the text | `resultsnumbers.tex` (ACN), `externalnumbers.tex` (Palo Alto) | `make_results_tex.py`, `make_external_tex.py` |
| Table 3, accuracy by protocol | `resultstableprotocols.tex` | `make_results_tex.py` |
| Table 4, learner benchmark | `resultstablebenchmark.tex` | `make_results_tex.py` |
| Table 5, cold-start decomposition | `resultstabledecomp.tex` | `make_results_tex.py` |
| Table 6, ACN against Palo Alto | `externaltable.tex` | `make_external_tex.py` |
| Graphical abstract | `overview.pdf`, `.svg`, `.png` | `make_overview.py` |
| Figure 1 | `fig1_framework.svg`, `.pdf`, `.png` | `make_fig1.py` |
| Figures 2 to 5 | `figs/fig2_protocols` to `figs/fig5_operation` (`.pdf`, `.svg`, `.png`) | `make_figures.py` |
| Supplementary Tables S1 to S25, Figures S1 to S4 | `si_body.tex`, `figs/si_*.pdf` | `make_si.py` |
| Supplementary Tables S26 to S30 | `si_external.tex` | `make_external_tex.py` |
| Smart charging with calibration over time: the sentence in Section 3.4 and Supplementary Table S31 | `calibrationnumbers.tex`, `si_calibration.tex` | `make_calibration_tex.py`, from the output of `scripts/calibration_robustness.py` |

Tables 1 and 2 (literature and protocol definitions) contain no computed values.

## Verifying a reproduction

`./reproduce.sh check` compares every regenerated value with its published value in `paper/expected/` and every generated table line by line, and ends with `REPRODUCED` or a list of the values that differ. Seeds are fixed (11, 23, 37, 51 and 79), splits are frozen and every result is computed from the pipeline outputs. Training on a GPU is not bit-for-bit deterministic, however, so a value is accepted when it is identical or differs by at most one unit in its last reported digit, and every such difference is listed.

## Tests

```bash
python -m pytest -q        # pytest comes with environment.yml, or with pip install -e ".[dev]"
```

## Licence and citation

The code is released under the MIT licence. The data remain subject to the terms of their providers. Please cite the paper; citation details are in `CITATION.cff`.
