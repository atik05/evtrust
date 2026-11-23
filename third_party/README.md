# Third-party packages

| Wheel | Source | Licence |
|---|---|---|
| `acnportal-0.3.3-py3-none-any.whl` | https://github.com/zach401/acnportal (Caltech) | BSD 3-Clause (included in the wheel) |
| `adacharge-0.1.0-py3-none-any.whl` | https://github.com/zach401/AdaCharge (Caltech) | BSD (included in the wheel) |

They are installed by `scripts/run_robustness.sh` for the ACN-Sim study. acnportal 0.3.3 imports
`pkg_resources`, which recent setuptools no longer provides; `python -m robustness.acnportal_compat` replaces
that import with `importlib.metadata` in the installed copy (it changes nothing else).
