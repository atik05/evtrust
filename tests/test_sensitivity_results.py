"""The published sensitivity tables: the summary script runs on them and the paper's values are recovered."""
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'
pytestmark = pytest.mark.skipif(not (RES / 'robustness' / 'acn' / 'hosting_timeseries').exists(), reason='published tables not present')


def test_verify_results_matches_paper():
    r = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'verify_results.py'), '--results', str(RES)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'all values agree with the paper' in r.stdout


def test_summary_script_and_consistency(tmp_path):
    dst = tmp_path / 'results'
    shutil.copytree(RES / 'tuning', dst / 'tuning')
    for d in ('seeds', 'tariffs', 'hosting_timeseries'):
        shutil.copytree(RES / 'robustness' / 'acn' / d, dst / 'robustness' / 'acn' / d)
    r = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'summarise_sensitivity.py'), '--results', str(dst)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    c = pd.read_csv(dst / 'robustness' / 'acn' / 'hosting_timeseries' / 'consistency_check.csv')
    assert len(c) == 100 and c.difference_hubs.abs().max() < 0.2      # time series reproduces the deterministic screen
    s = pd.read_csv(dst / 'robustness' / 'acn' / 'tariffs' / 'e3_tariffs_summary.csv')
    assert (s.seeds == 5).all() and (s.chunk_errors == 0).all()
    pub = pd.read_csv(RES / 'robustness' / 'acn' / 'tariffs' / 'e3_tariffs_summary.csv')
    pd.testing.assert_frame_equal(s, pub)
