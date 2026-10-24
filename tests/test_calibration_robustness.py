"""The prospective calibration used in the robustness check never looks at later sessions."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('cal', Path(__file__).resolve().parents[1] / 'scripts' / 'calibration_robustness.py')
cal = importlib.util.module_from_spec(spec); spec.loader.exec_module(cal)


def test_prospective_quantile_ignores_later_sessions():
    rng = np.random.default_rng(0)
    n = 3000
    groups = rng.integers(0, 5, n)
    month = np.repeat(np.arange(10), n // 10)
    resid = rng.normal(0, 1, n)
    resid[month == 9] = 1e6
    q = cal.prospective_quantile(resid, groups, 0.99, month, month)
    assert np.nanmax(q) < 10
    assert np.isnan(q[month == 0]).all()


def test_flex_at_interpolates_and_flags_unreachable_targets():
    import pandas as pd
    c = pd.DataFrame({'unmet_pct': [0.5, 1.5, 3.0], 'flex_pct': [10.0, 20.0, 40.0]})
    assert cal.flex_at(c, 1.0) == 15.0
    assert np.isnan(cal.flex_at(pd.DataFrame({'unmet_pct': [3.0, 4.0], 'flex_pct': [1.0, 2.0]}), 1.0))
