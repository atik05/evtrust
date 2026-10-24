"""Evaluate the domain-aware remedies under the frozen protocols.

Each remedy targets a mechanism measured in the main study. They are evaluated,
not assumed to work: every number is written whether or not it favours the
remedy.

  R1  HRGE: history-reliability gated ensemble (src/evtrust/hrge.py)
  R2  history-conditional conformal calibration on top of HRGE
  R3  published-style user aggregates (deliberately leaky, random protocol only)

Outputs in results/metrics/:
  contribution_accuracy.csv   MAE overall and per driver-history group
  contribution_gate.csv       gate weight chosen per history group
  contribution_coverage.csv   marginal versus history-conditional coverage
  published_style.csv         leakage-aware versus published-style features

Resumable: one shard per seed, split and target under results/contrib_shards/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from evtrust.config import load_config                                    # noqa: E402
from evtrust.conformal import split_conformal_calibrate, split_conformal_interval  # noqa: E402
from evtrust.experiments import make_splits, _split_frames                # noqa: E402
from evtrust.hrge import fit_hrge                                          # noqa: E402
from evtrust.improvements import (add_published_style_features, coverage_by_group,  # noqa: E402
                                  history_group, mondrian_regression_calibrate,
                                  mondrian_regression_interval)
from evtrust.models import fit_catboost                                    # noqa: E402
from evtrust.utils import seed_everything                                  # noqa: E402

TARGETS = {'duration': 'duration_hours', 'energy': 'kwh_delivered'}
GROUPS = ['0', '1-2', '3-9', '10+']


def _mae(y, p):
    return float(np.mean(np.abs(y - p))) if len(y) else float('nan')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/acn.yaml')
    ap.add_argument('--seeds', type=int, default=None)
    ap.add_argument('--min-group', type=int, default=30,
                    help='minimum calibration rows for a history group to get its own quantile')
    args = ap.parse_args()

    cfg = load_config(args.config)
    alpha = float(cfg['conformal']['alpha'])
    numeric, categorical = list(cfg['features']['numeric']), list(cfg['features']['categorical'])
    seeds = list(cfg['project']['five_seeds'])
    seeds = seeds[: args.seeds] if args.seeds else seeds
    params = dict(cfg['models']['catboost'])

    sessions = pd.read_parquet(Path(cfg['data'].get('processed_dir', 'data/processed')) / 'sessions.parquet')
    leaky = add_published_style_features(sessions)
    shard_dir = Path('results/contrib_shards')
    shard_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        seed_everything(seed)
        fit = lambda a, b, c, d, n, k, s=seed: fit_catboost(a, b, c, d, n, k, params, s)
        for sp in make_splits(sessions, cfg, seed):
            tr, ca, va, te = _split_frames(sessions, sp)
            for target, ycol in TARGETS.items():
                tag = f"seed{seed}__{sp.name.replace(':', '__')}__{target}"
                if (shard_dir / f'acc__{tag}.csv').exists():
                    continue
                ytr, yva, yca, yte = (f[ycol].to_numpy(float) for f in (tr, va, ca, te))
                ok = np.isfinite(ytr)
                base = {'seed': seed, 'protocol': sp.name.split(':')[0], 'split_name': sp.name,
                        'target': target}

                model = fit_hrge(tr[ok], ytr[ok], va, yva, numeric, categorical, fit)
                ph, pf = model.predict_parts(te)
                pe = model.gate(te) * ph + (1 - model.gate(te)) * pf
                g_te = history_group(te).to_numpy()

                acc = []
                for name, pred in [('catboost_full', ph), ('history_free', pf), ('hrge', pe)]:
                    row = {**base, 'model': name, 'mae': _mae(yte, pred), 'n': len(yte)}
                    for grp in GROUPS:
                        sel = g_te == grp
                        row[f'mae_{grp}'] = _mae(yte[sel], pred[sel])
                        row[f'n_{grp}'] = int(sel.sum())
                    acc.append(row)
                gate = [{**base, 'group': k, 'weight': v} for k, v in model.weights.items()]
                gate.append({**base, 'group': '__pooled__', 'weight': model.pooled_weight})

                p_ca = model.predict(ca)
                g_ca = history_group(ca)
                q = split_conformal_calibrate(yca, p_ca, alpha)
                lo_m, hi_m = split_conformal_interval(pe, q)
                table = mondrian_regression_calibrate(yca, p_ca, g_ca, alpha, min_per_group=args.min_group)
                lo_c, hi_c = mondrian_regression_interval(pe, history_group(te), table)
                cov = pd.concat([
                    coverage_by_group(yte, lo_m, hi_m, history_group(te)).assign(variant='marginal'),
                    coverage_by_group(yte, lo_c, hi_c, history_group(te)).assign(variant='history_conditional'),
                ], ignore_index=True)
                for k, v in base.items():
                    cov[k] = v
                cov['nominal'] = 1 - alpha

                pd.DataFrame(gate).to_csv(shard_dir / f'gate__{tag}.csv', index=False)
                cov.to_csv(shard_dir / f'cov__{tag}.csv', index=False)

                if sp.name == 'random':
                    ltr, lca, lva, lte = _split_frames(leaky, sp)
                    m_leak = fit(ltr[ok], ytr[ok], lva, yva, numeric, categorical)
                    pub = pd.DataFrame([
                        {**base, 'features': 'leakage_aware', 'mae': _mae(yte, ph)},
                        {**base, 'features': 'published_style', 'mae': _mae(yte, np.asarray(m_leak.predict(lte), float))},
                    ])
                    pub.to_csv(shard_dir / f'pub__{tag}.csv', index=False)
                # written last: its presence marks the unit as complete
                pd.DataFrame(acc).to_csv(shard_dir / f'acc__{tag}.csv', index=False)
                print(f'[contrib] {tag} done', flush=True)

    out = Path('results/metrics')
    out.mkdir(parents=True, exist_ok=True)
    for kind, name in [('acc', 'contribution_accuracy'), ('gate', 'contribution_gate'),
                       ('cov', 'contribution_coverage'), ('pub', 'published_style')]:
        files = sorted(shard_dir.glob(f'{kind}__*.csv'))
        if files:
            pd.concat([pd.read_csv(f) for f in files], ignore_index=True).to_csv(out / f'{name}.csv', index=False)

    acc = pd.read_csv(out / 'contribution_accuracy.csv')
    print('\nMAE by protocol (mean over seeds):')
    print(acc.pivot_table(index=['target', 'protocol'], columns='model', values='mae').round(3).to_string())
    cov = pd.read_csv(out / 'contribution_coverage.csv')
    print('\nCoverage by history group (nominal %.2f):' % (1 - alpha))
    print(cov.pivot_table(index=['target', 'protocol', 'group'], columns='variant', values='coverage').round(3).to_string())
    if (out / 'published_style.csv').exists():
        pub = pd.read_csv(out / 'published_style.csv')
        print('\nRandom protocol, feature construction:')
        print(pub.pivot_table(index='target', columns='features', values='mae').round(3).to_string())


if __name__ == '__main__':
    main()
