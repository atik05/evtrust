"""Run the robustness and ACN-Sim stages for one network.

    python -m robustness.run --config configs/robustness.yaml --network acn [--smoke] [--only a,b] [--skip a,b]

Each finished stage writes out/<mode>/<network>/.done/<stage>; a rerun skips finished stages and every
expensive unit (one model fit set per seed) is saved as it finishes, so an interrupted run resumes.
A failing stage is logged with its traceback and the next stage runs; preflight failure stops the network.
"""
from __future__ import annotations

import argparse
import time
import traceback

from .common import Ctx

STAGES = {
    'preflight': ('preflight', 'run'), 'firstvisit': ('firstvisit', 'run'), 'charging': ('charging', 'run'),
    'anonymous': ('anonymous', 'run'), 'decomp': ('decomp', 'run_alt'), 'driver_id': ('decomp', 'run_id'),
    'metrics': ('metrics', 'run'), 'cs_baselines': ('cs_baselines', 'run'), 'laxity': ('laxity', 'run'), 'temporal': ('temporal', 'run'), 'acnsim': ('acnsim', 'run'), 'report': ('report', 'run'),
    'subsets': ('population', 'run_subsets'), 'reliability': ('population', 'run_reliability'), 'periods': ('population', 'run_periods'),
    'infostate': ('infostate', 'run'), 'infostate_sim': ('infostate', 'run_sim'), 'feeder': ('feeder', 'run'),
}
NEEDS = {'acnsim': 'anonymous', 'infostate': 'anonymous', 'infostate_sim': 'anonymous', 'feeder': 'acnsim'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/robustness.yaml')
    ap.add_argument('--network', required=True)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--only', default='')
    ap.add_argument('--skip', default='')
    a = ap.parse_args()
    ctx = Ctx(a.config, a.network, a.smoke)
    stages = list(ctx.ncfg['stages'])
    if a.only:
        # stages named with --only run in the order given and need not be listed in the configuration
        unknown = [s for s in a.only.split(',') if s and s not in STAGES]
        if unknown:
            raise SystemExit(f'unknown stage(s): {unknown}')
        stages = [s for s in a.only.split(',') if s]
    stages = [s for s in stages if s not in a.skip.split(',')]
    failed = []
    for st in stages:
        marker = ctx.done(st)
        if marker.exists():
            ctx.log(f'stage {st}: already done'); continue
        if st in NEEDS and not ctx.done(NEEDS[st]).exists():
            ctx.log(f'stage {st}: waiting for {NEEDS[st]}, which has not finished; skipped'); failed.append(st); continue
        ctx.log(f'stage {st}: start')
        t0 = time.time()
        try:
            mod, fn = STAGES[st]
            getattr(__import__(f'robustness.{mod}', fromlist=[fn]), fn)(ctx)
            marker.write_text(f'{time.strftime("%Y-%m-%d %H:%M:%S")} {time.time() - t0:.0f}s\n')
            ctx.log(f'stage {st}: done in {(time.time() - t0) / 60:.1f} min')
        except SystemExit as e:
            ctx.log(f'stage {st}: STOPPED: {e}')
            raise
        except Exception:                                     # noqa: BLE001
            ctx.log(f'stage {st}: FAILED after {(time.time() - t0) / 60:.1f} min\n{traceback.format_exc()}')
            ctx.p('.failed', st).write_text(traceback.format_exc())
            failed.append(st)
    ctx.log(f'network finished; failed stages: {failed or "none"}')
    if failed and a.smoke:
        raise SystemExit(f'smoke run failed in {failed}')


if __name__ == '__main__':
    main()
