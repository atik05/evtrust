"""Environment check before a run (Python, numerical libraries, GPU).

Writes results/environment.json and prints the decisions the pipeline will
make. Exits non-zero only for conditions that would invalidate the run.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

report: dict = {'platform': platform.platform(), 'python': sys.version.split()[0]}
problems: list[str] = []
warnings: list[str] = []

try:
    import numpy as np
    report['numpy'] = np.__version__
    if np.__version__.startswith('2.5'):
        warnings.append('numpy 2.5.x has been associated with native crashes on this class of machine; 1.26.4 is pinned by the setup script.')
except Exception as exc:  # pragma: no cover
    problems.append(f'numpy import failed: {exc!r}')

try:
    import torch
    report['torch'] = torch.__version__
    report['cuda_available'] = bool(torch.cuda.is_available())
    report['cuda_version'] = torch.version.cuda
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024 ** 3)
        report['gpu_name'] = props.name
        report['gpu_vram_gb'] = round(vram_gb, 1)
        x = torch.randn(2048, 2048, device='cuda')
        torch.cuda.synchronize()
        report['gpu_smoke_ok'] = bool(float((x @ x).sum().item()) == float((x @ x).sum().item()))
        # Batch size is chosen from measured VRAM, not assumed.
        if vram_gb >= 20:
            batch = 2048
        elif vram_gb >= 10:
            batch = 1024
        elif vram_gb >= 6:
            batch = 512
        else:
            batch = 256
        report['recommended_batch_size'] = batch
    else:
        warnings.append('CUDA is not available; the run will fall back to CPU and will be much slower.')
        report['recommended_batch_size'] = 256
except Exception as exc:
    problems.append(f'torch check failed: {exc!r}')
    report['recommended_batch_size'] = 256

try:
    import catboost
    report['catboost'] = catboost.__version__
except Exception as exc:
    problems.append(f'catboost import failed: {exc!r}')

try:
    out = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                                   '--format=csv,noheader'], text=True).strip()
    report['nvidia_smi'] = out
except Exception as exc:
    warnings.append(f'nvidia-smi unavailable: {exc!r}')

report['cpu_count'] = os.cpu_count()
report['taskset_pinning'] = os.environ.get('EVTRUST_TASKSET', '(not set)')
report['warnings'] = warnings
report['problems'] = problems

Path('results').mkdir(parents=True, exist_ok=True)
Path('results/environment.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
if problems:
    print('\nPreflight failed:')
    for p in problems:
        print(' -', p)
    raise SystemExit(2)
print('\nPreflight OK.')
