#!/usr/bin/env python3
"""Check that the metric files the paper relies on exist and are non-empty, and validate any diagnostic figures.

Run from the run directory (runs/<network>).
"""
from pathlib import Path
import json, sys

ROOT=Path.cwd()   # the run directory (runs/<network>), where the pipeline writes results/
required_metrics=[
    'regression_metrics_by_seed.csv','regression_metrics_summary.csv','protocol_optimism.csv',
    'rank_stability.csv','distribution_shift.csv','paired_model_comparisons.csv',
    'site_transfer_summary.csv','rq3_driver_matched_metrics_by_seed.csv',
    'rq3_driver_matched_metrics_summary.csv','pandemic_stratified_metrics_by_seed.csv',
    'pandemic_stratified_metrics_summary.csv',
    'identified_user_metrics_by_seed.csv','protocol_optimism_identified_users.csv',
    'cross_site_user_overlap.csv','conformal_reliability.csv','calibration_budget.csv',
    'occupancy_metrics.csv','data_flow.csv'
]
errors=[]
for name in required_metrics:
    p=ROOT/'results'/'metrics'/name
    if not p.exists() or p.stat().st_size==0:
        errors.append(f'missing/empty metric: {p.relative_to(ROOT)}')

png_dir=ROOT/'figures'/'png'; pdf_dir=ROOT/'figures'/'pdf'; svg_dir=ROOT/'figures'/'svg'
base_png={p.stem for p in png_dir.glob('*.png')}
base_pdf={p.stem for p in pdf_dir.glob('*.pdf')}
base_svg={p.stem for p in svg_dir.glob('*.svg')}
if png_dir.exists() and not base_png:   # diagnostic figures are optional; the paper's figures are made by paper/
    errors.append('no PNG figures generated')
for stem in sorted(base_png | base_pdf | base_svg):
    for ext,d in [('png',png_dir),('pdf',pdf_dir),('svg',svg_dir)]:
        p=d/f'{stem}.{ext}'
        if not p.exists() or p.stat().st_size < 500:
            errors.append(f'missing/suspicious figure: {p.relative_to(ROOT)}')

# Validate PNG resolution metadata and basic vector file signatures.
try:
    from PIL import Image
    for p in png_dir.glob('*.png'):
        with Image.open(p) as im:
            dpi=im.info.get('dpi',(0,0))
            if min(dpi) < 590:
                errors.append(f'PNG DPI below 600 target: {p.name} -> {dpi}')
except Exception as e:
    errors.append(f'PNG validation failed: {e}')

for p in pdf_dir.glob('*.pdf'):
    if not p.read_bytes()[:5] == b'%PDF-': errors.append(f'invalid PDF signature: {p.name}')
for p in svg_dir.glob('*.svg'):
    head=p.read_text(errors='ignore')[:500].lower()
    if '<svg' not in head: errors.append(f'invalid SVG: {p.name}')

report={
    'ok': not errors,
    'n_figures': len(base_png),
    'formats_complete': base_png==base_pdf==base_svg,
    'errors': errors,
}
(ROOT/'results'/'ARTIFACT_VALIDATION.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
sys.exit(0 if report['ok'] else 2)
