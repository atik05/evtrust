"""acnportal 0.3.3 imports pkg_resources, which recent setuptools no longer provides. It is used only to read
acnportal's own version when saving a simulation; replace it with importlib.metadata (standard library)."""
import importlib.util
import pathlib
import sys

spec = importlib.util.find_spec('acnportal')
if spec is None:
    sys.exit('acnportal is not installed')
base = pathlib.Path(spec.origin).parent / 'acnsim' / 'base.py'
src = base.read_text()
if 'pkg_resources' not in src:
    print('acnportal already compatible'); sys.exit(0)
src = src.replace('import pkg_resources\n', 'from importlib.metadata import version as _acnportal_version\n')
src = src.replace('pkg_resources.require("acnportal")[0].version', '_acnportal_version("acnportal")')
assert 'pkg_resources' not in src, 'unexpected pkg_resources use left in acnportal'
base.write_text(src)
print(f'updated {base}')
