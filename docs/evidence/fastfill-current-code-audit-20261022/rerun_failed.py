import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as E

root = Path('/tmp/fastfill-audit-20261022')
tree = E.parse(root / 'pytest-supported.xml')
ids = [case.get('classname').replace('.', '/') + '.py::' + case.get('name')
       for case in tree.findall('.//testcase') if case.find('failure') is not None]
# Pytest importlib names this checkout as a package under its parent.
ids = [name.removeprefix('OptiScene/') for name in ids]
assert len(ids) == 18, ids
environment = dict(os.environ)
environment['PYTHONPATH'] = os.pathsep.join([str(root / 'qa-deps'), str(Path.cwd()), str(Path.cwd().parent)])
environment.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='',
                   COVERAGE_FILE=str(root / '.coverage-supported'))
command = [sys.executable, '-m', 'pytest', '--import-mode=importlib', '--cov=fastfill', '--cov-append',
           '--cov-report=json:' + str(root / 'coverage-supported.json'),
           '--cov-report=term:skip-covered', '--junitxml=' + str(root / 'pytest-rerun.xml'), '-q', *ids]
(root / 'rerun-nodeids.json').write_text(__import__('json').dumps(ids, indent=2) + '\n')
with (root / 'pytest-rerun.log').open('w') as log:
    process = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
raise SystemExit(process.returncode)
