from pathlib import Path
import json
import tempfile
from fastfill.comparison import mesh_collision as m
from fastfill.comparison.tests import test_mesh_collision as test

results = {}
with tempfile.TemporaryDirectory(prefix='fastfill-mesh-review-') as d:
    root = Path(d)
    # Existing production test fixture writes two rendered cases + one not rendered,
    # and first verifies the normal no-limit invocation succeeds.
    test.test_cli_writes_once_and_reports_missing(root)
    try:
        m.run(root / 'run', root / 'limited', workers=1, limit=100, repeats=2)
    except Exception as exc:
        results['limit_exceeds_rendered_population'] = {
            'rendered_rooms': 2, 'limit': 100,
            'exception': type(exc).__name__ + ': ' + str(exc),
            'published': (root / 'limited').exists(),
        }

failed = [{'uid': 'u', 'rows': [], 'pairs': [], 'error': 'FileNotFoundError: missing mesh',
           'seconds': 0.0, 'cache': {}, 'meshes': {}}]
summary = m.summarize(failed, {'u': 'g'}, {'u': {'uid': 'u'}}, {}, repeats=2)
try:
    m.report(summary, 0.0, 1, 'CPU review probe')
except Exception as exc:
    results['no_successful_mesh_rooms'] = {
        'analysed_rooms': summary['analysed_rooms'],
        'missing_rooms': summary['missing_rooms'],
        'exception': type(exc).__name__ + ': ' + str(exc),
    }
print(json.dumps(results, indent=2))
