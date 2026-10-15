"""Amendment 024: render and score extra arms beside visual run 002, without re-rendering run 002's arms.

Spec: docs/research/20261013-unknown-size-comparison/amendment-024-visual-metrics-delivered-system-20261015.md.
Every step runs run 002's own code: visual_pipeline (inputs, attachment, resource freeze, Blender command, receipt
checks, judge bundle), fid.run_manifest (amendment-011 populations), mesh_collision.run (amendment 019) and
judge_batch's preflight. Nothing here calls an API or a server.

    H=/Volumes/harddisk/fastfillout/unknownsize-20261013; BASE=$H/visual-final-selected-002; OUT=$H/visual-024
    python -m fastfill.comparison.visual_024 prepare --arm A --rows $H/a021-test-geometry-arm/placement-geometry.jsonl \
        --base $BASE --out $OUT
    python -m fastfill.comparison.visual_024 prepare --arm S --rows $H/system-022-final/test/system/rooms.jsonl \
        --base $BASE --out $OUT                    [--system-eval $H/system-022-final/test/eval/results.json]
    caffeinate -dimsu python -m fastfill.comparison.visual_024 render --arm A --out $OUT [--workers 2 --threads 3]
    python -m fastfill.comparison.visual_024 fid --out $OUT [--arm A --arm S]
    python -m fastfill.comparison.visual_024 mesh --out $OUT [--arm A --arm S] [--workers 4]
    python -m fastfill.comparison.visual_024 bundle --out $OUT --arm A --arm S

prepare  OUT/ARM, write-once. --rows is an evaluate_pair / amendment-021 *-geometry.jsonl inside its sealed dir (arm A),
         or a fastfill.system.pipeline rooms.jsonl (arm S), bound to the amendment-022 final test evaluation that
         scored exactly those bytes (default ../eval/results.json) and converted to OUT/S/source/system-geometry.jsonl
         (raw_layout = the final layout). visual_pipeline._inputs on run 002's reference, eval data and resource freeze
         (same provenance, cohort and judge selection required), then visual_pipeline._attach: ARM-geometry.jsonl (run
         002's visual input format), cases.jsonl, attachment-ledger.jsonl. Every attached case must equal run 002's case
         except its layouts and per-system input hashes, and every row's house group run 002's.
render   visual_pipeline's render of that arm's layout only (run 002 rendered the reference and its systems already):
         same Blender binary (BASE/config.json), command, cameras, assets, views and resolution; resource checks before
         and after; receipts checked and combined by visual_pipeline. Failures are recorded, never replaced or retried.
         visual_pipeline has no resume (render_assets refuses an existing ledger), so neither has this: an existing
         render is refused. --limit N renders the first N rooms only (a smoke; fid and bundle refuse it).
fid      OUT/fid-ARMS: run 002's fid-manifest (absolute paths) plus each arm's perspective image, systems = run 002's +
         arms, judged = run 002's; fid.run_manifest scores every amendment-011 population (agent/ARM = reference & J &
         P & ARM, run 002's agents on their own rule, primary, judged). Features of run 002's images are fid-011's
         cache (identity checked: weights, vendor, versions, batch, threads, device), new images go through
         fid.extract_features with the same options; they enter by run_manifest's extractor hook, so fid.json says
         'injected'. fid-summary.json: every population, reuse counts, and whether every fid-011 population is
         reproduced exactly.
mesh     OUT/mesh-ARMS: mesh_collision.run (amendment 019, unchanged: 2 cm voxels, 0.5 cm sampling, tau 125/500/2000,
         500 primary) on run 002's rendered rooms with layouts GT, P (run 002's) and each arm; its system list is
         extended by the arms for that call only. reproduction-019.json: recomputed GT and P rows vs mesh-collision-019.
bundle   OUT/bundle-ARMS (>= 2 arms, judge_batch needs two): visual_pipeline._api_bundle per arm on run 002's 174 judge
         rooms (prompt identical to run 002's, checked), merged into one api-bundle + api-mapping.json +
         api-failures.json + MANIFEST.json; judge_batch's own preflight is run on it. On the server:
         judge_batch.py --bundle BUNDLE --output NEW --client CLIENT --arms A,S --max-cases 400 --expected-total 348
         --timeout 600 --model gpt-6-astra --reasoning-effort high --workers 4
"""
import argparse
from collections import Counter
from importlib.metadata import version
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
from unittest import mock

from fastfill.comparison import fid, judge_batch, mesh_collision, render_assets as render, visual_cases
from fastfill.comparison import visual_pipeline as vp
from fastfill.comparison.corpus import _publish

SPEC = 'docs/research/20261013-unknown-size-comparison/amendment-024-visual-metrics-delivered-system-20261015.md'
SAME = ('reference_manifest_sha256', 'data_manifest_sha256', 'corpus_manifest_sha256', 'corpus_link', 'eval_data')
KEEP = ('reference', 'P')             # run-002 layouts beside the arms in the mesh run (GT context, paired vs P)


def _rows(path):
    return visual_cases._index(visual_cases._read_rows(path))


def _checked(root, name, manifest='MANIFEST.json'):
    """root / name after checking it against the hash root / manifest records."""
    if vp._read(root / manifest)['files'].get(name) != render.file_sha256(root / name):
        raise ValueError(f'{root / name}: not the file {manifest} records')
    return root / name


def _label(arm):
    if (not isinstance(arm, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', arm)
            or arm in (*vp.SYSTEMS, 'GT', 'reference')):
        raise ValueError(f'arm must match [A-Za-z][A-Za-z0-9_]* and not be a run-002 label: {arm}')
    return arm


def _run002(base):
    """(base, config, visual dir, preparation.json) of visual run 002."""
    base = Path(base).resolve()
    visual = base / 'visual'
    return base, vp._read(base / 'config.json'), visual, vp._read(_checked(visual, 'preparation.json'))


def _groups(visual, cases):
    """run 002's visual/J-geometry.jsonl (house groups), checked against the hash its cases recorded."""
    path = visual / 'J-geometry.jsonl'
    if render.file_sha256(path) not in {c['prediction_provenance']['J_sha256'] for c in cases.values()
                                        if 'prediction_provenance' in c}:
        raise ValueError(f'{path} is not the J input run 002 attached')
    return path


def _reference(config, prep002):
    ref = Path(config['reference_data'])
    if vp._snapshot(ref, ('reference-cases.jsonl', 'resources.json'))[1] != prep002['provenance'][
            'reference_manifest_sha256']:
        raise ValueError('reference set differs from run 002')
    return ref


def _prepared(dest):
    """preparation.json of OUT/ARM after checking every file its MANIFEST.json records."""
    changed = [n for n, d in vp._read(dest / 'MANIFEST.json')['files'].items() if render.file_sha256(dest / n) != d]
    if changed:
        raise ValueError(f'{dest}: changed since prepare: {changed}')
    return vp._read(dest / 'preparation.json')


def _rendered(dest):
    """{uid: run-002-format ledger row} of a complete (not --limit) render of OUT/ARM."""
    if vp._read(dest / 'render.json')['limit'] is not None:
        raise ValueError(f'{dest}: a --limit smoke render is never scored')
    return {r['uid']: r for r in vp._read(_checked(dest, 'ledger.json', 'render.json'))['cases']}


def _arms(out, arms):
    arms = list(arms or sorted(p.parent.name for p in Path(out).glob('*/preparation.json')))
    if not arms or len(set(arms)) != len(arms):
        raise ValueError('no prepared arms, or an arm given twice')
    return [_label(a) for a in arms]


def _base(out, arms):
    bases = {_prepared(out / a)['base'] for a in arms}
    if len(bases) != 1:
        raise ValueError('arms prepared from different run-002 bases')
    return bases.pop()


# ---- prepare ---------------------------------------------------------------------------------------

def _system_rows(rows, dest, evaluation):
    """pipeline rooms.jsonl -> a sealed visual input dir, bound to the 022 final test evaluation of those bytes."""
    result = vp._read(evaluation)
    inputs = result['inputs']
    if ((result.get('mode'), result.get('scope')) != ('final', 'test')
            or inputs['system']['rooms_sha256'] != render.file_sha256(rows)):
        raise ValueError(f'{rows} is not the rooms.jsonl the amendment-022 final test evaluation {evaluation} scored')
    dest.mkdir()
    vp._rows_write(dest / 'system-geometry.jsonl', [
        {'uid': r['uid'], 'source': r['source'], 'group': r['group'], 'room_type': r.get('room_type'),
         'arm': 'system', 'stage': r['stage'], 'complete': r['complete'], 'raw_layout': r['layout']}
        for r in visual_cases._read_rows(rows)])
    summary = {'protocol': 'amendment-024 system rows', 'label': result.get('label'),
               'data_manifest_sha256': inputs['data_MANIFEST'], 'rows': str(rows),
               'rows_sha256': render.file_sha256(rows), 'evaluation': str(evaluation),
               'evaluation_sha256': render.file_sha256(evaluation)}
    vp._write(dest / 'summary.json', summary)
    vp._write(dest / 'MANIFEST.json', {'files': {n: render.file_sha256(dest / n)
                                                 for n in ('system-geometry.jsonl', 'summary.json')}})
    return summary


def _same_scenes(cases, rows, visual):
    """Every attached case is run 002's case but for its layouts; every row has run 002's house group."""
    old = _rows(_checked(visual, 'cases.jsonl'))
    strip = lambda case: {k: v for k, v in case.items() if k not in ('layouts', 'prediction_provenance')}  # noqa: E731
    for uid, case in cases.items():
        if strip(case) != strip(old[uid]) or case['layouts']['reference'] != old[uid]['layouts']['reference']:
            raise ValueError(f'{uid}: scene differs from run 002')
    groups = {r['uid']: r['group'] for r in visual_cases._read_rows(_groups(visual, old))}
    other = [u for u, r in rows.items() if r.get('group') != groups.get(u)]
    if other:
        raise ValueError(f'house groups differ from run 002: {other[:3]}')
    return len(cases)


def prepare(arm, rows, base, out, system_eval=None):
    """Attach one arm's rows to run 002's rooms and scenes -> OUT/ARM (write-once)."""
    _label(arm)
    rows, out = Path(rows).resolve(), Path(out).resolve()
    dest = out / arm
    if dest.exists():
        raise FileExistsError(dest)
    base, config, visual, prep002 = _run002(base)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.{arm}-', dir=out) as temp:
        stage = Path(temp)
        if rows.name.endswith('-geometry.jsonl'):
            source, kind, origin = rows.parent, rows.name[:-len('-geometry.jsonl')], None
        elif rows.name == 'rooms.jsonl':
            source, kind = stage / 'source', 'system'
            origin = _system_rows(rows, source, Path(system_eval).resolve() if system_eval
                                  else rows.parents[1] / 'eval' / 'results.json')
        else:
            raise ValueError('--rows must be an evaluate_pair *-geometry.jsonl or a fastfill.system rooms.jsonl')
        systems = {arm: (source, kind)}
        lineage = vp.systems_lineage(systems)
        ref, references, predictions, selected, provenance, _ = vp._inputs(
            config['reference_data'], systems, prep002['provenance']['eval_data'], lineage)
        if (any(provenance[k] != prep002['provenance'][k] for k in SAME) or selected != prep002['selected_uids']
                or list(references) != prep002['expected_uids']):
            raise ValueError('reference, evaluation data, cohort or judge selection differ from run 002')
        cases, attached = vp._attach(stage, ref, references, predictions, systems)
        compared = _same_scenes(cases, predictions[arm], visual)
        record = {'amendment': '024', 'arm': arm, 'base': str(base), 'rows': str(rows),
                  'rows_sha256': render.file_sha256(rows), 'kind': kind, 'system_rows': origin,
                  'source': str(dest / 'source' if origin else source), 'lineage': lineage, 'provenance': provenance,
                  'expected_uids': list(references), 'selected_uids': selected,
                  'counts': {'frozen': len(references), 'rows_in_cohort': len(predictions[arm]),
                             'attached': len(cases), 'failed': len(references) - len(cases),
                             'failures': dict(Counter(json.dumps(r['errors'], sort_keys=True)
                                                      for r in attached.values() if r['errors']))},
                  'scene_check': {'compared_with_run_002': compared, 'identical': True},
                  'code_sha256': {Path(m.__file__).name: render.file_sha256(m.__file__)
                                  for m in (sys.modules[__name__], vp, visual_cases, render)}}
        vp._write(stage / 'preparation.json', record)
        files = sorted(p.relative_to(stage).as_posix() for p in stage.rglob('*') if p.is_file())
        vp._write(stage / 'MANIFEST.json', {'arm': arm, 'files': {n: render.file_sha256(stage / n) for n in files}})
        _publish(stage, dest)
    return record


# ---- render ----------------------------------------------------------------------------------------

def render_arm(arm, out, workers=2, threads=3, timeout_seconds=vp.MAX_TIMEOUT_SECONDS, limit=None):
    """visual_pipeline's render of ARM's layouts in run 002's scenes; ledger.json in run 002's format."""
    vp._limits(workers, threads, timeout_seconds)
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError('limit must be a positive integer')
    dest = Path(out).resolve() / _label(arm)
    prep = _prepared(dest)
    if any((dest / n).exists() for n in ('chunks', 'ledger.json', 'render.json')):
        raise FileExistsError(f'{dest}: rendered before; a render is never resumed or repeated in place')
    _, config, _, prep002 = _run002(prep['base'])
    ref = _reference(config, prep002)
    references = _rows(ref / 'reference-cases.jsonl')
    resources = vp._read(ref / 'resources.json')
    attached = _rows(dest / 'attachment-ledger.jsonl')
    cases = {c['uid']: {**c, 'layouts': {arm: c['layouts'][arm]}}
             for c in visual_cases._read_rows(dest / 'cases.jsonl') if arm in c['layouts']}
    if limit is not None:
        keep = [u for u in references if u in cases][:limit]
        references, cases = {u: references[u] for u in keep}, {u: cases[u] for u in keep}
    start = time.monotonic()
    before = vp._resource_failures(cases, resources, 'before_render')
    renderable = {u: c for u, c in cases.items() if u not in before}
    chunks = vp._render(renderable, dest, config['blender'], workers, threads, timeout_seconds)
    after = vp._resource_failures(renderable, resources, 'after_render')
    ledger = vp._combine(references, cases, attached, chunks, threads, {**before, **after})
    ledger = [{**r, 'system_errors': {k: v for k, v in attached[r['uid']]['errors'].items() if k != 'reference'}}
              for r in ledger]
    vp._write(dest / 'ledger.json', {'cases': ledger, 'chunks': chunks, 'config': vp.SETTINGS, 'arm': arm,
                                     'limit': limit, 'provenance': prep['provenance'], 'expected_uids': list(references)})
    vp._rows_write(dest / 'failures.jsonl', [r for r in ledger if r['failure'] is not None])
    result = {'amendment': '024', 'arm': arm, 'blender': config['blender'], 'workers': workers, 'threads': threads,
              'timeout_seconds': timeout_seconds, 'limit': limit, 'wall_seconds': time.monotonic() - start,
              'counts': {'frozen': len(references), 'rendered': sum(arm in r['images'] for r in ledger),
                         'failed': sum(r['failure'] is not None for r in ledger)},
              'statuses': dict(Counter(r['status'] for r in ledger)),
              'blender_versions': dict(Counter((r.get('receipt') or {}).get('blender_version') for r in ledger
                                               if r['status'] == 'rendered')),
              'files': {n: render.file_sha256(dest / n) for n in ('ledger.json', 'failures.jsonl')}}
    vp._write(dest / 'render.json', result)
    return result


# ---- fid -------------------------------------------------------------------------------------------

def fid_arms(out, arms=None, batch_size=8, threads=4, device='cpu', fid_011=None, extract=None):
    """Amendment-011 FID/KID populations with the arms added; run 002's image features reused from fid-011."""
    import numpy as np

    out = Path(out).resolve()
    arms = _arms(out, arms)
    ledgers = {a: _rendered(out / a) for a in arms}
    base, config, visual, prep002 = _run002(_base(out, arms))
    dest = out / ('fid-' + '-'.join(arms))
    if dest.exists():
        raise FileExistsError(dest)
    old_path = Path(fid_011) if fid_011 else base.parent / 'fid-011-visual-002' / 'fid.json'
    cache = vp._read(old_path)['feature_cache']
    npz, meta_path = Path(cache['path']), Path(cache['metadata_path'])
    if render.file_sha256(npz) != cache['sha256'] or render.file_sha256(meta_path) != cache['metadata_sha256']:
        raise ValueError(f'{old_path}: its feature cache changed')
    identity = vp._read(meta_path)['identity']
    current = {'weights_sha256': fid.verify_weights(config['weights']), 'batch_size': batch_size,
               'threads': threads, 'device': device, 'extractor': 'injected' if extract else 'pytorch_fid:0.3.0:pool3'}
    if extract is None:
        current |= {'vendor_inception_sha256': fid.file_sha256(Path(config['vendor']) / 'pytorch_fid/inception.py'),
                    'versions': {n: version(n) for n in ('torch', 'torchvision', 'numpy', 'Pillow')}}
    if any(identity.get(k) != v for k, v in current.items()):
        raise ValueError('fid-011 features come from other extraction settings; refusing to mix them')
    with np.load(npz, allow_pickle=False) as stored:
        known = dict(zip((fid.ImageRef(i['path'], i['sha256']) for i in identity['images']), stored['features']))
    manifest = vp._read(_checked(visual, 'fid-manifest.json'))
    cases = []
    for case in manifest['cases']:
        images = {k: {**v, 'path': str(visual / v['path'])} if 'path' in v else v for k, v in case['images'].items()}
        for a in arms:
            row = ledgers[a][case['uid']]
            images[a] = dict(row['images'][a]['perspective']) if a in row['images'] else {
                'failure': row['failure'] or row['status']}
        cases.append({**case, 'images': images})
    dest.mkdir(parents=True)
    vp._write(dest / 'fid-manifest.json', {
        **{k: v for k, v in manifest.items() if k != 'cases'}, 'cases': cases,
        'systems': [*manifest['systems'], *arms], 'judged': prep002['judged'],
        'provenance': {'run_002': manifest['provenance'], 'amendment': '024',
                       'arms': {a: {'ledger_sha256': render.file_sha256(out / a / 'ledger.json'),
                                    **vp._read(out / a / 'preparation.json')['provenance']} for a in arms}}})
    counts = {}

    def extractor(refs, **options):
        missing = [r for r in refs if r not in known]
        found = dict(zip(missing, (extract or fid.extract_features)(missing, **options))) if missing else {}
        counts.update(reused=len(refs) - len(missing), extracted=len(missing))
        return np.stack([known[r] if r in known else found[r] for r in refs])

    fid.run_manifest(dest / 'fid-manifest.json', dest / 'fid.json', vendor=config['vendor'], weights=config['weights'],
                     batch_size=batch_size, threads=threads, device=device, extractor=extractor)
    pops = lambda r: {'pooled': r['pooled']['populations'],  # noqa: E731
                      **{s: x['populations'] for s, x in r['strata'].items()}}
    mine, theirs = pops(vp._read(dest / 'fid.json')), pops(vp._read(old_path))
    missed = [f'{s}: {p}' for s, ps in theirs.items() for p, x in ps.items() if mine[s].get(p) != x]
    summary = {'amendment': '024', 'spec': SPEC, 'arms': arms, 'disclosure': fid.SYSTEMS_DISCLOSURE,
               'populations': {s: {p: {k: x[k] for k in ('systems', 'n', 'fid', 'kid', 'small_n', 'status')}
                                   for p, x in ps.items()} for s, ps in mine.items()},
               'reproduces_fid_011': not missed, 'not_reproduced': missed,
               'features': {**counts, 'reused_from': str(npz), 'reused_npz_sha256': cache['sha256'],
                            'note': "run-002 images: fid-011's cached features; new images: fid.extract_features "
                                    "with the same options; both enter through run_manifest's extractor hook, "
                                    "which fid.json labels 'injected'"},
               'fid_json_sha256': render.file_sha256(dest / 'fid.json')}
    vp._write(dest / 'fid-summary.json', summary)
    return summary


# ---- mesh ------------------------------------------------------------------------------------------

def mesh_arms(out, arms=None, workers=4, limit=None, repeats=2000, mesh_019=None, command=''):
    """Amendment-019 mesh collisions of the arms beside run 002's GT and P layouts, unchanged settings."""
    out = Path(out).resolve()
    arms = _arms(out, arms)
    base, _, visual, _ = _run002(_base(out, arms))
    dest = out / ('mesh-' + '-'.join(arms))
    if dest.exists():
        raise FileExistsError(dest)
    old = _rows(_checked(visual, 'cases.jsonl'))
    errors = _rows(_checked(visual, 'attachment-ledger.jsonl'))
    rendered = {c['uid'] for p in sorted(visual.glob('chunks/*/input.json')) for c in vp._read(p)['cases']}
    mine = {a: _rows(out / a / 'cases.jsonl') for a in arms}
    ledgers = {a: _rows(out / a / 'attachment-ledger.jsonl') for a in arms}
    cohort, chunk, ledger = [], [], []
    for uid, case in old.items():
        if uid in rendered:       # the 019 rooms, as mesh_collision.run reads them: run 002's render inputs
            case = {**case, 'layouts': {**{k: case['layouts'][k] for k in KEEP if k in case['layouts']},
                                        **{a: mine[a][uid]['layouts'][a] for a in arms if a in mine[a][uid]['layouts']}}}
            chunk.append(case)
        cohort.append(case)
        ledger.append({'uid': uid, 'errors': {**{k: v for k, v in errors[uid]['errors'].items() if k in KEEP},
                                              **{a: ledgers[a][uid]['errors'][a] for a in arms
                                                 if a in ledgers[a][uid]['errors']}}})
    run = dest / 'input' / 'visual'
    vp._rows_write(run / 'cases.jsonl', cohort)
    vp._rows_write(run / 'attachment-ledger.jsonl', ledger)
    vp._write(run / 'chunks' / '00' / 'input.json', {'cases': chunk})
    shutil.copyfile(_groups(visual, old), run / 'J-geometry.jsonl')
    # summarize() reports only the labels in its SYSTEMS; add the arms for this call (mesh_collision.py unchanged).
    with mock.patch.object(mesh_collision, 'SYSTEMS', (*mesh_collision.SYSTEMS, *arms)):
        summary = mesh_collision.run(run.parent, dest / 'collision', workers, limit, repeats, command)
    m019 = Path(mesh_019) if mesh_019 else base.parent / 'mesh-collision-019'
    before = {(r['uid'], r['system']): r for r in visual_cases._read_rows(_checked(m019, 'rows.jsonl'))}
    now = [r for r in visual_cases._read_rows(dest / 'collision' / 'rows.jsonl') if r['system'] in ('GT', 'P')]
    differ = [[r['uid'], r['system']] for r in now if before.get((r['uid'], r['system'])) != r]
    reproduction = {'mesh_019': str(m019), 'rows_sha256': render.file_sha256(m019 / 'rows.jsonl'),
                    'compared_rows': len(now), 'identical_rows': len(now) - len(differ), 'differing': differ}
    vp._write(dest / 'reproduction-019.json', reproduction)
    return {'arms': arms, 'analysed_rooms': summary['analysed_rooms'], 'missing_rooms': len(summary['missing_rooms']),
            'mesh_free_500': {a: summary['systems'][a]['levels']['mesh_free_500']['estimate']
                              for a in arms if a in summary['systems']},
            'reproduces_019': not differ and bool(now)}


# ---- bundle ----------------------------------------------------------------------------------------

def bundle(out, arms=None):
    """Run 002's judge bundle format for its 174 judge rooms x the arms; judge_batch's preflight passes on it."""
    out = Path(out).resolve()
    arms = _arms(out, arms)
    if len(arms) < 2:
        raise ValueError('judge_batch.py needs at least two arms (--arms A,S)')
    ledgers = {a: list(_rendered(out / a).values()) for a in arms}
    base, config, visual, prep002 = _run002(_base(out, arms))
    dest = out / ('bundle-' + '-'.join(arms))
    if dest.exists():
        raise FileExistsError(dest)
    references = _rows(_reference(config, prep002) / 'reference-cases.jsonl')
    selected = prep002['selected_uids']
    old_map = vp._read(_checked(visual, 'api-mapping.json'))
    if sorted({m['uid'] for m in old_map.values()}) != sorted(selected):
        raise ValueError('run-002 judge rooms differ from its selection')
    preps = {a: vp._read(out / a / 'preparation.json') for a in arms}
    provenance = {**preps[arms[0]]['provenance'],
                  'geometry_manifest_sha256': render.json_sha256({a: preps[a]['lineage'][a] for a in arms})}
    if any(preps[a]['provenance'][k] != provenance[k] for a in arms for k in SAME):
        raise ValueError('arms prepared on different reference or evaluation data')
    with tempfile.TemporaryDirectory(prefix='.bundle-', dir=out) as temp:
        stage = Path(temp) / 'bundle'
        mapping, failures, requests = {}, [], []
        for a in arms:
            part = Path(temp) / 'arms' / a
            part.mkdir(parents=True)
            vp._api_bundle(part, references, {a: _rows(out / a / f'{a}-geometry.jsonl')}, selected, ledgers[a],
                           provenance, (a,))
            shutil.copytree(part / 'api-bundle', stage / 'api-bundle', dirs_exist_ok=True)
            mapping |= vp._read(part / 'api-mapping.json')
            failures += vp._read(part / 'api-failures.json')
            requests += vp._read(part / 'api-bundle' / 'manifest.json')['cases']
        vp._write(stage / 'api-bundle' / 'manifest.json', {'schema_version': 1, 'expected_model': vp.MODEL,
                                                           'cases': sorted(requests, key=lambda r: r['id'])})
        vp._write(stage / 'api-mapping.json', mapping)
        vp._write(stage / 'api-failures.json', failures)
        old = vp._read(_checked(visual, 'api-bundle/manifest.json'))['cases']
        old_prompts = {old_map[c['id']]['uid']: c['prompt_sha256'] for c in old}
        if any(old_prompts.get(mapping[c['id']]['uid'], c['prompt_sha256']) != c['prompt_sha256'] for c in requests):
            raise ValueError('a judge prompt differs from run 002')
        files = {p.relative_to(stage).as_posix(): render.file_sha256(p) for p in sorted(stage.rglob('*')) if p.is_file()}
        result = {'amendment': '024', 'arms': arms, 'selected_uids': selected, 'provenance': provenance,
                  'counts': {'selected': len(mapping), 'api_cases': len(requests), 'failed': len(failures)},
                  'judge_batch': f"--arms {','.join(arms)} --expected-total {len(mapping)}"}
        vp._write(stage / 'MANIFEST.json', {**result, 'files': files})
        client = judge_batch._client(Path(judge_batch.__file__).with_name('server_api_judge.py'))
        judge_batch._inputs(stage, None, None, client, len(mapping), len(mapping), tuple(arms))
        _publish(stage, dest)
    return result


# ---- CLI -------------------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    step = sub.add_parser('prepare')
    for name in ('rows', 'base', 'out'):
        step.add_argument('--' + name, type=Path, required=True)
    step.add_argument('--arm', required=True)
    step.add_argument('--system-eval', type=Path)
    step = sub.add_parser('render')
    step.add_argument('--arm', required=True)
    step.add_argument('--out', type=Path, required=True)
    step.add_argument('--workers', type=int, default=2)
    step.add_argument('--threads', type=int, default=3)
    step.add_argument('--timeout-seconds', type=float, default=vp.MAX_TIMEOUT_SECONDS)
    step.add_argument('--limit', type=int)
    for name, options in (('fid', ('batch-size', 'threads')), ('mesh', ('workers', 'repeats', 'limit')),
                          ('bundle', ())):
        step = sub.add_parser(name)
        step.add_argument('--out', type=Path, required=True)
        step.add_argument('--arm', action='append', dest='arms')
        defaults = {'batch-size': 8, 'threads': 4, 'workers': 4, 'repeats': 2000, 'limit': None}
        for option in options:
            step.add_argument('--' + option, type=int, default=defaults[option])
    sub.choices['fid'].add_argument('--device', default='cpu')
    sub.choices['fid'].add_argument('--fid-011', type=Path)
    sub.choices['mesh'].add_argument('--mesh-019', type=Path)
    args = vars(parser.parse_args(argv))
    command = args.pop('command')
    run = {'prepare': prepare, 'render': render_arm, 'fid': fid_arms, 'bundle': bundle,
           'mesh': lambda **kw: mesh_arms(**kw, command=' '.join(['python -m fastfill.comparison.visual_024',
                                                                  *(argv if argv is not None else sys.argv[1:])]))}
    try:
        result = run[command](**args)
    except vp.ERRORS as exc:
        parser.exit(2, f'visual_024 {command}: {exc}\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('expected_uids', 'selected_uids', 'populations')},
                     default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
