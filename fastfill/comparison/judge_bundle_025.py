"""Amendment 025: judge bundles, in run 002's api-bundle format, for dev arms B/C and test arm SC.

Spec: docs/research/20261013-unknown-size-comparison/amendment-025-critic-repair-20261015.md (sections 2 and 3).
Nothing here calls an API or a server; judge_batch's own preflight is run on every bundle before it is published.

Image source (decided 2026-10-15, before any 025 output): dev rooms cannot use run 002's Blender pipeline.
visual_cases.attach_systems on 3 dev-select rooms: 'foreign prediction UID outside reference cohort' (0 of 600 dev
rooms are in run 002's frozen reference cohort); visual_cases' freeze refuses non-test rooms; the frozen assets cover
IL3D_3dfront and OptiScene_holodeck only, the dev rooms 13 sources. So both dev arms are judged on critic_025's box
views (the critic's own images: oblique as 'perspective', then top), disclosed as image_source 'box_views' in
MANIFEST.json (api-bundle/provenance.json keeps judge_batch's six fixed keys). The test arms are Blender renders.

    H=/Volumes/harddisk/fastfillout/unknownsize-20261013; DEV=$H/a025-inputs/dev-select-v1
    python -m fastfill.comparison.judge_bundle_025 dev --dev $DEV --b $H/a025/round-0 --c $H/a025/round-2 \
        --out $H/a025/judge-dev [--workers 8]
    judge_batch.py --bundle $H/a025/judge-dev --output NEW --client CLIENT MANIFEST['judge_batch'] --timeout 600 \
        --workers 4                                (MANIFEST['judge_batch'] = the arms, 2 x rooms cases, judge model)
    python -m fastfill.comparison.critic_025 evaluate --b $H/a025/round-0 --c $H/a025/round-2 --judge NEW --out ...

dev  critic_025 state dirs: --b round-0 (arm B), --c the final round (arm C, built on that B, no room still
     active, as critic_025 evaluate requires), --dev the dev-select
     data B was built on. Per room and arm: critic_025.flags + render_views of the arm's layout in the critic's
     request-frame room, titled by the uid only (the critic's title carries its round, which would reveal the arm);
     images = oblique as 'perspective', then top. Prompt = visual_pipeline.build_prompt verbatim (run 002's rubric; it
     says 'perspective, then top' and 512x512, the box views are 1024 px) on the room's boundary/fixed/category
     inventory, one prompt per room shared by both arms; stratum = source (judge_batch's descriptive within-stratum
     bootstrap), group = house group. Case id = visual_pipeline's sha256 of [uid, arm] (opaque; the judge sees only
     prompt and images), mapping in case-id order, so arms interleave. A views failure is an api-failures entry, never
     replaced. --arms C,B makes judge_batch's paired tables C minus B (the adoption direction); its endpoint averages
     three scores, the amendment's (functionality + layout) / 2 with the house bootstrap is critic_025 evaluate --judge.
sc   SC rows = S's 174 judge rooms (BASE/visual/preparation.json selected_uids) after the critic -> a visual_024
     input dir (SC-geometry.jsonl, summary.json bound to S's data manifest, MANIFEST.json). Ids, categories, sizes
     and parents must equal S's; object order is S's; every S judge room is required. Then visual_024:
    python -m fastfill.comparison.judge_bundle_025 sc --rows SC_ROWS --s $H/visual-024/S --out $H/a025/SC-rows
    python -m fastfill.comparison.visual_024 prepare --arm SC --rows $H/a025/SC-rows/SC-geometry.jsonl \
        --base $H/visual-final-selected-002 --out $H/visual-024
    caffeinate -dimsu python -m fastfill.comparison.visual_024 render --arm SC --out $H/visual-024
    python -m fastfill.comparison.visual_024 bundle --out $H/visual-024 --arm SC --paired-with S
    judge_batch.py --bundle $H/visual-024/bundle-SC --arms SC,S --max-cases 174 --expected-total 174 ... (as 024)
     Only SC is judged (spec section 4: <= 174 test judge calls); S's judgements are 024's (same judge, prompt and
     rooms), joined to SC's by uid.
"""
import argparse
import json
from pathlib import Path
import tempfile

from fastfill.comparison import critic_025 as critic, judge_batch, visual_024, visual_cases
from fastfill.comparison import render_assets as render, report_v2 as r2
from fastfill.comparison import visual_pipeline as vp
from fastfill.comparison.corpus import _publish

ARMS = ('B', 'C')                        # critic_025.judge_values reads exactly these labels
SPEC = 'docs/research/20261013-unknown-size-comparison/amendment-025-critic-repair-20261015.md'
DISCLOSURES = [
    "build_prompt's rubric names 512x512 'perspective, then top' renders; the images are critic_025's 1024 px box "
    "views (oblique as 'perspective', then top)",
    "the judge's box views are neutral (every movable box in one colour, no checker-flag legend), unlike the "
    "critic's flag-coloured views, so the judge cannot read checker results from colour"]
IMAGE_CHECK = {'image_source': 'box_views', 'checked': '2026-10-15', 'dev_rooms_tried': 3,
               'attach_systems': 'foreign prediction UID outside reference cohort',
               'dev_rooms_in_run_002_reference': 0, 'freeze': 'reference export contains a non-test candidate',
               'frozen_asset_sources': ['IL3D_3dfront', 'OptiScene_holodeck']}


def _layout(row):
    layout = row.get('raw_layout', row.get('layout'))
    if not isinstance(layout, list) or not layout:
        raise ValueError(f"{row.get('uid')}: no layout")
    return layout


def _views(task):
    """(room, layout, title) -> ((top, oblique) PNG bytes, None) by critic_025's own code, or (None, reason)."""
    room, layout, title = task
    try:
        critic.flags(room, layout)  # validates the room as before; its colours are not shown to the judge
        # Neutral views: every box in the 'ok' colour, no flag legend, so the judge cannot read checker results.
        return critic.render_views(room, layout, {o['id']: 'ok' for o in layout}, title, flag_legend=False), None
    except vp.ERRORS as exc:
        return None, f'{type(exc).__name__}: {exc}'


def _context(room, layout):
    """The run-002 judge case fields build_prompt reads, from critic_025's request-frame room."""
    return {'boundary_type': room['boundary_type'], 'boundary': room['boundary'], 'height': room.get('height'),
            'stratum': visual_cases._stratum(room), 'layouts': {'reference': layout}, 'fixed': room['fixed']}


def _legend(layout):
    """Amendment 025 addendum B: the boxes' id -> category list (identical for B and C; ids/categories never change)."""
    pairs = sorted((o['id'], o['category']) for o in layout)
    return ('\nThe boxes in the images are labelled by id; each id is this piece of furniture: '
            + json.dumps([{'id': i, 'category': c} for i, c in pairs], ensure_ascii=False) + '\n')


def dev(dev_dir, b, c, out, workers=1, views=_views):
    """Box-view judge bundle of critic_025's B (round-0) and C (last round) states -> OUT (write-once)."""
    dev_dir, out = Path(dev_dir).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    (b_rows, b_summary, b_sha), (c_rows, c_summary, c_sha) = critic._state(b), critic._state(c)
    if b_summary['round'] != 0 or c_summary['round'] < 1 or c_summary['b_manifest_sha256'] != b_sha \
            or [r['uid'] for r in b_rows] != [r['uid'] for r in c_rows]:
        raise ValueError('C must be a critic_025 round state built on this B')
    if any(r['active'] for r in c_rows):
        raise ValueError('C is not final: some rooms go on to the next round')
    data_hash = render.file_sha256(dev_dir / 'MANIFEST.json')
    if b_summary['inputs']['data_manifest_sha256'] != data_hash:
        raise ValueError(f'B was not built on {dev_dir}')
    lineage = {'B': {'state': str(Path(b).resolve()), 'manifest_sha256': b_sha},
               'C': {'state': str(Path(c).resolve()), 'manifest_sha256': c_sha, 'round': c_summary['round']}}
    provenance = {'geometry_manifest_sha256': render.json_sha256({'B': b_sha, 'C': c_sha}), 'reference_manifest_sha256': data_hash,
                  'data_manifest_sha256': data_hash,
                  'corpus_manifest_sha256': vp._read(dev_dir / 'MANIFEST.json')['main_data_manifest_sha256'],
                  'corpus_link': 'unverified_without_eval_data'}
    # Same title for both arms (the critic's carries its round, which would reveal the arm).
    tasks = [(arm, r) for r in b_rows for arm in ARMS]
    states = {'B': {r['uid']: r for r in b_rows}, 'C': {r['uid']: r for r in c_rows}}
    pngs = r2._map(views, [(r['room'], states[arm][r['uid']]['layout'], r['uid'][-48:]) for arm, r in tasks], workers)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.' + out.name + '-', dir=out.parent) as temp:
        stage = Path(temp)
        bundle = stage / 'api-bundle'
        mapping, failures, requests = {}, [], []
        for (arm, row), (images, reason) in zip(tasks, pngs):
            uid = row['uid']
            prompt = Path('prompts') / (vp._hash(uid) + '.txt')
            if not (bundle / prompt).exists():
                (bundle / prompt).parent.mkdir(parents=True, exist_ok=True)
                (bundle / prompt).write_text(vp.build_prompt(_context(row['room'], row['layout'])) + _legend(row['layout']),
                                             encoding='utf-8')
            identity = vp._hash([uid, arm])
            meta = {'uid': uid, 'arm': arm, 'stratum': row['source'], 'group': row['group']}
            mapping[identity] = meta
            if reason:
                failures.append({'id': identity, **meta, 'failure': {'stage': 'box_views', 'reason': reason}})
                continue
            entries = []
            for view, data in zip(('top', 'perspective'), images):     # render_views returns (top, oblique)
                target = Path('images') / identity / f'{view}.png'
                (bundle / target).parent.mkdir(parents=True, exist_ok=True)
                (bundle / target).write_bytes(data)
                entries.append({'path': target.as_posix(), 'sha256': render.file_sha256(bundle / target)})
            requests.append({'id': identity, 'prompt_file': prompt.as_posix(),
                             'prompt_sha256': render.file_sha256(bundle / prompt), 'images': entries[::-1]})
        vp._write(bundle / 'manifest.json', {'schema_version': 1, 'expected_model': vp.MODEL,
                                             'cases': sorted(requests, key=lambda r: r['id'])})
        vp._write(bundle / 'provenance.json', {'schema_version': 1, **provenance})
        vp._write(stage / 'api-mapping.json', dict(sorted(mapping.items())))
        vp._write(stage / 'api-failures.json', failures)
        files = {p.relative_to(stage).as_posix(): render.file_sha256(p)
                 for p in sorted(stage.rglob('*')) if p.is_file()}
        n = len(mapping)
        result = {'amendment': '025', 'spec': SPEC, 'arms': list(ARMS), 'image_source': 'box_views',
                  'image_source_check': IMAGE_CHECK, 'disclosures': DISCLOSURES, 'selected_uids': [r['uid'] for r in b_rows], 'lineage': lineage,
                  'provenance': {**provenance, 'image_source': 'box_views'},
                  'prompt': 'visual_pipeline.build_prompt verbatim + id->category list (addendum B); images: critic_025 box views (oblique, then top)',
                  'views_code_sha256': render.file_sha256(critic.__file__),
                  'counts': {'rooms': len(b_rows), 'selected': n, 'api_cases': len(requests), 'failed': len(failures)},
                  'judge_batch': f'--arms C,B --max-cases {n} --expected-total {n} '
                                 '--model gpt-6-astra --reasoning-effort high'}
        vp._write(stage / 'MANIFEST.json', {**result, 'files': files})
        client = judge_batch._client(Path(judge_batch.__file__).with_name('server_api_judge.py'))
        judge_batch._inputs(stage, None, None, client, n, n, ARMS)
        _publish(stage, out)
    return result


def sc(rows, s_dir, out):
    """Critic rows on S's layouts -> a sealed visual_024 input dir for arm SC (write-once)."""
    s_dir, out = Path(s_dir).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    visual_024._prepared(s_dir)
    source = s_dir / 'source'
    base = visual_cases._index(visual_cases._read_rows(visual_024._checked(source, 'system-geometry.jsonl')))
    summary = vp._read(visual_024._checked(source, 'summary.json'))
    new = visual_cases._index(visual_cases._read_rows(rows))
    if not new.keys() <= base.keys():
        raise ValueError('SC rooms outside S')
    key = lambda o: (o['category'], o['size'], o.get('parent'))  # noqa: E731
    if not set(vp._read(s_dir / 'preparation.json')['selected_uids']) <= new.keys():
        raise ValueError("SC rows miss some of S's judge rooms")
    sc_rows = []
    for uid, row in new.items():
        old = base[uid]
        layout = {o['id']: o for o in _layout(row)}
        if len(layout) != len(_layout(row)):
            raise ValueError(f'{uid}: duplicate object ids')
        if {o['id']: key(o) for o in old['raw_layout']} != {i: key(o) for i, o in layout.items()}:
            raise ValueError(f'{uid}: ids, categories, sizes or parents differ from S')
        sc_rows.append({**{k: old[k] for k in ('uid', 'source', 'group', 'room_type', 'stage', 'complete')},
                        'arm': 'SC', 'raw_layout': [layout[o['id']] for o in old['raw_layout']]})
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.' + out.name + '-', dir=out.parent) as temp:
        stage = Path(temp)
        vp._rows_write(stage / 'SC-geometry.jsonl', sc_rows)
        record = {'protocol': 'amendment-025 SC rows', 'data_manifest_sha256': summary['data_manifest_sha256'],
                  'rows': str(Path(rows).resolve()), 'rows_sha256': render.file_sha256(rows), 's': str(s_dir),
                  's_manifest_sha256': render.file_sha256(s_dir / 'MANIFEST.json'), 'rooms': len(sc_rows)}
        vp._write(stage / 'summary.json', record)
        vp._write(stage / 'MANIFEST.json', {'files': {n: render.file_sha256(stage / n)
                                                      for n in ('SC-geometry.jsonl', 'summary.json')}})
        _publish(stage, out)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    step = sub.add_parser('dev')
    for name in ('dev', 'b', 'c', 'out'):
        step.add_argument('--' + name, type=Path, required=True)
    step.add_argument('--workers', type=int, default=1)
    step = sub.add_parser('sc')
    for name in ('rows', 's', 'out'):
        step.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'dev':
            result = dev(args.dev, args.b, args.c, args.out, args.workers)
        else:
            result = sc(args.rows, args.s, args.out)
    except (*vp.ERRORS, judge_batch.BatchError) as exc:
        parser.exit(2, f'judge_bundle_025 {args.command}: {exc}\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'selected_uids'}, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
