"""Audit the two real server smoke outputs; no API, GPU, or model inference."""
import json
from pathlib import Path
import sys

from fastfill.system import pipeline as p, sizing
from fastfill.comparison import infer


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify(root):
    if not __debug__:
        raise RuntimeError('Verification requires Python assertions enabled')
    selection = read(root / 'selection.json')
    jobs = read(root / 'jobs.json')
    assert set(selection['cases']) == set(jobs['jobs']) == {'missing-category', 'no-prior'}
    assert selection['cases']['missing-category']['uses_prior'] is True
    assert selection['cases']['no-prior']['uses_prior'] is False
    prior = p.load_prior(root.parent / 'main-data-v1/size-priors.json',
                        '073522c18e10ce2443189dfb8ebca90074c16ff1c2a4c292dfd6f788409d2422')
    cases = {}
    for name, selected in selection['cases'].items():
        job_root = root / jobs['jobs'][name].get('output_dir', name)
        out = job_root / 'system'
        result, status = read(out / 'result.json'), read(job_root / 'job-status.json')
        assert status['exit_code'] == 0
        assert status['job_id'] == jobs['jobs'][name]['job_id']
        assert result['status'] == 'SYSTEM_COMPLETE' and not result['dry_run']
        assert result['placer']['model'] == '/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/models/Qwen3.8-27B'
        assert result['prior']['sha256'] == (prior[1]['sha256'] if selected['uses_prior'] else None)
        assert result['placer']['engine'] == 'vllm_greedy_bf16_lora'
        assert result['placer']['adapter_sha256'] == 'cc3e286fa715be1892d13c0e4f29721ac32324903d8bb08e81f1951fee5fd1e6'
        assert result['escalate'] == 'none' and not result['escalation']
        api = read(out / 'size-api/result.json')
        assert api['status'] == 'SIZE_ESTIMATION_COMPLETE' and not api['failed_uids']
        ledger = rows(out / 'size-api/calls.jsonl')
        assert 1 <= len(ledger) == api['calls_ledger'] == result['api_calls_total'] <= 6
        assert all(item['uid'] == selected['uid'] for item in ledger)
        receipts = list((out / 'size-api/receipts').glob('*.json'))
        assert len(receipts) == 1
        receipt = read(receipts[0])
        assert receipt['uid'] == selected['uid'] and receipt['valid_final'] and not receipt['error']
        last = receipt['answers'][-1]
        assert last['status'] == 'completed' and last['incomplete_reason'] is None and last['schema']['valid']
        request, = rows(root / (name + '.jsonl'))
        assert infer._hash(request) == selected['request_sha256']
        assert len(request['messages']) == 2 and 'target' not in request and 'flags' not in request
        initial = sizing.plan([(request, None)], prior[0] if selected['uses_prior'] else None)
        layout_rows, estimate_rows, export_rows = (rows(out / f) for f in ('rooms.jsonl', 'size-estimates.jsonl', 'roomgenbench.jsonl'))
        assert len(layout_rows) == len(estimate_rows) == len(export_rows) == 1
        layout, estimates, exported = layout_rows[0], estimate_rows[0], export_rows[0]
        assert layout['uid'] == selected['uid'] == estimates['uid'] == exported['uid']
        assert layout['candidates'][0]['stage'] == 'placer' and layout['candidates'][0]['complete']
        assert layout['stage'] == 'placer' and layout['complete']
        assert exported['error'] is None
        sizes = estimates['estimated_sizes']
        assert len(layout['layout']) == len(sizes) == selected['objects']
        assert {o['id']: o['size'] for o in layout['layout']} == sizes
        assert {o['id']: o['target_size_local_m'] for o in exported['layout']['objects']} == sizes
        assert len(exported['layout']['objects']) == len(exported['scene']['objects']) == len(sizes)
        assert {o['id']: [o['dimensions'][k] for k in ('length', 'width', 'height')]
                for o in exported['scene']['objects']} == sizes
        assert layout['size_sources'] == estimates['sources']
        assert sum(source == 'api' for source in estimates['sources'].values()) == selected['api_objects']
        known = initial['sizes'][selected['uid']]
        pending_ids = set(sizes) - set(known)
        assert set(last['schema']['sizes']) == pending_ids
        assert {oid: sizes[oid] for oid in pending_ids} == {oid: sizing.rounded(s) for oid, s in last['schema']['sizes'].items()}
        parsed = sizing.S.parse_sizes(last['raw'], sorted(pending_ids))
        assert parsed['valid'] and {oid: sizing.rounded(s) for oid, s in parsed['sizes'].items()} == last['schema']['sizes']
        assert estimates['sources'] == {**initial['sources'][selected['uid']], **dict.fromkeys(pending_ids, 'api')}
        assert infer._hash(p.placement_request(request, sizes)) == estimates['placement_request_sha256']
        for oid, size in initial['sizes'][selected['uid']].items():
            assert sizes[oid] == size and estimates['sources'][oid] == initial['sources'][selected['uid']][oid]
        categories = {o['id']: o['category'] for o in json.loads(request['messages'][1]['content'])['objects']}
        cases[name] = {'job_id': jobs['jobs'][name]['job_id'], 'uid': selected['uid'],
                       'room_type': selected['room_type'], 'objects': selected['objects'],
                       'api_objects': selected['api_objects'], 'api_calls': len(ledger),
                       'tokens': api['tokens'], 'runtime_seconds': result['runtime_seconds'],
                       'checker_pass': layout['checker_pass'], 'strict_accepted': layout['accepted'],
                       'final_stage': layout['stage'], 'placer_complete': True,
                       'size_preserved_through_S1_and_export': True,
                       'furniture': [{'id': oid, 'category': categories[oid], 'size': size,
                                      'source': estimates['sources'][oid]} for oid, size in sizes.items()]}
    previous = []
    history = jobs.get('history')
    assert isinstance(history, list) and len(history) == 1
    assert (history[0]['case'], history[0]['job_id'], history[0]['output_dir']) == ('no-prior', '7450325', 'no-prior')
    for attempt in history:
        old_status = read(root / attempt['output_dir'] / 'job-status.json')
        assert old_status['job_id'] == attempt['job_id'] and old_status['exit_code'] == 1
        ledger = rows(root / attempt['output_dir'] / 'system/size-api/calls.jsonl')
        assert len(ledger) == attempt['api_calls'] == 1
        assert all(item['uid'] == selection['cases'][attempt['case']]['uid'] for item in ledger)
        previous.append({'case': attempt['case'], 'job_id': attempt['job_id'], 'api_calls': len(ledger),
                         'failure': attempt['failure']})
    for name, case in cases.items():
        assert case['api_calls'] + sum(a['api_calls'] for a in previous if a['case'] == name) <= 6
    total = sum(case['api_calls'] for case in cases.values()) + sum(a['api_calls'] for a in previous)
    assert total <= jobs['total_http_cap'] == 12
    return {'status': 'REAL_SERVER_SMOKE_VERIFIED', 'quality_claim': 'two-room functional test only',
            'total_api_calls': total, 'maximum_api_calls': 12, 'previous_attempts': previous, 'cases': cases}


if __name__ == '__main__':
    root = Path(sys.argv[1]).resolve()
    result = verify(root)
    (root / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
