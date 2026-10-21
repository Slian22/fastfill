"""Probe p36, paid part (<= 100 gpt-6.1-sol calls, to be run by the orchestrator on Isambard as a CPU job).

One call per record of api-requests.jsonl (written by p36_mask_inpaint.py): the rewritten placement request where
the clamped movables are fixed_ boxes and only the masked objects are to be placed. No repair rounds, no retries
beyond llm_agent's transient ones (each retry counts as a call; the hard cap stops the run). Answers are mapped
back to the original ids and scored with the frozen checker (geometry.geometry) and strict admission
(downstream.accept) on the ORIGINAL room, next to the same rooms' dumb-inpaint floor from rows.jsonl.

python -I p36_api_probe.py --requests DIR/api-requests.jsonl --rows DIR/rows.jsonl --env-file ~/.fastfill_api.env \
    --out DIR/api [--cap 100] [--max-output-tokens 8192] [--timeout 600]
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
sys.path.insert(0, '/Users/slian/Desktop/3D/Worldedge/OptiScene')   # orchestrator: point at the Isambard checkout
from p36_mask_inpaint import ROOMS, REQUESTS, S1_ROWS, jsonl, map_back  # noqa: E402

from fastfill.comparison import contract, downstream as d, geometry as g, llm_agent as L, output_repair  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--requests', type=Path, required=True)
    ap.add_argument('--rows', type=Path, required=True)
    ap.add_argument('--env-file', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--cap', type=int, default=100)
    ap.add_argument('--max-output-tokens', type=int, default=8192)
    ap.add_argument('--timeout', type=float, default=600.0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    config = L.load_config(a.env_file)
    if config.model != L.EXPECTED_MODEL:
        raise SystemExit(f'env model {config.model!r} != {L.EXPECTED_MODEL!r}')
    recs = jsonl(a.requests)[:a.cap]
    floor = {r['uid']: r for r in jsonl(a.rows)}
    rooms = {r['uid']: r for r in jsonl(ROOMS)}
    requests = {r['uid']: r for r in jsonl(REQUESTS)}
    s1 = {r['uid']: r for r in jsonl(S1_ROWS) if r['uid'] in floor}
    receipts = a.out / 'receipts.jsonl'
    done = {r['uid'] for r in jsonl(receipts)} if receipts.exists() else set()
    calls = len(done)
    with open(receipts, 'a') as f:
        for rec in recs:
            if rec['uid'] in done:
                continue
            if calls >= a.cap:
                break
            calls += 1
            messages = [*rec['messages'], {'role': 'user', 'content': L.SUFFIX}]
            t0 = time.monotonic()
            try:
                reply = L.call_api(config, messages, a.max_output_tokens, a.timeout)
                row = {'uid': rec['uid'], 'text': reply.get('text'), 'usage': reply.get('usage'), 'error': None}
            except L.ApiError as exc:
                row = {'uid': rec['uid'], 'text': None, 'usage': None, 'error': exc.code}
            f.write(json.dumps({**row, 'latency_s': time.monotonic() - t0}) + '\n'); f.flush()
    # score
    out_rows, n = [], 0
    for rec, r in ((rec, r) for rec in recs for r in jsonl(receipts) if r['uid'] == rec['uid']):
        uid, p = rec['uid'], rec['p36']
        layout = s1[uid]['s1_layout']
        room = output_repair.predicted_room(rooms[uid], requests[uid]['messages'], layout)
        bound_fixed = [{**x, 'yaw': math.radians(x['yaw'])} for x in json.loads(rec['messages'][1]['content'])['fixed']]
        bound = {**rooms[uid], 'boundary': room['boundary'], 'height': room.get('height'), 'fixed': bound_fixed,
                 'objects': [{'id': k, 'category': 'object', 'size': [1, 1, 1], 'pos': [0, 0, 0], 'yaw': 0.0,
                              'parent': None} for k in sorted(p['back_obj'])],   # ids/order only: validate_output
                 contract._MARKER: contract.PROTOCOL}
        res = {'uid': uid, 'n': len(layout), 'masked': len(p['masked']), 'error': r['error'], 'valid': False,
               'pass': False, 'strict': False, 'floor_pass': floor[uid]['pass_after']}
        if r['text']:
            try:
                ans = contract.validate_output(L.enforce_sizes(r['text'], {'messages': rec['messages']})[0], bound, 'placement')
                new = map_back(ans, p['back_obj'], p['back_fixed'], layout)
                geo = g.geometry({**room, 'objects': new})
                res['valid'], res['pass'] = True, geo['geometry_success']
                res['strict'] = bool(d.accept(room, new).get('accepted'))
            except (ValueError, KeyError, TypeError) as exc:
                res['error'] = type(exc).__name__
        out_rows.append(res); n += 1
    k = max(1, n)
    summary = {'rooms': n, 'calls': calls, 'valid': sum(r['valid'] for r in out_rows) / k,
               'checker_pass_api_inpaint': sum(r['pass'] for r in out_rows) / k,
               'strict_api_inpaint': sum(r['strict'] for r in out_rows) / k,
               'checker_pass_dumb_floor_same_rooms': sum(r['floor_pass'] for r in out_rows) / k}
    (a.out / 'scored.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in out_rows))
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
