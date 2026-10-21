"""Score the 600 a021 Arm-A dev-select records exactly like a021_select.test_geometry (placement-geometry rows + cap flags)."""
import json, sys
from pathlib import Path
sys.path.insert(0, '/Users/slian/Desktop/3D/Worldedge/OptiScene')
from fastfill.comparison import evaluate_pair as e, evaluate_llm as ev, report_v2 as r2, geometry as g
from fastfill.comparison import dpo_stages as d
from fastfill.comparison import train_pair_helpers as h

def main():
    H = Path('/Volumes/harddisk/fastfillout/unknownsize-20261013/a025-inputs')
    data, run, out = H / 'dev-select-v1', H / 'a021-dev-select-arm', Path(sys.argv[1])
    manifest = e._verified(data)
    rooms = h._by_uid(h._rows(data, 'rooms.jsonl', manifest['files']['rooms.jsonl']))
    requests = h._by_uid(h._rows(data, 'placement/requests.jsonl', manifest['files']['placement/requests.jsonl']))
    records, receipt = e._inference(run, data, manifest['expected_uids'])
    plain = r2._map(e._score, [(rooms[x['uid']], requests[x['uid']], x, 'placement') for x in records], 8)
    payload = [{**s, **{k: x[k] for k in ev.CAP_FLAGS}} for s, x in zip(plain, records, strict=True)]
    d._publish_dir(out, {'placement-geometry.jsonl': d._jsonl(payload),
                         'summary.json': json.dumps({'arms': {'placement': g.summarize(payload)}, 'run_sha256': receipt['run_sha256']}, indent=1).encode()})
    print(json.dumps(g.summarize(payload)['all'])[:400])


if __name__ == '__main__':
    main()
