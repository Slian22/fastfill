"""Amendment 014: the official OptiScene pipeline (vendored in optiscene_official/, README hyper-parameters,
Qwen2.5-7B-Instruct) trained on Isambard and evaluated with our checker. EXPLORATORY, separate from 005-013.
Every number is labelled "OptiScene code recipe (README, Qwen2.5-7B-Instruct)": not the paper's recipe, not the
authors' weights.

Training venv (stdlib at import; run as a script, PYTHONPATH = code root):
  optiscene_eval.py env-check --data DATA [--model BASE] --out ENV.json    pins, vendored hashes, README DPO argv
                                                                           parse, SFTConfig, data/model sha256; once
  optiscene_eval.py train --stage sft|dpo1|dpo2 --run OUT --data DATA --base BASE --env ENV.json [--prev-job ID]
          one job of a stage: refuses unless job ID ended COMPLETED or TIMEOUT, exits 0 when the stage is complete,
          one visible GPU only (no DataParallel); DPO resumes from its last checkpoint (F3), SFT cannot resume
  optiscene_eval.py merge --stage S --run OUT --base BASE --env ENV.json
          checkpoint rule (last checkpoint = expected step = trainer_state global_step = max_steps; DPO adapter sha256 =
          the save_model copy), their merge_lora on CPU, published write-once to OUT/S-merged with a receipt
Our env / the vLLM env (python -m fastfill.comparison.optiscene_eval ...):
  plan-e1 --hf HF --data DATA --out PLAN      local CPU, uploaded with its MANIFEST. E1-unseen: HF scenes that no
          training row (SFT + chosen DPO1/DPO2) covers >= 50 % on (description, x, y) at 0.1 m, valid floor and finite
          poses, whose rendered input is no training input (SFT or either DPO prompt), sha256(seed, uid) rank, capped
          at the paper's per-stratum counts (shortfalls recorded, never filled). E1-heldout20: their own 20 % SFT eval
          rows (numpy default_rng(42) = datasets' split) not covered >= 50 % by any 80 % training row of SFT/DPO1/DPO2
          and whose input is no 80 % training input (no floor: parse + OOR only). GT gate on the plan's Holodeck rooms.
          hf-coverage.json: every HF scene's best training-row coverage (read by plan-e2).
  plan-e2 --eval-data RUN/eval-data-v1 --data DATA --e1 PLAN_E1 --out OUT/e2 [--limit 20]
          our test rooms in eval_data.PAPER_STRATA (count = gpt-selection populations), their input in gt and prior
          size modes, FastFill v1 (scene.messages) requests for the 8B, and an OptiScene training-exposure audit
          (the 8B's own training exposure is not audited: reported as unknown)
  generate --plan PLAN --prompt sft|reasoning|v1 --model DIR --out GEN [--gate-parse 0.9]
          vLLM greedy, seed 0, chat template of the model dir (default system turn); OptiScene max_tokens 2048
  score --e1 PLAN_E1 --e2 OUT/e2 --gen OUT/gen --eval-data RUN/eval-data-v1 --p-geometry RUN/geometry-v5
        --p-run SELECTED_P_INFER --out OUT/score    E1 Table-6 analogue, E2 (gt/prior x full/floor x all/unseen), P
        rows with the amendment-005 cap rule, paired OptiScene(prior) - P
Amendment 015 (EXPLORATORY), E3 = our test rooms whose room_families.family is not a paper family (bedroom, living,
kitchen, bathroom) nor none/other, at most 100 per family by sha256('optiscene-015-e3:' + uid):
  plan-e3 --eval-data RUN/eval-data-v1 --data DATA --e1 PLAN_E1 --out OUT15/e3    plan-e2's requests and exposure audit
  score-e3 --e3 OUT15/e3 --gen OUT15/gen --eval-data ... --p-geometry ... --p-run ... --out OUT15/score-e3
          every published E3_GENS arm and P per family and pooled, paired OptiScene(prior) - P per family and pooled
          (all and unseen, report_v2 small-sample flags); all of it again on rooms with <= 14 objects (SFT maximum)

Frames. Their text frame T (gprompt): x_T = raw.x, y_T = raw.z, z_T = raw.y, angle a = rad(rotation.y), box (h, w, d)
= extents along mesh (y, x, z), origin = floor area centroid. IR: pos = (x_T + c_x, y_T + c_y, z_T), size = (d, w, h),
yaw = -a - pi/2 (adapters/optiscene_holodeck.py: LH->RH, mesh +X at -a, front = mesh -Z); c = IR boundary centroid
(E1: 0, the boundary is their floor). The front holds for Holodeck only: 3D-FRONT scenes come out turned by pi (95 % of
their wall-backed pieces face the wall). Footprints, geometry and downstream acceptance do not change under +pi, but
the converted yaw of 3D-FRONT-style rooms and outputs must not be used for orientation analysis or renders. on = the movable whose top is within 0.05 m of z (bottom strictly lower) and
whose footprint contains (x, y), else the floor. oor_paper (App. D.2): unrotated w along x, d along y; oor: our
validate.oor (size[0] = d along x). OOR averages complete rooms only; every rate uses all requests.
"""
import argparse
import collections
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

LABEL = 'OptiScene code recipe (README, Qwen2.5-7B-Instruct)'
SEED = 'optiscene-014-20261014'
UPSTREAM_COMMIT = '409f1317e1a8afe852f4cdc99dfefaed30aeb425'
VENDORED = Path(__file__).resolve().parent / 'optiscene_official'
DATA_FILES = {'sft': ('sft_prompts.json', '12b3ccc10e8e'), 'dpo1': ('dpo_prompts.json', '0d331989'),
              'dpo2': ('dpo2_prompts.json', 'ed6c1bae')}
INSTRUCTION = (4485, '541055463398')  # SFT training instruction: length, sha256 prefix (not HF example.py's).
PINS = {'trl': '0.15.2', 'transformers': '4.49.0', 'peft': '0.14.0', 'accelerate': '1.4.0', 'datasets': '3.3.2',
        'modelscope': '1.23.2'}
STEPS = {'sft': 1226, 'dpo1': 15900, 'dpo2': 16820}
PREVIOUS = {'dpo1': 'sft', 'dpo2': 'dpo1'}
OPS_ARGV = ['--save_total_limit', '2', '--report_to', 'none']  # Operational only, no effect on optimisation.
FIXES = {'F1': 'argv: dataset/dpo1_prompts.json -> dpo_prompts.json (the released name)',
         'F2': 'main.py: OptiSceneScriptArguments(dataset_name default, dataset_file) in the DPO TrlParser',
         'F3': 'dpo_train.py: trainer.train(resume_from_checkpoint=get_last_checkpoint(output_dir))',
         'F5': 'sft_train.py: gradient_checkpointing=True in SFTConfig (memory only, same gradients)',
         'ops': '--save_total_limit 2 --report_to none (DPO argv)'}
KEPT = ('main.py remaps --eval_strategy to --evaluation_strategy (works on transformers 4.49)',
        'DPO policy loads in fp32 (ModelConfig.torch_dtype None); --bf16 only enables autocast',
        'DPO: linear schedule, no warmup, beta 0.1 (DPOConfig defaults)',
        'SFT: max_seq_length 2048 truncates about 3,965 long rows; cosine, warmup 0.1 (main.py defaults)',
        'SFT: model.to("cuda") on one GPU; no save_model, checkpoint-1226 comes from save_strategy epoch',
        'SFT: pad_token = eos_token (<|im_end|>), so <|im_end|> is masked out of the labels; the loss covers the prompt',
        'DPO: get_bin_dataset pairs the chosen prompt with both outputs; the rejected row\'s own input differs in 4,325 '
        'of 7,951 DPO1 pairs (4,317 box sizes only) and 813 of 8,410 DPO2 pairs (box sizes only)')
PAPER_E1 = {'3DFRONT/bedroom': 423, '3DFRONT/living room': 53, 'HOLODECK/bedroom': 50,
            'HOLODECK/living room': 50, 'HOLODECK/kitchen': 50, 'HOLODECK/bathroom': 50}
COVER = 0.5
GATE = {'holodeck_gt_geometry_success_min': .95, 'holodeck_swap_geometry_success_max': .05}
MAX_TOKENS, MAX_MODEL_LEN = 2048, 16384
TOL = 0.05
PARSE_ERRORS = ('no_design', 'bad_json', 'nonfinite', 'missing_group', 'extra_group', 'count_mismatch')
GENS = (('sft', 'e1', 'sft'), ('sft', 'e2', 'sft'), ('dpo1', 'e1', 'sft'), ('dpo1', 'e2', 'sft'),
        ('dpo2', 'e1', 'sft'), ('dpo2', 'e2', 'sft'), ('dpo2', 'e1', 'reasoning'), ('8b', 'e2', 'v1'))
EIGHT_B = {'architectures': ['Qwen3ForCausalLM'], 'num_hidden_layers': 36}
EIGHT_B_EXPOSURE = ('unknown: the training rooms of OptiScene-Qwen-8B-Final (FastFill v1.0) were not audited against '
                    'eval-data-v1; "unseen" means unseen by the OptiScene release only')
README_SAMPLING = ('not used: the README inference samples with generation_config (temperature 0.7, top-p 0.8, top-k 20, '
                   'repetition penalty 1.05); amendment 014 decodes greedily')
WAITING = (None, 'PENDING', 'RUNNING', 'COMPLETING')  # sacct can still show these just after an afterany start
E3_SEED, E3_CAP = 'optiscene-015-e3:', 100
E3_GENS = ('sft-e3-sft', 'dpo1-e3-sft', 'dpo2-e3-sft', '8b-e3-v1')
TRAIN_MAX_OBJECTS = 14  # sft_prompts.json: median 7, max 14 objects per room (dpo: max 17; dpo2: median 8, max 32)


# ---- io ------------------------------------------------------------------------------------------------------------

def _sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _dumps(value, **kw):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, **kw)


def _jsonl_bytes(rows):
    return ''.join(_dumps(r) + '\n' for r in rows).encode('utf-8')


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def _refuse(out):
    if os.path.lexists(out):
        raise FileExistsError(f'refusing existing output: {out}')


def _hash_tree(root, skip=('MANIFEST.json',)):
    root = Path(root)
    return {p.relative_to(root).as_posix(): _sha(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p.relative_to(root).as_posix() not in skip and '.cache' not in p.relative_to(root).parts}


def _finish(dest, out, meta):
    """MANIFEST.json (meta + every staged file's sha256) last, then a no-replace publish of the directory."""
    from fastfill.comparison.corpus import _publish
    manifest = {**meta, 'label': LABEL, 'files': _hash_tree(dest)}
    (Path(dest) / 'MANIFEST.json').write_text(_dumps(manifest, indent=2) + '\n')
    _publish(dest, out)
    return manifest


def _publish_files(out, files, meta):
    out = Path(out)
    _refuse(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.' + out.name + '-', dir=out.parent) as tmp:
        for name, payload in files.items():
            (Path(tmp) / name).parent.mkdir(parents=True, exist_ok=True)
            (Path(tmp) / name).write_bytes(payload)
        return _finish(tmp, out, meta)


def _verified(root):
    """MANIFEST.json of a published directory with every listed file rehashed; returns the manifest."""
    root = Path(root)
    manifest = json.loads((root / 'MANIFEST.json').read_text())
    files = manifest.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError(f'empty manifest: {root}')
    for name, digest in files.items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError(f'invalid manifest path: {name}')
        digest = digest.get('sha256') if isinstance(digest, dict) else digest
        if not (root / name).is_file() or _sha(root / name) != digest:
            raise ValueError(f'hash mismatch: {root / name}')
    return manifest


def _rank(*parts):
    return hashlib.sha256('\0'.join((SEED,) + parts).encode('utf-8')).hexdigest()


def _code():
    return {p.name: _sha(p) for p in (Path(__file__), VENDORED / 'prompt.py', VENDORED / 'UPSTREAM.json')}


# ---- their data ----------------------------------------------------------------------------------------------------

def official(data):
    """The three released files, each checked against its sha256 prefix: {stage: (rows, sha256)}."""
    out = {}
    for stage, (name, prefix) in DATA_FILES.items():
        digest = _sha(Path(data) / name)
        if not digest.startswith(prefix):
            raise ValueError(f'{name}: sha256 {digest[:12]} is not {prefix}')
        out[stage] = (json.loads((Path(data) / name).read_text(encoding='utf-8')), digest)
    return out


def chosen(rows, stage):
    return rows if stage == 'sft' else [p['combined_prompts'][0] for p in rows]


def instruction(sft_rows):
    texts = {r['instruction'] for r in sft_rows}
    text = texts.pop() if len(texts) == 1 else ''
    if (len(text), hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]) != INSTRUCTION:
        raise ValueError('SFT instruction is not the 4,485-character training instruction')
    return text


def floor_plan_prompts():
    """README inference prefix (vendored prompt.py, a constant only)."""
    scope = {}
    exec(compile((VENDORED / 'prompt.py').read_text(encoding='utf-8'), 'prompt.py', 'exec'), scope)
    return scope['floor_plan_prompts']


def split(n, seed=42, test_size=0.2):
    """datasets.Dataset.train_test_split(test_size, seed): (train, test) index lists."""
    import numpy as np
    k = math.ceil(test_size * n)
    p = np.random.default_rng(seed).permutation(n)
    return p[k:].tolist(), p[:k].tolist()


_TAG = {t: re.compile(rf'\[{re.escape(t)}\]\n(.*?)\n\[/{re.escape(t)}\]', re.S)
        for t in ('Task Room Type', 'Task Room Area', 'Task Objects and Bounding Box Size')}


def render_input(room_type, area, groups):
    """Their SFT input layout: groups [(description, [(h, w, d), ...])], boxes already rounded to 2 decimals."""
    body = ',\n'.join(json.dumps({'object': f'{len(boxes)} {desc}',
                                  'bounding box': [{'h': h, 'w': w, 'd': d} for h, w, d in boxes]}, indent=4)
                      for desc, boxes in groups)
    return (f'[Task Room Type]\n{room_type}\n[/Task Room Type]\n\n[Task Room Area]\n{round(area, 2)}\n'
            f'[/Task Room Area]\n\n[Task Objects and Bounding Box Size]\n{{"objects":\n[\n{body}\n]\n}}\n'
            '[/Task Objects and Bounding Box Size]\n\n')


def parse_input(text):
    """-> (room type, area, groups [[name, n]], boxes [[h, w, d]] in group order)."""
    room_type = _TAG['Task Room Type'].search(text).group(1)
    area = float(_TAG['Task Room Area'].search(text).group(1))
    groups, boxes = [], []
    for g in json.loads(_TAG['Task Objects and Bounding Box Size'].search(text).group(1))['objects']:
        n = int(g['object'].split(' ', 1)[0])
        if n != len(g['bounding box']):
            raise ValueError('group count differs from its boxes')
        groups.append([g['object'], n])
        boxes += [[b['h'], b['w'], b['d']] for b in g['bounding box']]
    return room_type, area, groups, boxes


def inputs(rows, stage):
    """Every input text of training records (DPO: both prompts of a pair)."""
    return {r['input'] for r in rows} if stage == 'sft' else {c['input'] for p in rows for c in p['combined_prompts']}


def design_keys(text):
    """Instance keys (description, x, y) at 0.1 m of the first [Design] block (the verified matching rule)."""
    d = json.loads(re.search(r'\[Design\]\s*(.*?)\s*\[/Design\]', text, re.S).group(1))
    return {(re.sub(r'^\d+\s+', '', x['object']).strip().lower(), round(float(c['x']), 1), round(float(c['y']), 1))
            for x in d['objects'] for c in x['coordinates']}


def _keys_or_empty(text, keys=design_keys):
    try:
        return keys(text)
    except (ValueError, KeyError, TypeError, AttributeError):
        return set()


class Coverage:
    """Best fraction of a key set that a single training row covers."""
    def __init__(self, key_sets):
        self.index = collections.defaultdict(set)
        for k, keys in enumerate(key_sets):
            for t in keys:
                self.index[t].add(k)

    def best(self, keys):
        hits = collections.Counter(k for t in keys for k in self.index.get(t, ()))
        return max(hits.values()) / len(keys) if hits and keys else 0.0


# ---- frames, parsing, OOR ------------------------------------------------------------------------------------------

def t_to_ir(oid, category, x, y, z, angle, box, center=(0.0, 0.0)):
    h, w, d = box
    return {'id': oid, 'category': category, 'size': [d, w, h], 'pos': [x + center[0], y + center[1], z],
            'yaw': (-angle - math.pi / 2) % math.tau, 'parent': None}


def ir_to_t(obj, center=(0.0, 0.0)):
    sx, sy, sz = obj['size']
    return {'x': obj['pos'][0] - center[0], 'y': obj['pos'][1] - center[1], 'z': obj['pos'][2],
            'angle': (-obj['yaw'] - math.pi / 2) % math.tau, 'box': [sz, sy, sx]}


def _contains(o, xy):
    c, s = math.cos(o['yaw']), math.sin(o['yaw'])
    dx, dy = xy[0] - o['pos'][0], xy[1] - o['pos'][1]
    return abs(c * dx + s * dy) <= o['size'][0] / 2 + 1e-9 and abs(-s * dx + c * dy) <= o['size'][1] / 2 + 1e-9


def derive_on(objs, tol=TOL):
    """Parent ids: |z| <= tol is the floor, else the closest-top movable under (x, y) with a strictly lower bottom."""
    parents = []
    for o in objs:
        z, best = o['pos'][2], None
        if abs(z) > tol:
            cands = [(abs(p['pos'][2] + p['size'][2] - z), k) for k, p in enumerate(objs)
                     if p is not o and p['pos'][2] < z and abs(p['pos'][2] + p['size'][2] - z) <= tol + 1e-9
                     and _contains(p, o['pos'])]  # 1e-9 = validate.SUPPORT_EPS
            best = objs[min(cands)[1]]['id'] if cands else None
        parents.append(best)
    return parents


def oor(rects):
    """sum_{i<j} |B_i n B_j| / sum |B_i| for axis-aligned (x, y, extent_x, extent_y)."""
    total = sum(ex * ey for _, _, ex, ey in rects)
    inter = 0.0
    for i, (xa, ya, wa, ha) in enumerate(rects):
        for xb, yb, wb, hb in rects[i + 1:]:
            ix = min(xa + wa / 2, xb + wb / 2) - max(xa - wa / 2, xb - wb / 2)
            iy = min(ya + ha / 2, yb + hb / 2) - max(ya - ha / 2, yb - hb / 2)
            inter += max(ix, 0.0) * max(iy, 0.0)
    return inter / total if total > 0 else 0.0


def oor_paper(objs):
    return oor([(o['pos'][0], o['pos'][1], o['size'][1], o['size'][0]) for o in objs])  # w along x, d along y


def oor_ours(objs):
    return oor([(o['pos'][0], o['pos'][1], o['size'][0], o['size'][1]) for o in objs])  # = validate.check()['oor']


class _NonFinite(ValueError):
    pass


def _reject(token):
    raise _NonFinite(token)


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _finite(v):
    try:
        return _num(v) and math.isfinite(float(v))
    except OverflowError:
        return False


DESIGN = re.compile(r'\[Design\](.*?)\[/Design\]', re.S)


def parse_design(text, groups):
    """Last [Design] block, strict json.loads -> ([(x, y, z, angle)] in box order, None) or (None, code)."""
    blocks = DESIGN.findall(text) if isinstance(text, str) else []
    if not blocks:
        return None, 'no_design'
    try:
        value = json.loads(blocks[-1], parse_constant=_reject)
    except _NonFinite:
        return None, 'nonfinite'
    except (ValueError, RecursionError):
        return None, 'bad_json'
    items = value.get('objects') if isinstance(value, dict) else None
    if not isinstance(items, list) or not all(isinstance(i, dict) and isinstance(i.get('object'), str) for i in items):
        return None, 'bad_json'
    names, expected = [i['object'] for i in items], dict(groups)
    if set(expected) - set(names):
        return None, 'missing_group'
    if len(names) != len(set(names)) or set(names) - set(expected):
        return None, 'extra_group'
    by, out = {i['object']: i for i in items}, []
    for name, n in groups:
        coords, rots = by[name].get('coordinates'), by[name].get('rotate')
        if not isinstance(coords, list) or not isinstance(rots, list):
            return None, 'bad_json'
        if len(coords) != n or len(rots) != n:
            return None, 'count_mismatch'
        for c, r in zip(coords, rots):
            if not isinstance(c, dict) or not isinstance(r, dict):
                return None, 'bad_json'
            values = [c.get('x'), c.get('y'), c.get('z', 0), r.get('angle')]
            if not all(_num(v) for v in values):
                return None, 'bad_json'
            if not all(_finite(v) for v in values):
                return None, 'nonfinite'
            out.append(tuple(float(v) for v in values))
    return out, None


# ---- E1: their scenes ----------------------------------------------------------------------------------------------

def polygon_area(points):
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]))) / 2


def _category(o):
    return re.sub(r'-\d+$', '', o.get('object_name') or 'object')


def scene_room(scene):
    """HF scene -> (IR GT room, groups [[name, n]], boxes [[h, w, d]], input text) in gprompt's box order."""
    from shapely.geometry import Polygon
    floor = scene.get('floor')
    if not isinstance(floor, list) or len(floor) < 3 or not all(
            isinstance(p, (list, tuple)) and len(p) >= 2 and all(_finite(v) for v in p[:2]) for p in floor):
        raise ValueError('bad_floor')
    ring = [[float(p[0]), float(p[1])] for p in floor]
    poly = Polygon(ring)
    if not poly.is_valid or poly.area <= 0:
        raise ValueError('bad_floor')
    if not poly.exterior.is_ccw:
        ring = ring[::-1]
    objects = scene.get('objects') or []
    if not objects:
        raise ValueError('no_objects')
    for o in objects:
        b, p = o.get('bbox'), o.get('position') or {}
        if not isinstance(b, list) or len(b) != 3 or not all(_finite(v) and round(float(v) / 100, 2) > 0 for v in b):
            raise ValueError('bad_bbox')
        if not all(_finite(p.get(k)) for k in 'xyz') or not _finite((o.get('rotation') or {}).get('y')):
            raise ValueError('nonfinite_pose')
    groups = {}
    for k, o in enumerate(objects):
        desc = re.sub(r'(?<![a-zA-Z])a ', '', o.get('description') or o.get('object_name', 'object'))
        groups.setdefault(desc, []).append(k)
    order = [k for ks in groups.values() for k in ks]
    boxes = [[round(float(v) / 100, 2) for v in objects[k]['bbox']] for k in order]
    objs = [t_to_ir(f'o{i}', _category(objects[k]), objects[k]['position']['x'], objects[k]['position']['z'],
                    objects[k]['position']['y'], math.radians(objects[k]['rotation']['y']), box)
            for i, (k, box) in enumerate(zip(order, boxes))]
    for o, parent in zip(objs, derive_on(objs)):
        o['parent'] = parent
    room = {'room_type': scene.get('room'), 'boundary': ring, 'boundary_type': 'polygon', 'height': None,
            'fixed': [], 'objects': objs}
    named, at = [], 0
    for desc, ks in groups.items():
        named.append((desc, boxes[at:at + len(ks)]))
        at += len(ks)
    return room, [[f'{len(ks)} {desc}', len(ks)] for desc, ks in groups.items()], boxes, \
        render_input(scene.get('room'), polygon_area(ring), named)


def layout_from(coords, boxes, room=None, ids=None, center=(0.0, 0.0)):
    """Parsed T coordinates + boxes -> IR objects (categories/ids from the room) with derived supports."""
    objs = [t_to_ir(ids[k] if ids else f'o{k}', room['objects'][k]['category'] if room else 'object',
                    x, y, z, a, box, center) for k, ((x, y, z, a), box) in enumerate(zip(coords, boxes))]
    for o, parent in zip(objs, derive_on(objs)):
        o['parent'] = parent
    return objs


def room_metrics(room, objs):
    """Geometry (frozen checker), downstream acceptance and both OOR variants of a full layout in its room."""
    from fastfill.comparison import downstream, geometry
    pred = {**room, 'objects': objs}
    geo = geometry.geometry(pred)
    return {'geometry_valid': geo['geometry_valid'], 'geometry_success': geo['geometry_success'], 'oor': geo['oor'],
            'oor_paper': oor_paper(objs), 'downstream_accept': downstream.accept(pred, objs)['accepted']}


def swapped(objs):
    return [{**o, 'size': [o['size'][1], o['size'][0], o['size'][2]]} for o in objs]


def select_strata(candidates, caps):
    """{stratum: [uid]} -> ({stratum: ranked selected uids}, receipt); shortfalls are recorded, never filled."""
    selected, receipt = {}, {}
    for stratum, cap in caps.items():
        ranked = sorted(candidates.get(stratum, ()), key=lambda u: (_rank(u), u))
        selected[stratum] = ranked[:cap]
        receipt[stratum] = {'population': len(ranked), 'cap': cap, 'selected': len(selected[stratum]),
                            'shortfall': cap - len(selected[stratum])}
    receipt['_outside_caps'] = {s: len(v) for s, v in sorted(candidates.items()) if s not in caps}
    return selected, receipt


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def plan_e1(hf, data, out):
    hf, out = Path(hf), Path(out)
    _refuse(out)
    rows = official(data)
    sft = rows['sft'][0]
    instr, prefix = instruction(sft), floor_plan_prompts()
    seen_by = Coverage([_keys_or_empty(r['output']) for stage in DATA_FILES for r in chosen(rows[stage][0], stage)])
    seen_inputs = set().union(*(inputs(rows[stage][0], stage) for stage in DATA_FILES))
    hf_rows = json.loads((hf / 'prompts_all_scenes.json').read_text(encoding='utf-8'))
    unseen, candidates, built, excluded = collections.Counter(), collections.defaultdict(list), {}, []
    coverage = {}
    for h in hf_rows:
        uid = h['uid']
        source, name = uid.split('-', 1)
        coverage[uid] = seen_by.best(design_keys(h['output']))
        if coverage[uid] >= COVER:
            continue
        scene = json.loads((hf / 'scenes' / (name + '.json')).read_text(encoding='utf-8'))
        if scene.get('source') != source:
            raise ValueError(f'{uid}: scene source {scene.get("source")} differs')
        stratum = f"{source}/{scene.get('room')}"
        unseen[stratum] += 1
        try:
            built[uid] = (stratum, *scene_room(scene))
        except ValueError as exc:
            excluded.append({'uid': uid, 'stratum': stratum, 'reason': str(exc)})
            continue
        if built[uid][-1] in seen_inputs:  # Their exact prompt was trained on, whatever the layout coverage.
            excluded.append({'uid': uid, 'stratum': stratum, 'reason': 'seen_input'})
            continue
        candidates[stratum].append(uid)
    selected, strata = select_strata(candidates, PAPER_E1)
    plan, gate_rows = [], collections.defaultdict(list)
    for stratum, uids in selected.items():
        for uid in uids:
            _, room, groups, boxes, text = built[uid]
            swap = room_metrics(room, swapped(room['objects']))
            gt = {**room_metrics(room, room['objects']), 'swap_geometry_success': swap['geometry_success'],
                  'swap_geometry_valid': swap['geometry_valid']}
            gate_rows[stratum].append(gt)
            plan.append({'key': 'e1-unseen:' + uid, 'uid': uid, 'population': 'e1-unseen', 'stratum': stratum,
                         'groups': groups, 'boxes': boxes, 'room': room, 'gt': gt,
                         'messages': {'sft': [{'role': 'user', 'content': instr + text}],
                                      'reasoning': [{'role': 'user', 'content': prefix + text}]}})
    train80, inputs80 = [], set()
    for stage in DATA_FILES:
        rs, idx = chosen(rows[stage][0], stage), split(len(rows[stage][0]))[0]
        train80 += [_keys_or_empty(rs[i]['output']) for i in idx]
        inputs80 |= inputs([rows[stage][0][i] for i in idx], stage)
    covered80, heldout = Coverage(train80), collections.Counter()
    test = split(len(sft))[1]
    for i in sorted(test):
        r = sft[i]
        if covered80.best(_keys_or_empty(r['output'])) >= COVER:
            continue
        if r['input'] in inputs80:
            excluded.append({'uid': r['uid'], 'stratum': 'e1-heldout20', 'reason': 'seen_input'})
            continue
        try:
            room_type, _, groups, boxes = parse_input(r['input'])
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            excluded.append({'uid': r['uid'], 'stratum': 'e1-heldout20', 'reason': 'bad_input: ' + type(exc).__name__})
            continue
        coords, err = parse_design(r['output'], groups)
        objs = layout_from(coords, boxes) if coords else None
        stratum = f"{r['uid'].split('-')[0]}/{room_type}"
        heldout[stratum] += 1
        plan.append({'key': 'e1-heldout20:' + r['uid'], 'uid': r['uid'], 'population': 'e1-heldout20',
                     'stratum': stratum, 'groups': groups, 'boxes': boxes,
                     'gt': {'parse_error': err, 'oor_paper': oor_paper(objs) if objs else None,
                            'oor': oor_ours(objs) if objs else None},
                     'messages': {'sft': [{'role': 'user', 'content': instr + r['input']}],
                                  'reasoning': [{'role': 'user', 'content': prefix + r['input']}]}})
    gt = {s: {'rooms': len(v), **{k: _mean([g[k] for g in v]) for k in
                                  ('geometry_valid', 'geometry_success', 'oor', 'oor_paper', 'downstream_accept',
                                   'swap_geometry_success', 'swap_geometry_valid')}} for s, v in gate_rows.items()}
    holo = [g for s, v in gate_rows.items() if s.startswith('HOLODECK/') for g in v]
    gate = {'holodeck_rooms': len(holo), **{k: _mean([g[k] for g in holo]) for k in (
        'geometry_success', 'swap_geometry_success', 'swap_geometry_valid')}, **GATE}
    if not holo or gate['geometry_success'] < GATE['holodeck_gt_geometry_success_min'] \
            or gate['swap_geometry_success'] > GATE['holodeck_swap_geometry_success_max']:
        raise ValueError(f'GT conversion gate failed, nothing published: {gate}')
    held = [r for r in plan if r['population'] == 'e1-heldout20']
    hf_hash = _sha(hf / 'prompts_all_scenes.json')
    meta = {'protocol': 'amendment-014', 'plan': 'e1', 'seed': SEED, 'cover_threshold': COVER,
            'seen_input_rule': 'exact input text of any SFT row or either DPO prompt (heldout20: of their 80 % split)',
            'inputs': {'prompts_all_scenes.json': hf_hash, **{DATA_FILES[s][0]: rows[s][1] for s in DATA_FILES}},
            'implementation_sha256': _code(), 'unseen_by_stratum': dict(sorted(unseen.items())), 'strata': strata,
            'heldout20': {'sft_eval_rows': len(test), 'kept': sum(heldout.values()), 'by_stratum': dict(heldout)},
            'excluded': excluded, 'gt_by_stratum': gt, 'gate': gate, 'gt_heldout20': {
                'parse_errors': collections.Counter(r['gt']['parse_error'] for r in held if r['gt']['parse_error']),
                **{k: _mean([r['gt'][k] for r in held]) for k in ('oor_paper', 'oor')}},
            'counts': collections.Counter(r['population'] for r in plan)}
    return _publish_files(out, {'requests.jsonl': _jsonl_bytes(plan),
                                'hf-coverage.json': (_dumps(coverage, sort_keys=True) + '\n').encode('utf-8')}, meta)


# ---- E2: our paper strata ------------------------------------------------------------------------------------------

def _box(v):
    return max(0.01, round(float(v), 2))


def optiscene_request(room, sizes, instr):
    """Our bound room -> their input: groups = normalized categories in bound order, box h = s_z, w = s_y, d = s_x."""
    from shapely.geometry import Polygon
    from fastfill import scene
    poly = Polygon(room['boundary'])
    groups = {}
    for o in room['objects']:
        groups.setdefault(scene.norm_cat(o.get('category')) or 'object', []).append(o['id'])
    ids = [i for members in groups.values() for i in members]
    boxes = [[_box(sizes[i][2]), _box(sizes[i][1]), _box(sizes[i][0])] for i in ids]
    named, at = [], 0
    for desc, members in groups.items():
        named.append((desc, boxes[at:at + len(members)]))
        at += len(members)
    text = render_input(scene.norm_room_type(room.get('room_type')), poly.area, named)
    return {'groups': [[f'{len(m)} {d}', len(m)] for d, m in groups.items()], 'ids': ids, 'boxes': boxes,
            'center': [poly.centroid.x, poly.centroid.y], 'messages': [{'role': 'user', 'content': instr + text}]}


def v1_request(room, sizes):
    """FastFill v1 (OptiScene-Qwen-8B-Final) request: objects tagged _bound, no position, scene.canonical order."""
    from fastfill import scene
    tag = lambda o: {k: o[k] for k in ('id', 'category', 'size', 'pos', 'yaw') if k in o}
    fixed = [{**tag(f), '_bound': f['id']} for f in room.get('fixed') or []]
    objs = [{'id': o['id'], '_bound': o['id'], 'category': o.get('category'), 'size': list(sizes[o['id']])}
            for o in room['objects']]
    can = scene.canonical({'room_type': room.get('room_type'), 'boundary': room['boundary'],
                           'boundary_type': room['boundary_type'], 'height': room.get('height'),
                           'fixed': fixed, 'objects': objs})
    return {'id_map': {o['id']: o['_bound'] for o in can['objects'] + (can.get('fixed') or [])},
            'order': [o['id'] for o in can['objects']],
            'messages': scene.messages(can, with_desc=False, with_target=False)}


def _q(v):
    """0.1 m key, snapped at 1e-6 first: float noise (1.5499999999999998) must not cross a .x5 boundary (1.55)."""
    return round(round(v, 6), 1)


def holodeck_keys(objs, center):
    return {(_q(o['pos'][0] - center[0]), _q(o['pos'][1] - center[1]), _q(o['size'][1]), _q(o['size'][0]))
            for o in objs}


def _training_box_keys(row):
    """(x, y, w, d) at 0.1 m of a training row: input box k <-> output coordinate k per group."""
    _, _, groups, boxes = parse_input(row['input'])
    coords, err = parse_design(row['output'], groups)
    if err:
        raise ValueError(err)
    return {(_q(x), _q(y), _q(b[1]), _q(b[2])) for (x, y, _, _), b in zip(coords, boxes)}


def exposure_index(rows):
    houses = set()
    for stage in DATA_FILES:
        records = rows[stage][0] if stage == 'sft' else [c for p in rows[stage][0] for c in p['combined_prompts']]
        houses |= {m.group(1).lower() for r in records
                   for m in [re.match(r'3DFRONT-([0-9a-fA-F-]{36})', r['uid'])] if m}
    holo = Coverage([_keys_or_empty(r, _training_box_keys) for stage in DATA_FILES
                     for r in chosen(rows[stage][0], stage) if r['uid'].startswith('HOLODECK')])
    return houses, holo


def room_requests(room, prior, instr, population, stratum):
    """One room's requests (plan-e2 and plan-e3): their input and the FastFill v1 request, gt and prior sizes."""
    uid, rows = room['uid'], []
    for mode, size in (('gt', {o['id']: o['size'] for o in room['objects']}), ('prior', prior)):
        base = {'uid': uid, 'population': population, 'mode': mode, 'stratum': stratum, 'source': room['source']}
        req = optiscene_request(room, size, instr)
        rows.append({**base, 'key': f'{uid}|{mode}|optiscene', 'format': 'optiscene',
                     **{k: req[k] for k in ('groups', 'ids', 'boxes', 'center')}, 'messages': {'sft': req['messages']}})
        v1 = v1_request(room, size)
        rows.append({**base, 'key': f'{uid}|{mode}|v1', 'format': 'v1', 'id_map': v1['id_map'],
                     'order': v1['order'], 'messages': {'v1': v1['messages']}})
    return rows


def room_exposure(room, instr, sidecar, houses, holo, hf_cover):
    """OptiScene training exposure of one room: 3D-FRONT house rule, else Holodeck xywd / HF scene coverage."""
    uid = room['uid']
    if room['source'] == 'IL3D_3dfront':
        group = (sidecar.get(uid) or {}).get('original_group') or ''
        house = group.split(':', 1)[1].lower() if group.startswith('3dfront:') else None
        return {'rule': 'house', 'house': house, 'seen': None if house is None else house in houses}
    center = optiscene_request(room, {o['id']: o['size'] for o in room['objects']}, instr)['center']
    best = holo.best(holodeck_keys(room['objects'], center))
    hf = hf_cover.get(uid.split(':', 1)[-1])  # None: no HF prompt row (Structured3D uids have no ':'), xywd decides
    return {'rule': 'xywd_0.1m_cover_0.5 or hf_scene_cover_0.5', 'coverage': best, 'hf_scene_coverage': hf,
            'seen': max(best, hf or 0.0) >= COVER}


def plan_inputs(eval_data, e1):
    """Verified eval-data rooms, prior sizes, inventory sidecar and plan-e1's HF scene coverage (plan-e2/e3)."""
    manifest = _verified(eval_data)
    if _verified(e1).get('plan') != 'e1':
        raise ValueError(f'{e1} is not a plan-e1 output')
    hf_cover = json.loads((e1 / 'hf-coverage.json').read_text())
    rooms = _jsonl(eval_data / 'rooms.jsonl')
    sizes = {r['uid']: r['estimated_sizes'] for r in _jsonl(eval_data / 'size-estimates.jsonl')}
    sidecar = json.loads((eval_data / 'inventory-sidecar.json').read_text()) \
        if 'inventory-sidecar.json' in manifest['files'] else {}
    return rooms, sizes, sidecar, hf_cover


def plan_e2(eval_data, data, e1, out, limit=None):
    from fastfill import scene
    from fastfill.comparison.eval_data import PAPER_STRATA
    eval_data, e1, out = Path(eval_data), Path(e1), Path(out)
    _refuse(out)
    rooms, sizes, sidecar, hf_cover = plan_inputs(eval_data, e1)
    stratum = lambda r: f"{r['source']}/{scene.norm_room_type(r.get('room_type'))}"
    wanted = {f'{s}/{k}' for s, k in PAPER_STRATA}
    picked = [r for r in rooms if stratum(r) in wanted]
    counts = collections.Counter(stratum(r) for r in picked)
    if limit is not None:
        picked = sorted(picked, key=lambda r: (_rank(r['uid']), r['uid']))[:limit]
    else:
        selection = json.loads((eval_data / 'gpt-selection.json').read_text())
        expected = {k: v['population'] for k, v in selection['strata'].items() if v['population']}
        if dict(counts) != expected:
            raise ValueError(f'paper-strata rooms {dict(counts)} differ from gpt-selection populations {expected}')
    rows = official(data)
    instr = instruction(rows['sft'][0])
    houses, holo = exposure_index(rows)
    plan, exposure = [], {}
    for room in picked:
        plan += room_requests(room, sizes[room['uid']], instr, 'e2', stratum(room))
        exposure[room['uid']] = room_exposure(room, instr, sidecar, houses, holo, hf_cover)
    meta = {'protocol': 'amendment-014', 'plan': 'e2', 'seed': SEED, 'limit': limit,
            'eval_data_manifest_sha256': _sha(eval_data / 'MANIFEST.json'),
            'e1_plan_manifest_sha256': _sha(e1 / 'MANIFEST.json'),
            'inputs': {DATA_FILES[s][0]: rows[s][1] for s in DATA_FILES}, 'implementation_sha256': _code(),
            'population_by_stratum': dict(sorted(counts.items())), 'rooms': len(picked),
            'exposure_counts': collections.Counter(str(v['seen']) for v in exposure.values()),
            'hf_scene_missing': sum(v.get('hf_scene_coverage', 0) is None for v in exposure.values()),
            'fairness_flag': 'their prompt carries only the room area: no polygon, doors or aspect ratio'}
    return _publish_files(out, {'requests.jsonl': _jsonl_bytes(plan),
                                'exposure.json': (_dumps(exposure, indent=1) + '\n').encode('utf-8')}, meta)


# ---- E3 (amendment 015): our room families outside the paper's four -----------------------------------------------

def families():
    """room_families.family (raw room_type -> family), its PAPER_FAMILIES and the file's sha256 (lazy; tests patch)."""
    from fastfill.comparison import room_families
    return room_families.family, room_families.PAPER_FAMILIES, _sha(room_families.__file__)


def plan_e3(eval_data, data, e1, out, cap=E3_CAP):
    from fastfill import scene  # its norm_room_type decides every family
    eval_data, e1, out = Path(eval_data), Path(e1), Path(out)
    _refuse(out)
    rooms, sizes, sidecar, hf_cover = plan_inputs(eval_data, e1)
    family, paper, family_sha = families()
    skip = (*paper, 'none', 'other')
    by = collections.defaultdict(list)
    for r in rooms:
        by[family(r.get('room_type'))].append(r)
    rank = lambda r: (hashlib.sha256((E3_SEED + r['uid']).encode('utf-8')).hexdigest(), r['uid'])
    picked = [(f, r) for f in sorted(set(by) - set(skip)) for r in sorted(by[f], key=rank)[:cap]]
    rows = official(data)
    instr = instruction(rows['sft'][0])
    houses, holo = exposure_index(rows)
    plan, exposure = [], {}
    for f, room in picked:
        plan += room_requests(room, sizes[room['uid']], instr, 'e3', f)
        exposure[room['uid']] = room_exposure(room, instr, sidecar, houses, holo, hf_cover)
    count = lambda f, key: dict(collections.Counter(key(r) for g, r in picked if g == f))
    receipt = {f: {'population': len(by[f]), 'selected': min(cap, len(by[f])),
                   'sources': count(f, lambda r: r['source']), 'room_types': count(f, lambda r: r['room_type']),
                   'seen': count(f, lambda r: str(exposure[r['uid']]['seen']))} for f in sorted(set(by) - set(skip))}
    meta = {'protocol': 'amendment-015', 'plan': 'e3', 'status': 'EXPLORATORY', 'seed': E3_SEED, 'cap': cap,
            'eval_data_manifest_sha256': _sha(eval_data / 'MANIFEST.json'),
            'e1_plan_manifest_sha256': _sha(e1 / 'MANIFEST.json'),
            'inputs': {DATA_FILES[s][0]: rows[s][1] for s in DATA_FILES},
            'implementation_sha256': {**_code(), 'room_families.py': family_sha, 'scene.py': _sha(scene.__file__)},
            'families': receipt,
            'training_families': {s: dict(collections.Counter(  # training rows: SFT rows, DPO pairs (chosen input)
                family(_TAG['Task Room Type'].search(r['input']).group(1)) for r in chosen(rows[s][0], s)))
                for s in DATA_FILES},
            'skipped_families': {f: len(by[f]) for f in skip if f in by}, 'rooms': len(picked),
            'exposure_counts': collections.Counter(str(v['seen']) for v in exposure.values()),
            'hf_scene_missing': sum(v.get('hf_scene_coverage', 0) is None for v in exposure.values()),
            'fairness_flag': 'their prompt carries only the room area: no polygon, doors or aspect ratio'}
    return _publish_files(out, {'requests.jsonl': _jsonl_bytes(plan),
                                'exposure.json': (_dumps(exposure, indent=1) + '\n').encode('utf-8')}, meta)


# ---- generation ----------------------------------------------------------------------------------------------------

def check_eight_b(model):
    """Read-only identity checks of OptiScene-Qwen-8B-Final (FastFill v1.0 final): hashes every file."""
    model = Path(model)
    config = json.loads((model / 'config.json').read_text())
    jinja = model / 'chat_template.jinja'  # newer tokenizers keep the template here, not in tokenizer_config.json
    template = json.loads((model / 'tokenizer_config.json').read_text()).get('chat_template') or \
        (jinja.read_text() if jinja.is_file() else '')
    problems = [k for k, v in EIGHT_B.items() if config.get(k) != v]
    problems += ['chat_template lacks enable_thinking'] if 'enable_thinking' not in template else []
    problems += ['no generation_config.json'] if not (model / 'generation_config.json').is_file() else []
    problems += [f'unexpected {n}' for n in ('README.md', 'adapter_config.json') if (model / n).exists()]
    if problems:
        raise ValueError(f'{model}: {problems}')
    return {'path': str(model), 'files': _hash_tree(model, skip=())}


def _engine(engine):  # Patched by tests.
    from vllm import LLM, SamplingParams
    return LLM(**engine), SamplingParams


def parse_ok(row, text):
    if row.get('format') == 'v1':
        from fastfill import scene
        return scene.parse(text, row['order'])[1] is None
    return parse_design(text, row['groups'])[1] is None


def generate(plan, prompt, model, out, gate_parse=None, limit=None):
    plan, model, out = Path(plan), Path(model), Path(out)
    _refuse(out)
    protocol = _verified(plan).get('protocol', 'amendment-014')  # amendment 015 for plan-e3
    rows = [r for r in _jsonl(plan / 'requests.jsonl') if prompt in r['messages']][:limit]
    identity = check_eight_b(model) if prompt == 'v1' else {'path': str(model), 'manifest': _verified(model)}
    engine = {'model': str(model), 'max_model_len': MAX_MODEL_LEN, 'seed': 0, 'dtype': 'bfloat16',
              'tensor_parallel_size': 1}
    llm, sampling = _engine(engine)
    tok = llm.get_tokenizer()
    extra = {'enable_thinking': False} if prompt == 'v1' else {}
    prompts = [tok.apply_chat_template(r['messages'][prompt], tokenize=False, add_generation_prompt=True, **extra)
               for r in rows]
    lengths = [len(tok(p, add_special_tokens=False)['input_ids']) for p in prompts]
    caps = [MAX_MODEL_LEN - n if prompt == 'v1' else min(MAX_TOKENS, MAX_MODEL_LEN - n) for n in lengths]
    fit = [k for k, cap in enumerate(caps) if cap > 0]
    outs = llm.generate([prompts[k] for k in fit], [sampling(temperature=0, max_tokens=caps[k], seed=0) for k in fit])
    done = {k: o.outputs[0] for k, o in zip(fit, outs)}
    records = [{'key': r['key'], 'uid': r['uid'], 'prompt': prompt, 'prompt_tokens': lengths[k],
                'text': done[k].text if k in done else None,
                'finish_reason': done[k].finish_reason if k in done else None,
                'completion_tokens': len(done[k].token_ids) if k in done else 0,
                'error': None if k in done else 'prompt_too_long'} for k, r in enumerate(rows)]
    rate = _mean([r['text'] is not None and parse_ok(row, r['text']) for r, row in zip(records, rows)])
    if gate_parse is not None and (rate is None or rate < gate_parse):
        raise ValueError(f'strict parse rate {rate} < {gate_parse}: nothing published')
    versions = {p: _version(p) for p in ('vllm', 'torch', 'transformers')}
    meta = {'protocol': protocol, 'prompt': prompt, 'plan_manifest_sha256': _sha(plan / 'MANIFEST.json'),
            'model': identity, 'engine': engine, 'sampling': {'temperature': 0, 'seed': 0, 'max_tokens': (
                'max_model_len - prompt' if prompt == 'v1' else f'min({MAX_TOKENS}, max_model_len - prompt)')},
            **({} if prompt == 'v1' else {'readme_sampling': README_SAMPLING}),
            'chat_template_kwargs': extra, 'versions': versions, 'limit': limit, 'requests': len(rows),
            'strict_parse_rate': rate, 'gate_parse': gate_parse, 'implementation_sha256': _code()}
    return _publish_files(out, {'records.jsonl': _jsonl_bytes(records)}, meta)


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


# ---- scoring -------------------------------------------------------------------------------------------------------

def score_e1(row, record):
    """One E1 request: parse, convert, score. Failed requests stay with complete False."""
    text = record.get('text') if record else None
    coords, err = parse_design(text, row['groups']) if text is not None else (None, (record or {}).get('error')
                                                                               or 'missing_output')
    floorless = None if row['population'] == 'e1-heldout20' else False  # no floor polygon: parse + OOR only
    out = {'key': row['key'], 'population': row['population'], 'stratum': row['stratum'], 'complete': bool(coords),
           'error': err, 'finish_reason': (record or {}).get('finish_reason'), 'oor': None, 'oor_paper': None,
           'geometry_success': floorless, 'downstream_accept': floorless}
    if not coords:
        return out
    objs = layout_from(coords, row['boxes'], row.get('room'))
    if row['population'] == 'e1-heldout20':
        return {**out, 'oor': oor_ours(objs), 'oor_paper': oor_paper(objs)}
    metrics = room_metrics(row['room'], objs)
    return {**out, **{k: metrics[k] for k in ('oor', 'oor_paper', 'geometry_success', 'downstream_accept')}}


def placements(row, text):
    """Model text -> (placement JSON in bound order, None) or (None, error code)."""
    if row['format'] == 'v1':
        from fastfill import scene
        parsed, err = scene.parse(text, row['order'])
        if err:
            return None, err
        canonical = {row['id_map'][i]: i for i in row['order']}  # bound object id -> canonical id
        items = []
        for bound in row['_order']:
            p = parsed[canonical[bound]]  # scene.parse already rejected supports outside the placed objects
            items.append({'id': bound, 'pos': p['pos'], 'yaw': round(math.degrees(p['yaw'])) % 360,
                          **({'on': row['id_map'][p['on']]} if p['on'] is not None else {})})
        return {'placements': items}, None
    coords, err = parse_design(text, row['groups'])
    if err:
        return None, err
    objs = [t_to_ir(i, 'object', x, y, z, a, b, row['center'])
            for i, (x, y, z, a), b in zip(row['ids'], coords, row['boxes'])]
    by = {o['id']: (o, parent) for o, parent in zip(objs, derive_on(objs))}
    return {'placements': [{'id': i, 'pos': by[i][0]['pos'], 'yaw': round(math.degrees(by[i][0]['yaw'])) % 360,
                            **({'on': by[i][1]} if by[i][1] else {})} for i in row['_order']]}, None


def floor_ids(room):
    return {o['id'] for o in room['objects'] if o.get('parent') is None and abs(o['pos'][2]) <= TOL}


def extras(room, request, row):
    """downstream.accept, oor_paper and the floor-only view of a geometry.score row (P's rows too)."""
    from fastfill.comparison import downstream, geometry
    from fastfill.comparison.output_repair import predicted_room
    keep = floor_ids(room)
    floor = {'objects': len(keep), 'complete': row['complete'], 'geometry_success': False,
             'downstream_accept': False, 'oor': None, 'oor_paper': None}
    if not row['complete']:
        return {**row, 'downstream_accept': False, 'oor_paper': None, 'floor': floor}
    layout = row['raw_layout']
    accepted = downstream.accept(predicted_room(room, request, layout), layout)['accepted']
    if keep:
        sub = [o for o in layout if o['id'] in keep]
        pred = predicted_room({**room, 'objects': [o for o in room['objects'] if o['id'] in keep]}, request, sub)
        geo = geometry.geometry(pred)
        floor = {**floor, 'geometry_success': geo['geometry_success'], 'oor': geo['oor'], 'oor_paper': oor_paper(sub),
                 'downstream_accept': downstream.accept(pred, sub)['accepted']}
    return {**row, 'downstream_accept': accepted, 'oor_paper': oor_paper(layout), 'floor': floor}


def score_e2(task):
    room, request, row, record, arm = task
    from fastfill.comparison import geometry
    text = record.get('text') if record else None
    if text is None:
        raw, err = None, (record or {}).get('error') or 'missing_output'
    else:
        raw, err = placements({**row, '_order': [o['id'] for o in room['objects']]}, text)
        err = err and 'parse:' + err
    result = geometry.score(room, request, _dumps(raw) if raw else None, 'placement', error=err)
    result = extras(room, request, {**result, 'arm': arm})
    result.pop('repaired_layout', None)
    return {**result, 'mode': row['mode'], 'stratum': row['stratum'], 'key': row['key'],
            'finish_reason': (record or {}).get('finish_reason')}


def _rates(rows):
    from fastfill.comparison import geometry
    n = len(rows)
    a = geometry.aggregate(rows) if rows and 'size_iou' in rows[0] else {'requests': n}
    rate = lambda k: sum(bool(r[k]) for r in rows) / n if any(r[k] is not None for r in rows) else None
    a.update({'requests': n, 'complete_rate': rate('complete'), 'geometry_success_rate': rate('geometry_success'),
              'downstream_accept_rate': rate('downstream_accept'),
              'oor': _mean([r['oor'] for r in rows if r['complete']]),
              'oor_paper': _mean([r['oor_paper'] for r in rows if r['complete']]),
              'complete_rooms': sum(r['complete'] for r in rows),
              'errors': dict(collections.Counter(r['error'] for r in rows if r['error'])),
              'finish_length': sum(r.get('finish_reason') == 'length' for r in rows)})
    return a


def _floor_rates(rows):
    f = [r['floor'] for r in rows if r['floor']['objects']]
    n = len(f)
    return {'requests': n, 'complete_rate': sum(x['complete'] for x in f) / n if n else None,
            'geometry_success_rate': sum(bool(x['geometry_success']) for x in f) / n if n else None,
            'downstream_accept_rate': sum(bool(x['downstream_accept']) for x in f) / n if n else None,
            'oor': _mean([x['oor'] for x in f if x['complete']]),
            'oor_paper': _mean([x['oor_paper'] for x in f if x['complete']])}


def _by_stratum(rows, fn):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r['stratum']].append(r)
    return {'pooled': fn(rows), **{s: fn(v) for s, v in sorted(groups.items())}}


def _load_gen(gen, name, plan_dir, keys):
    root = Path(gen) / name
    manifest = _verified(root)
    if manifest['plan_manifest_sha256'] != _sha(Path(plan_dir) / 'MANIFEST.json'):
        raise ValueError(f'{name}: generated from another plan')
    records = {r['key']: r for r in _jsonl(root / 'records.jsonl')}
    if set(records) != set(keys) or manifest.get('limit') is not None:
        raise ValueError(f'{name}: records do not cover exactly the plan requests')
    return records


def _map(fn, tasks, workers):
    if workers == 1:
        return [fn(t) for t in tasks]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(fn, tasks, chunksize=8))


def score(e1, e2, gen, eval_data, p_geometry, p_run, out, workers=1, repeats=2000):
    from fastfill.comparison import geometry
    out = Path(out)
    _refuse(out)
    plans = {'e1': (Path(e1), _verified(e1)), 'e2': (Path(e2), _verified(e2))}
    data_manifest = _verified(eval_data)
    if plans['e2'][1]['eval_data_manifest_sha256'] != _sha(Path(eval_data) / 'MANIFEST.json') \
            or plans['e2'][1].get('limit') is not None:
        raise ValueError('e2 plan was not built from this eval-data (or is a limited smoke plan)')
    if plans['e2'][1]['e1_plan_manifest_sha256'] != _sha(Path(e1) / 'MANIFEST.json'):
        raise ValueError('e2 exposure was audited against another e1 plan')
    plan_rows = {k: _jsonl(path / 'requests.jsonl') for k, (path, _) in plans.items()}
    exposure = json.loads((Path(e2) / 'exposure.json').read_text())
    rooms, requests, gt_requests = room_inputs(eval_data, exposure)
    e1_rows, e2_rows, summary = [], [], {'label': LABEL, 'e1': {}, 'e2': {}, 'paired_vs_P': {}, 'training_exposure': {
        f'{m}-{p}-{q}': EIGHT_B_EXPOSURE if m == '8b' else 'OptiScene release: e2/exposure.json'
        for m, p, q in GENS if p == 'e2'}}
    for model, plan, prompt in GENS:
        name = f'{model}-{plan}-{prompt}'
        rows = [r for r in plan_rows[plan] if prompt in r['messages']]
        records = _load_gen(gen, name, plans[plan][0], [r['key'] for r in rows])
        if plan == 'e1':
            scored = [{**score_e1(r, records[r['key']]), 'arm': name} for r in rows]
            e1_rows += scored
            summary['e1'][name] = {pop: _by_stratum([s for s in scored if s['population'] == pop], _rates)
                                   for pop in ('e1-unseen', 'e1-heldout20')}
            continue
        scored = score_arm(rows, records, name, rooms, requests, gt_requests, workers)
        e2_rows += scored
        summary['e2'][name] = _e2_summary(scored, exposure)
    p_scored = p_extras(p_geometry, p_run, {r['uid']: r['stratum'] for r in plan_rows['e2']}, rooms, requests, workers)
    summary['e2']['P'] = _e2_summary(p_scored, exposure)
    for model, plan, prompt in GENS:
        if plan == 'e2':
            name = f'{model}-{plan}-{prompt}'
            arm = [r for r in e2_rows if r['arm'] == name and r['mode'] == 'prior']
            summary['paired_vs_P'][name] = geometry.paired_intervals(arm, p_scored, repeats=repeats, seed=42)
    summary['gt_e1'] = plans['e1'][1]['gt_by_stratum']
    meta = {'protocol': 'amendment-014', 'plans': {k: _sha(p / 'MANIFEST.json') for k, (p, _) in plans.items()},
            'eval_data_manifest_sha256': _sha(Path(eval_data) / 'MANIFEST.json'),
            'p_geometry_manifest_sha256': _sha(Path(p_geometry) / 'MANIFEST.json'),
            'p_run_manifest_sha256': _sha(Path(p_run) / 'MANIFEST.json'),
            'gens': {f'{m}-{p}-{q}': _sha(Path(gen) / f'{m}-{p}-{q}' / 'MANIFEST.json') for m, p, q in GENS},
            'implementation_sha256': _code(), 'repeats': repeats, 'eval_data_files': len(data_manifest['files']),
            'paired': 'estimate_joint_minus_placement = OptiScene arm (prior sizes) - P (house-cluster bootstrap)',
            'p_rows': 'geometry-v5 placement rows with report_v2.cap_rule (amendment 005) on the selected P run'}
    return _publish_files(out, {'e1-rows.jsonl': _jsonl_bytes(e1_rows), 'e2-rows.jsonl': _jsonl_bytes(e2_rows),
                                'p-rows.jsonl': _jsonl_bytes(p_scored),
                                'summary.json': (_dumps(summary, indent=1) + '\n').encode('utf-8')}, meta)


def score_e3(e3, gen, eval_data, p_geometry, p_run, out, workers=1, repeats=2000):
    """E3 per family and pooled (EXPLORATORY): every published E3_GENS arm, P, paired OptiScene(prior) - P; again on the
    rooms within OptiScene's SFT size range (<= TRAIN_MAX_OBJECTS objects)."""
    e3, gen, out = Path(e3), Path(gen), Path(out)
    _refuse(out)
    plan = _verified(e3)
    _verified(eval_data)
    if plan.get('plan') != 'e3' or plan['eval_data_manifest_sha256'] != _sha(Path(eval_data) / 'MANIFEST.json'):
        raise ValueError(f'{e3} is not a plan-e3 built from this eval-data')
    rows = _jsonl(e3 / 'requests.jsonl')
    exposure = json.loads((e3 / 'exposure.json').read_text())
    rooms, requests, gt_requests = room_inputs(eval_data, exposure)
    arms = [n for n in E3_GENS if (gen / n / 'MANIFEST.json').is_file()]
    if not arms:
        raise ValueError(f'no E3 generation under {gen}')
    summary = {'label': LABEL, 'status': 'EXPLORATORY', 'families': plan['families'],
               'training_families': plan['training_families'], 'not_published': [n for n in E3_GENS if n not in arms],
               'training_exposure': {n: EIGHT_B_EXPOSURE if n.startswith('8b') else 'OptiScene release: e3/exposure.json'
                                     for n in arms}}
    e3_rows = []
    for name in arms:
        sel = [r for r in rows if name.rsplit('-', 1)[1] in r['messages']]
        e3_rows += score_arm(sel, _load_gen(gen, name, e3, [r['key'] for r in sel]), name, rooms, requests,
                             gt_requests, workers)
    p_scored = p_extras(p_geometry, p_run, {r['uid']: r['stratum'] for r in rows}, rooms, requests, workers)
    small = {u for u, r in rooms.items() if len(r['objects']) <= TRAIN_MAX_OBJECTS}
    pick = lambda r, scope, f: f in ('pooled', r['stratum']) and (scope == 'all' or exposure[r['uid']]['seen'] is False)
    for suffix, keep in (('', set(rooms)), (f'_le{TRAIN_MAX_OBJECTS}', small)):
        mine, p = [r for r in e3_rows if r['uid'] in keep], [r for r in p_scored if r['uid'] in keep]
        summary['e3' + suffix] = {**{n: _e2_summary([r for r in mine if r['arm'] == n], exposure) for n in arms},
                                  'P': _e2_summary(p, exposure)}
        summary['paired_vs_P' + suffix] = {n: {scope: {f: _paired_flagged(
            [r for r in mine if r['arm'] == n and r['mode'] == 'prior' and pick(r, scope, f)],
            [r for r in p if pick(r, scope, f)], repeats) for f in ('pooled', *plan['families'])}
            for scope in ('all', 'unseen')} for n in arms}
    meta = {'protocol': 'amendment-015', 'status': 'EXPLORATORY',
            'plan_e3_manifest_sha256': _sha(e3 / 'MANIFEST.json'), 'eval_data_manifest_sha256': _sha(Path(eval_data) / 'MANIFEST.json'),
            'p_geometry_manifest_sha256': _sha(Path(p_geometry) / 'MANIFEST.json'),
            'p_run_manifest_sha256': _sha(Path(p_run) / 'MANIFEST.json'),
            'gens': {n: _sha(gen / n / 'MANIFEST.json') for n in arms}, 'implementation_sha256': _code(),
            'repeats': repeats, 'parse_rate': 'complete_rate (strict parse + output contract); errors break it down',
            'paired': 'estimate_joint_minus_placement = OptiScene arm (prior sizes) - P (house-cluster bootstrap); '
                      'unseen = exposure seen is False; components/flags = report_v2._design (audit F7), ci95 None '
                      'below report_v2.NO_CI components',
            'pooled': 'sampled rooms (<= cap per family): family-balanced, not weighted to the test-set population',
            f'le{TRAIN_MAX_OBJECTS}': f'rooms with <= {TRAIN_MAX_OBJECTS} objects (the SFT training maximum; their '
                                      f'output cap {MAX_TOKENS} tokens): family not confounded with size/truncation',
            'p_rows': 'geometry-v5 placement rows with report_v2.cap_rule (amendment 005) on the selected P run'}
    return _publish_files(out, {'e3-rows.jsonl': _jsonl_bytes(e3_rows), 'p-rows.jsonl': _jsonl_bytes(p_scored),
                                'summary.json': (_dumps(summary, indent=1) + '\n').encode('utf-8')}, meta)


def _paired_flagged(arm, p, repeats):
    """paired_intervals(arm, P) with report_v2's small-sample design (audit F7): flags, no CI below NO_CI components."""
    from fastfill.comparison import geometry, report_v2 as r2
    design, result = r2._design(p), geometry.paired_intervals(arm, p, repeats=repeats, seed=42)
    if design['components'] < r2.NO_CI:
        result = {f: {**v, 'ci95': None, 'source_macro': {**v['source_macro'], 'ci95': None}}
                  for f, v in result.items()}
    return {**design, **result}


def room_inputs(eval_data, uids):
    """Eval-data rooms, P's prior-size requests and GT-size requests of the plan's rooms (score, score-e3)."""
    from fastfill.comparison.contract import messages
    rooms = {r['uid']: r for r in _jsonl(Path(eval_data) / 'rooms.jsonl') if r['uid'] in uids}
    requests = {r['uid']: r['messages'] for r in _jsonl(Path(eval_data) / 'placement/requests.jsonl')
                if r['uid'] in uids}
    gt_requests = {u: messages(r, 'placement', estimated_sizes={o['id']: o['size'] for o in r['objects']})[:2]
                   for u, r in rooms.items()}
    return rooms, requests, gt_requests


def score_arm(rows, records, name, rooms, requests, gt_requests, workers):
    tasks = [(rooms[r['uid']], requests[r['uid']] if r['mode'] == 'prior' else gt_requests[r['uid']], r,
              records[r['key']], name) for r in rows]
    return _map(score_e2, tasks, workers)


def p_extras(p_geometry, p_run, strata, rooms, requests, workers):
    """P's scored test rows of exactly the plan's rooms ({uid: stratum}), with the E2 extras."""
    p_rows = [r for r in p_test_rows(p_geometry, p_run) if r['uid'] in strata]
    if {r['uid'] for r in p_rows} != set(strata) or len(p_rows) != len(strata):
        raise ValueError('P geometry rows do not cover the plan rooms exactly')
    return _map(_p_extras, [(rooms[r['uid']], requests[r['uid']], r, strata[r['uid']]) for r in p_rows], workers)


def p_test_rows(p_geometry, p_run):
    """P's test rows as its report scored them: geometry rows of the selected run with the amendment-005 cap rule."""
    from fastfill.comparison.report_v2 import cap_rule
    _verified(p_geometry)
    _verified(p_run)
    bound = json.loads((Path(p_geometry) / 'summary.json').read_text())['inference_run_sha256']['placement']
    if bound != json.loads((Path(p_run) / 'run_receipt.json').read_text())['run_sha256']:
        raise ValueError(f'{p_geometry} does not score {p_run}')
    scored, records = _jsonl(Path(p_geometry) / 'placement-geometry.jsonl'), _jsonl(Path(p_run) / 'records.jsonl')
    if [r['uid'] for r in scored] != [r['uid'] for r in records]:
        raise ValueError('P geometry rows and inference records differ')
    return cap_rule(scored, records)[0]


def _p_extras(task):
    room, request, row, stratum = task
    result = extras(room, request, row)
    result.pop('repaired_layout', None)
    return {**result, 'arm': 'P', 'mode': 'prior', 'stratum': stratum}


def _e2_summary(rows, exposure):
    unseen = [r for r in rows if exposure[r['uid']]['seen'] is False]
    return {mode: {scope: {'full': _by_stratum(sel, _rates), 'floor': _by_stratum(sel, _floor_rates)}
                   for scope, sel in (('all', [r for r in rows if r['mode'] == mode]),
                                      ('unseen', [r for r in unseen if r['mode'] == mode]))}
            for mode in sorted({r['mode'] for r in rows})}


# ---- training (venv optiscene-014) ---------------------------------------------------------------------------------

def sft_argv(data, base, out):
    return ['--task', 'sft', '--dataset_file', str(Path(data) / DATA_FILES['sft'][0]),
            f'--model_name_or_path={base}', '--bf16', f'--checkpoint_dir={out}',
            '--per_device_train_batch_size=8', '--save_strategy=epoch', '--epochs=1']


def dpo_argv(data_file, base, out):
    return ['--task', 'dpo', '--dataset_file', str(data_file), '--model_name_or_path', str(base),
            '--learning_rate', '5.0e-6', '--num_train_epochs', '10', '--per_device_train_batch_size', '1',
            '--gradient_accumulation_steps', '4', '--lora_target_modules', 'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'gate_proj', 'up_proj', 'down_proj', '--max_length', '3200', '--max_prompt_length', '3200',
            '--gradient_checkpointing', '--logging_steps', '20', '--save_steps', '100', '--eval_strategy', 'steps',
            '--eval_steps', '500', '--output_dir', str(out), '--no_remove_unused_columns', '--use_peft',
            '--lora_r', '32', '--lora_alpha', '16', '--bf16'] + OPS_ARGV


def stage_argv(stage, data, base, out):
    return sft_argv(data, base, out) if stage == 'sft' else dpo_argv(Path(data) / DATA_FILES[stage][0], base, out)


def checkpoint_rule(stage, out):
    """Last checkpoint = expected step = global_step = max_steps; DPO adapter = the save_model copy in output_dir."""
    out, expected = Path(out), STEPS[stage]
    steps = sorted(int(p.name.split('-')[1]) for p in out.glob('checkpoint-*') if re.fullmatch(r'checkpoint-\d+', p.name))
    if not steps or steps[-1] != expected:
        raise ValueError(f'{stage}: last checkpoint {steps[-1] if steps else None} is not checkpoint-{expected}')
    ckpt = out / f'checkpoint-{expected}'
    state = json.loads((ckpt / 'trainer_state.json').read_text())
    if not state.get('global_step') == state.get('max_steps') == expected:
        raise ValueError(f'{stage}: trainer_state global_step {state.get("global_step")} / max_steps '
                         f'{state.get("max_steps")} != {expected}')
    adapter = _sha(ckpt / 'adapter_model.safetensors')
    final = None
    if stage != 'sft':
        final = _sha(out / 'adapter_model.safetensors')
        if final != adapter:
            raise ValueError(f'{stage}: checkpoint-{expected} adapter differs from the save_model copy')
    return {'checkpoint': str(ckpt), 'global_step': expected, 'adapter_sha256': adapter,
            'save_model_adapter_sha256': final}


def stage_complete(stage, out):
    try:
        checkpoint_rule(stage, out)
        return True
    except (ValueError, OSError):
        return False


def quarantine_partial(out, job):
    """A checkpoint killed while saving (no trainer_state.json) is moved aside so get_last_checkpoint skips it."""
    moved = []
    for p in sorted(Path(out).glob('checkpoint-*')):
        if not (p / 'trainer_state.json').is_file():
            target = p.with_name(f'.partial-{p.name}-job{job}')
            p.rename(target)
            moved.append(str(target))
    return moved


def _job_state(job):
    run = subprocess.run(['sacct', '-j', str(job), '-X', '-n', '-P', '-o', 'State'], capture_output=True, text=True)
    states = [line.split()[0] for line in run.stdout.splitlines() if line.strip()] if run.returncode == 0 else []
    return states[0] if states else None


def _final_state(job, tries=60, pause=10):
    """Polls sacct for up to ~10 min while the previous job still looks unfinished (accounting lags afterany)."""
    for k in range(tries):
        state = _job_state(job)
        if state not in WAITING or k == tries - 1:
            return state
        time.sleep(pause)


def _single_gpu():
    devices = [d for d in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if d.strip()]
    if len(devices) != 1:
        raise ValueError(f'need exactly one visible GPU, CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")!r}')


def _env(path):
    env = json.loads(Path(path).read_text())
    upstream = json.loads((VENDORED / 'UPSTREAM.json').read_text())
    if env.get('ok') is not True or Path(env['sys_prefix']).resolve() != Path(sys.prefix).resolve():
        raise ValueError('env.json is not a passing env-check of this interpreter')
    if any(_sha(VENDORED / f) != v['vendored_sha256'] for f, v in upstream['files'].items()):
        raise ValueError('vendored files differ from UPSTREAM.json')
    return env


def _base_identity(stage, run, base, env):
    if stage == 'sft':
        model = env.get('model') or {}
        if Path(model.get('path', '')).resolve() != Path(base).resolve() or _hash_tree(base) != model.get('files'):
            raise ValueError(f'{base}: files differ from the env-check model hashes')
        return Path(base), {'path': str(base), 'env_check_files': len(model['files'])}
    merged = Path(run) / f'{PREVIOUS[stage]}-merged'
    return merged, {'path': str(merged), 'manifest_sha256': _sha(merged / 'MANIFEST.json'),
                    'receipt': _verified(merged)['receipt']}


def train(stage, run, data, base, env_json, prev_job=None, python=sys.executable):
    run = Path(run)
    env = _env(env_json)
    _single_gpu()
    state = None if prev_job is None else _final_state(prev_job)
    if prev_job is not None and state not in ('COMPLETED', 'TIMEOUT'):
        raise ValueError(f'previous job {prev_job} ended {state}, not COMPLETED or TIMEOUT')
    out = run / stage / 'out'
    if stage_complete(stage, out):
        print(f'{stage}: complete ({checkpoint_rule(stage, out)["checkpoint"]}); nothing to do')
        return 0
    if stage == 'sft' and out.exists() and any(out.iterdir()):
        raise ValueError(f'{out}: an incomplete SFT run cannot resume (sft_train has no resume); new amendment needed')
    name, prefix = DATA_FILES[stage]
    digest = _sha(Path(data) / name)
    if not digest.startswith(prefix):
        raise ValueError(f'{name}: sha256 prefix is not {prefix}')
    model, identity = _base_identity(stage, run, base, env)
    out.mkdir(parents=True, exist_ok=True)
    job = os.environ.get('SLURM_JOB_ID', 'local')
    receipts = run / stage / 'receipts'
    receipts.mkdir(exist_ok=True)
    argv = stage_argv(stage, data, model, out)
    receipt = {'label': LABEL, 'stage': stage, 'job': job, 'prev_job': prev_job, 'argv': argv,
               'expected_steps': STEPS[stage], 'base': identity, 'data': {name: digest},
               'env_json_sha256': _sha(env_json), 'fixes': FIXES, 'kept_as_released': KEPT,
               'quarantined': quarantine_partial(out, job), 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES']}
    with open(receipts / f'job-{job}-start.json', 'x') as f:
        f.write(_dumps(receipt, indent=2) + '\n')
    code = subprocess.run([python, '-B', 'main.py', *argv], cwd=VENDORED).returncode
    with open(receipts / f'job-{job}-end.json', 'x') as f:
        f.write(_dumps({'returncode': code, 'complete': stage_complete(stage, out),
                        'checkpoints': sorted(p.name for p in out.glob('checkpoint-*'))}, indent=2) + '\n')
    return code


def merge(stage, run, base, env_json, python=sys.executable):
    run = Path(run)
    out = run / f'{stage}-merged'
    _refuse(out)
    env = _env(env_json)
    rule = checkpoint_rule(stage, run / stage / 'out')
    model, identity = _base_identity(stage, run, base, env)
    with tempfile.TemporaryDirectory(prefix=f'.{out.name}-', dir=run) as tmp:
        subprocess.run([python, '-B', 'main.py', '--task', 'merge', '--base_model_path', str(model),
                        '--lora_path', rule['checkpoint'], '--output_path', tmp], cwd=VENDORED, check=True,
                       env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
        return _finish(tmp, out, {'protocol': 'amendment-014', 'receipt': {
            'stage': stage, 'checkpoint_rule': rule, 'base': identity, 'merge': 'vendored merge_lora (bf16, CPU)',
            'env_json_sha256': _sha(env_json), 'fixes': FIXES, 'kept_as_released': KEPT,
            'implementation_sha256': _code()}})


def env_check(data, out, model=None, revision=None):
    _refuse(out)
    errors = []
    versions = {p: _version(p) for p in [*PINS, 'torch']}
    errors += [f'{p} {versions[p]} != {v}' for p, v in PINS.items() if versions[p] != v]
    upstream = json.loads((VENDORED / 'UPSTREAM.json').read_text())
    errors += [f'vendored {f} differs' for f, v in upstream['files'].items() if _sha(VENDORED / f) != v['vendored_sha256']]
    sys.path.insert(0, str(VENDORED))
    try:
        importlib.import_module('modelscope')  # F4 is needed if this fails.
    except ImportError as exc:
        errors.append(f'modelscope import failed ({exc}): apply F4 and re-vendor')
    if errors:
        raise ValueError(f'env-check failed, nothing written: {errors}')
    import torch
    import main  # The vendored entry point (imports scripts.* and utils, as `python main.py` does).
    from trl import DPOConfig, ModelConfig, SFTConfig, TrlParser
    argv = dpo_argv('DATA/dpo_prompts.json', 'BASE', tempfile.mkdtemp())[2:]
    argv[argv.index('--eval_strategy')] = '--evaluation_strategy'  # main.py's own remap
    script, config, model_args = TrlParser((main.OptiSceneScriptArguments, DPOConfig, ModelConfig)) \
        .parse_args_and_config(args=argv)
    parsed = {'dataset_file': script.dataset_file, 'learning_rate': config.learning_rate,
              'num_train_epochs': config.num_train_epochs, 'lr_scheduler_type': str(config.lr_scheduler_type),
              'warmup_ratio': config.warmup_ratio, 'warmup_steps': config.warmup_steps, 'beta': config.beta,
              'max_length': config.max_length, 'max_prompt_length': config.max_prompt_length, 'bf16': config.bf16,
              'eval_strategy': str(config.eval_strategy), 'save_total_limit': config.save_total_limit,
              'torch_dtype': model_args.torch_dtype, 'lora_r': model_args.lora_r, 'lora_alpha': model_args.lora_alpha}
    want = {'dataset_file': 'DATA/dpo_prompts.json', 'learning_rate': 5e-6, 'num_train_epochs': 10,
            'warmup_ratio': 0, 'warmup_steps': 0, 'beta': .1, 'max_length': 3200, 'max_prompt_length': 3200,
            'bf16': True, 'save_total_limit': 2, 'torch_dtype': None, 'lora_r': 32, 'lora_alpha': 16}
    errors += [f'README DPO argv: {k}={parsed[k]!r} != {v!r}' for k, v in want.items() if parsed[k] != v]
    sft = SFTConfig(output_dir=tempfile.mkdtemp(), max_seq_length=2048, bf16=True, num_train_epochs=1,
                    per_device_train_batch_size=8, save_strategy='epoch', eval_strategy='steps', eval_steps=500,
                    report_to='none', gradient_checkpointing=True)  # F5
    rows = official(data)
    instruction(rows['sft'][0])
    result = {'ok': not errors, 'errors': errors, 'label': LABEL, 'sys_prefix': sys.prefix, 'python': sys.version,
              'versions': versions, 'torch_cuda': torch.version.cuda, 'cuda_available': torch.cuda.is_available(),
              'readme_dpo_parsed': parsed, 'sft_config_max_seq_length': sft.max_seq_length,
              'sft_config_gradient_checkpointing': sft.gradient_checkpointing,
              'data': {DATA_FILES[s][0]: rows[s][1] for s in DATA_FILES}, 'fixes': FIXES, 'kept_as_released': KEPT,
              'upstream_commit': upstream['commit'], 'implementation_sha256': _code()}
    if model:
        config_json = json.loads((Path(model) / 'config.json').read_text())
        if config_json.get('architectures') != ['Qwen2ForCausalLM']:
            errors.append(f'{model}: not a Qwen2ForCausalLM')
        result['model'] = {'path': str(model), 'revision': Path(revision).read_text().strip() if revision else None,
                           'files': _hash_tree(model)}
        result.update(ok=not errors, errors=errors)
    if errors:
        raise ValueError(f'env-check failed, nothing written: {errors}')
    with open(out, 'x') as f:
        f.write(_dumps(result, indent=2) + '\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('env-check')
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--model', type=Path)
    p.add_argument('--revision-file', type=Path)
    for name in ('train', 'merge'):
        p = sub.add_parser(name)
        p.add_argument('--stage', choices=tuple(STEPS), required=True)
        p.add_argument('--run', type=Path, required=True)
        p.add_argument('--base', type=Path, required=True)
        p.add_argument('--env', type=Path, required=True)
        p.add_argument('--python', default=sys.executable)
        if name == 'train':
            p.add_argument('--data', type=Path, required=True)
            p.add_argument('--prev-job')
    p = sub.add_parser('plan-e1')
    for name in ('hf', 'data', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p = sub.add_parser('plan-e2')
    for name in ('eval-data', 'data', 'e1', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--limit', type=int)
    p = sub.add_parser('plan-e3')
    for name in ('eval-data', 'data', 'e1', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p = sub.add_parser('generate')
    for name in ('plan', 'model', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--prompt', choices=('sft', 'reasoning', 'v1'), required=True)
    p.add_argument('--gate-parse', type=float)
    p.add_argument('--limit', type=int)
    p = sub.add_parser('score')
    for name in ('e1', 'e2', 'gen', 'eval-data', 'p-geometry', 'p-run', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--repeats', type=int, default=2000)
    p = sub.add_parser('score-e3')
    for name in ('e3', 'gen', 'eval-data', 'p-geometry', 'p-run', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--repeats', type=int, default=2000)
    a = parser.parse_args(argv)
    if a.command == 'env-check':
        result = env_check(a.data, a.out, a.model, a.revision_file)
        print(_dumps({'ok': result['ok'], 'versions': result['versions'], 'out': str(a.out)}))
    elif a.command == 'train':
        return train(a.stage, a.run, a.data, a.base, a.env, a.prev_job, a.python)
    elif a.command == 'merge':
        print(_dumps(merge(a.stage, a.run, a.base, a.env, a.python)['receipt']['checkpoint_rule']))
    elif a.command == 'plan-e1':
        m = plan_e1(a.hf, a.data, a.out)
        print(_dumps({k: m[k] for k in ('counts', 'strata', 'heldout20', 'gate', 'gt_by_stratum')}, indent=1))
    elif a.command == 'plan-e2':
        m = plan_e2(a.eval_data, a.data, a.e1, a.out, a.limit)
        print(_dumps({k: m[k] for k in ('rooms', 'population_by_stratum', 'exposure_counts')}, indent=1))
    elif a.command == 'plan-e3':
        m = plan_e3(a.eval_data, a.data, a.e1, a.out)
        print(_dumps({k: m[k] for k in ('rooms', 'families', 'skipped_families', 'exposure_counts')}, indent=1))
    elif a.command == 'score-e3':
        score_e3(a.e3, a.gen, a.eval_data, a.p_geometry, a.p_run, a.out, a.workers, a.repeats)
    elif a.command == 'generate':
        m = generate(a.plan, a.prompt, a.model, a.out, a.gate_parse, a.limit)
        print(_dumps({k: m[k] for k in ('requests', 'strict_parse_rate')}))
    else:
        score(a.e1, a.e2, a.gen, a.eval_data, a.p_geometry, a.p_run, a.out, a.workers, a.repeats)
    return 0


if __name__ == '__main__':
    sys.exit(main())
