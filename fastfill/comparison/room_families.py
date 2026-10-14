"""Room families beyond the paper's four, and per-family result tables. EXPLORATORY (post hoc, not in the protocol).

python -m fastfill.comparison.room_families mapping --rooms ROOMS.jsonl          markdown mapping table to stdout
python -m fastfill.comparison.room_families report --data RUN/eval-data-v1 --geometry RUN/geometry-v5 \
    --joint-run MERGED_J --placement-run MERGED_P [--agent NAME=RUN/llm-eval-NAME-selected-v1 ...] \
    [--s1 S1_DIR ...] --out NEW_DIR [--repeats 2000] [--workers 1]

family(raw room_type): None or blank -> none; scene.norm_room_type, and a string it drops ('other room', 'misc.')
-> other; then EXPLICIT; then KEYWORDS in order (first match wins), tried on the head (text before ' with ', '/'
or ' and ': 'kitchen with dining area' is a kitchen, 'living and dining room' a living room) before the full
string; no match -> other. Decisions: 'bedroom with home office' -> bedroom (head rule); MansionWorld 'standard
room' (hotel rooms: double bed, luggage bench) and 'dorm room' -> bedroom; office 'pantry' (kitchenette: fridge,
microwave, bar stools) -> kitchen, 'pantry restroom' -> bathroom (toilet suite); MansionWorld 'hub room'
(co-working hub: work chairs, communal tables) -> study_office; lounges, game/media/recreation rooms ->
living; conference/meeting/board/huddle rooms -> meeting (their own family: 125 test rooms); reception and
waiting areas -> public_commercial; 'studio' and 'apartment' (no function) -> other.
PAPER_FAMILIES are the families of the OptiScene paper strata (eval_data.PAPER_STRATA). The paper evaluates no other
family; its released SFT and DPO2 prompts hold only these four, its DPO1 prompts (7,951 pairs) also Library
(study_office), Hallway + Corridor (circulation) 16 pairs each, KidsRoom (bedroom) 16 and OtherRoom 4. Scope
other_families = every named non-paper family (none and other excluded, as optiscene_eval.plan_e3). Families track
sources (meeting is mostly MansionWorld, garage IL3D_synthetic): every scope reports its source mix. Known limit: the
paper families keep ~411 commercial/office variants OptiScene never saw (hotel, dorm and standard rooms; office and
public restrooms; game, recreation and lounge rooms; office pantries).

report: rows of every system get one view per room: complete, checker success (geometry_success, frozen checker),
downstream acceptance (downstream.accept, path A, on the raw layout in the request frame, as output_repair /
optiscene_eval.extras build it), footprint bias B = room mean of log(pred/GT x*y) (report_v2.size_bias), size log
MAE and OOR (geometry rows; complete rooms only). GT = the eval-data room itself: geometry.geometry and
downstream.accept on the GT layout (as decide_012), B = log MAE = 0. The amendment-005 cap rule always fails J/P
rows (--joint-run/--placement-run: the merged inference dirs geometry-v5 binds, as report_v2.run); agents per row
where their rows carry the flags (report_v2). --s1 adds output_repair outputs of geometry-v5 as systems '<arm>+S1'
(checker, downstream and OOR after S1; sizes unchanged; one per arm). Rates use every request of the scope; levels
average each system's own complete rooms. Contrasts (J-P on the test set, P-agent on the agents' cohort) use
report_v2.contrasts, i.e. geometry.paired_intervals' design and draws (house-component bootstrap, source-signature
strata, 2,000 draws, seed 42, room micro; scalars on common-complete pairs). Per scope: n rooms, k house components,
sources and the flags of report_v2._design (few clusters k < 30, small n n < 50); no CI below 10 components.
Outputs (write-once): families.json, report.md (Chinese), levels.csv, contrasts.csv, mapping.md, MANIFEST.json.
"""
import argparse
import collections
import csv
import json
from pathlib import Path
import re
import statistics
import sys
import tempfile

from fastfill.comparison import downstream as dn
from fastfill.comparison import evaluate_pair as e
from fastfill.comparison import geometry as g
from fastfill.comparison import report_v2 as r2
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.corpus import _publish
from fastfill.comparison.output_repair import predicted_room
from fastfill import scene

FAMILIES = ('bedroom', 'living', 'dining', 'kitchen', 'bathroom', 'study_office', 'meeting', 'balcony',
            'laundry_utility', 'garage', 'circulation', 'storage_closet', 'public_commercial', 'other', 'none')
PAPER_FAMILIES = ('bedroom', 'living', 'kitchen', 'bathroom')
EXPLICIT = {'studio': 'other', 'wine bar': 'storage_closet', 'main hall': 'public_commercial'}
KEYWORDS = tuple((family, re.compile(pattern)) for family, pattern in (
    ('bathroom', r'\b(bath|bathroom|restroom|toilet|washroom|powder room|showers?|wc|lavatory)\b'),
    ('bedroom', r"bedroom|\b(bed|nursery|guest room|dorm|hotel room|standard room|on call room|elderly room|suite"
                r"|kids? room|child(ren)?('s)? room)\b"),
    ('storage_closet', r"closet|\b(storage|store ?room|cloakroom|dressing room|linen|cellar|archive)\b"),
    ('laundry_utility', r'laundr|mud ?room|utility|tool ?room|\bserver\b'),
    ('garage', r'\b(garage|workshop)\b'),
    ('balcony', r'\b(balcony|terrace|porch|deck|patio|veranda|outdoor|garden|greenhouse|yard|driveway)\b'),
    ('meeting', r'\b(conference|meeting|board ?room|huddle)\b'),
    ('public_commercial', r'store\b|\b(shop|restaurant|cafe|bar|pub|bakery|delicatessen|buffet|salon|parlor|gym|fitness'
                          r'|yoga|dance|workout|spa|therapy|hospital|patient|clinic|exam|examination|treatment'
                          r'|consultation|consulting|triage|emergency|operating|surgery|imaging|xray|pharmacy|lab'
                          r'|laboratory|diagnostic|dentist|medical|procedure|classroom|kindergarten|daycare|school'
                          r'|reception|waiting|exhibition|gallery|museum|showroom|warehouse|casino|prison|locker'
                          r'|studio|editing|shooting|activity|grocery|florist|fitting|nurses?|recovery|checkout'
                          r'|wellness|section)\b'),
    ('circulation', r'\b(corridor|hallway|hall|aisle|lobby|foyer|entrance|entry|entryway|vestibule|landing|stairs?'
                    r'|staircase|stairwell|passage)\b'),
    ('living', r'\b(living|lounge|family room|media room|tv|recreation|rec|game|gaming|playroom|common room'
               r'|break room|leisure|music room|great room|tatami)\b'),
    ('dining', r'\bdining\b'),
    ('kitchen', r'kitchen|\bpantry\b'),
    ('study_office', r'\b(study|office|library|booth|focus|work area|workstations?|collaboration|hub|copy|assistant)\b'),
))
FIELDS = ('complete', 'geometry_success', 'downstream', 'bias_fp', 'size_log_mae', 'oor')
NO_CI, SEED = r2.NO_CI, 42
NAMES = {'complete': '完整率', 'geometry_success': '检查器成功', 'downstream': '下游接受', 'bias_fp': '足迹偏差 B',
         'size_log_mae': '尺寸 log MAE', 'oor': 'OOR'}


def family(raw):
    if raw is None or not str(raw).strip():
        return 'none'
    text = scene.norm_room_type(raw)
    if text is None:
        return 'other'
    if text in EXPLICIT:
        return EXPLICIT[text]
    for candidate in (re.split(r' with |/| and ', text)[0], text):
        for name, pattern in KEYWORDS:
            if pattern.search(candidate):
                return name
    return 'other'


def mapping_table(raw_types):
    """Markdown: every raw room_type string with its count under its family; raw_types = iterable of raw values."""
    counts = collections.Counter(raw_types)
    by_family = collections.defaultdict(list)
    for raw, n in counts.items():
        by_family[family(raw)].append((raw, n))
    lines = ['| family | paper | rooms | strings | raw room_type (count) |', '|---|---|---:|---:|---|']
    for name in FAMILIES:
        items = sorted(by_family[name], key=lambda x: (-x[1], str(x[0])))
        text = '; '.join(f'{"(none)" if raw is None else raw} ({n})' for raw, n in items)
        lines.append(f"| {name} | {_paper(name, 'yes', 'no')} | {sum(n for _, n in items)} | "
                     f"{len(items)} | {text.replace('|', '/')} |")
    lines.append(f'| total | | {sum(counts.values())} | {len(counts)} | |')
    return '\n'.join(lines) + '\n'


def _paper(name, yes, no):
    return yes if name in PAPER_FAMILIES else '—' if name in ('none', 'other') else no


# ---- per-room views -------------------------------------------------------------------------------------------

def _view(row, room, downstream):
    done = bool(row['complete'])
    return {'uid': room['uid'], 'source': room['source'], 'group': room['group'],
            'family': family(room.get('room_type')), 'complete': done,
            'geometry_success': bool(row['geometry_success']), 'downstream': bool(downstream),
            'bias_fp': r2.size_bias(room['objects'], row['raw_layout'])[0] if done else None,
            'size_log_mae': row['size_log_mae'] if done else None, 'oor': row['oor'] if done else None}


def _scored(task):
    """(geometry row, room, request) -> view; downstream.accept on the raw layout in the request frame."""
    row, room, request = task
    if row['uid'] != room['uid']:
        raise ValueError('row/room identity mismatch')
    layout = row['raw_layout']
    accepted = row['complete'] and dn.accept(predicted_room(room, request['messages'], layout), layout)['accepted']
    return _view(row, room, accepted)


def _s1(row, room, capped):
    s1, done = row['s1'], row['complete'] and not capped
    view = _view(r2.cap_failure(row) if capped else row, room, False)
    return {**view, 'geometry_success': bool(done and s1['geometry_success_s1']),
            'downstream': bool(done and s1['strict_ok_s1']), 'oor': s1['geometry_s1']['oor'] if done else None}


def _gt(room):
    geo = g.geometry({**room, 'constraints': []})
    return {**_view({'complete': True, 'geometry_success': geo['geometry_success'], 'raw_layout': room['objects'],
                     'size_log_mae': 0.0, 'oor': geo['oor']}, room, dn.accept(room, room['objects'])['accepted'])}


# ---- tables ---------------------------------------------------------------------------------------------------

def scopes(views):
    named = {'all': views, 'paper_families': [v for v in views if v['family'] in PAPER_FAMILIES],
             'other_families': [v for v in views if v['family'] not in (*PAPER_FAMILIES, 'none', 'other')]}
    return {**named, **{f: [v for v in views if v['family'] == f] for f in FAMILIES}}


def design(views):
    return {'rooms': len(views), **r2._design(views), 'sources': dict(collections.Counter(v['source'] for v in views))}


def level(views):
    n, done = len(views), [v for v in views if v['complete']]
    mean = lambda key: statistics.fmean(v[key] for v in done) if done else None
    rate = lambda key: sum(v[key] for v in views) / n if n else None
    return {'requests': n, 'complete_rooms': len(done), **{k: rate(k) for k in FIELDS[:3]},
            **{k: mean(k) for k in FIELDS[3:]}}


def contrast(first, second, repeats=2000, seed=SEED):
    """first - second per field; report_v2.contrasts = geometry.paired_intervals' design and draws."""
    base = design(first)
    if not first:
        return {**base, 'fields': {}}
    result = r2.contrasts(first, second, FIELDS, repeats, seed)
    if base['components'] < NO_CI:
        result = {f: {**v, 'ci95': None, 'p': None} for f, v in result.items()}
    return {**base, 'fields': result}


def _subset(views, uids):
    keep = set(uids)
    return [v for v in views if v['uid'] in keep]


def build(test, cohort=None, repeats=2000):
    """test: {label: views on every request} with 'J', 'P', 'GT'; cohort: {label: views on the agents' cohort}."""
    def levels(systems):
        split = {label: scopes(views) for label, views in systems.items()}
        return {scope: {'design': design(split['GT'][scope]),
                        'systems': {label: level(s[scope]) for label, s in split.items()}}
                for scope in split['GT']}
    result = {'test': levels(test), 'contrasts': {'J-P': {
        scope: contrast(j, p, repeats) for (scope, j), p in zip(scopes(test['J']).items(), scopes(test['P']).values())}}}
    if cohort:
        result['cohort'] = levels(cohort)
        p = scopes(cohort['P'])
        result['contrasts'].update({f'P-{label}': {scope: contrast(p[scope], views, repeats)
                                                   for scope, views in scopes(cohort[label]).items()}
                                    for label in cohort if label not in ('J', 'P', 'GT')})
    return result


# ---- loading --------------------------------------------------------------------------------------------------

def _json(path):
    return h.strict_json(Path(path).read_text())


def load(data, geometry, runs, agents=None, s1=(), workers=1):
    """Fail-closed inputs -> ({label: test views}, {label: cohort views} or None, provenance, raw room types)."""
    data, geometry = Path(data), Path(geometry)
    manifest = e._verified(data)
    expected, data_sha = manifest['expected_uids'], h.sha256(data / 'MANIFEST.json')
    rooms = h._by_uid(h._rows(data, 'rooms.jsonl', manifest['files']['rooms.jsonl']))
    requests = {arm: h._by_uid(h._rows(data, arm + '/requests.jsonl', manifest['files'][arm + '/requests.jsonl']))
                for arm in h.ARMS}
    gman, summary = e._verified(geometry), _json(geometry / 'summary.json')
    if summary.get('status') != 'GEOMETRY_COMPLETE' or summary.get('data_manifest_sha256') != data_sha:
        raise ValueError('geometry is not complete for this evaluation data')
    r2._same_geometry(summary, 'geometry')  # GT is scored by the running geometry.py
    rows = {arm: h._rows(geometry, arm + '-geometry.jsonl', gman['files'][arm + '-geometry.jsonl']) for arm in h.ARMS}
    if any([r['uid'] for r in rows[arm]] != expected or any(r['arm'] != arm for r in rows[arm]) for arm in h.ARMS):
        raise ValueError('geometry rows do not match the frozen inventory')
    if set(runs) != set(h.ARMS):
        raise ValueError('pass both merged inference runs')
    caps, flipped = {}, {}
    for arm in h.ARMS:
        records, receipt = e._inference(Path(runs[arm]), data, expected)
        if receipt['run_sha256'] != summary['inference_run_sha256'][arm]:
            raise ValueError('geometry does not bind this inference run')
        caps[arm] = [r['reached_token_cap'] and not r['stopped_on_eos'] for r in records]
        rows[arm], flipped[arm] = r2.cap_rule(rows[arm], records)
    labels = {'joint': 'J', 'placement': 'P'}
    test = {labels[arm]: r2._map(_scored, [(r, rooms[r['uid']], requests[arm][r['uid']]) for r in rows[arm]], workers)
            for arm in h.ARMS}
    test['GT'] = r2._map(_gt, [rooms[u] for u in expected], workers)
    inputs = {'data_MANIFEST': data_sha, 'geometry_MANIFEST': h.sha256(geometry / 'MANIFEST.json')}
    for root in map(Path, s1):
        man, info = e._verified(root), _json(root / 'summary.json')
        if (info.get('status') != 'S1_COMPLETE' or info['inputs']['data_manifest_sha256'] != data_sha
                or info['inputs']['geometry_manifest_sha256'] != inputs['geometry_MANIFEST']):
            raise ValueError(f'{root}: S1 output does not bind this geometry and data')
        arm, name = info['requests_arm'], Path(info['inputs']['geometry']).stem + '-s1.jsonl'
        if labels[arm] + '+S1' in test:
            raise ValueError(f'{root}: a second S1 output for {arm}')
        repaired = h._rows(root, name, man['files'][name])
        if [r['uid'] for r in repaired] != expected:
            raise ValueError(f'{root}: S1 rows do not cover the frozen inventory in order')
        test[labels[arm] + '+S1'] = [_s1(r, rooms[r['uid']], c) for r, c in zip(repaired, caps[arm])]
        inputs[labels[arm] + '+S1_MANIFEST'] = h.sha256(root / 'MANIFEST.json')
    cohort, uids, agent_flipped = None, None, {}
    for name, root in (agents or {}).items():
        variants, info, ids = r2._agent(Path(root), data_sha, summary['inference_run_sha256'])
        if uids is not None and ids != uids:
            raise ValueError('agents were scored on different cohorts')
        uids = ids
        if cohort is None:
            cohort = {label: _subset(views, uids) for label, views in test.items() if label in ('J', 'P', 'GT')}
        flagged = any('reached_token_cap' in r and 'stopped_on_eos' in r for v in variants.values() for r in v)
        for variant, vrows in variants.items():
            label = f"{name}-{variant.removeprefix('llm_')}"
            capped = [flagged and r.get('reached_token_cap', False) and not r.get('stopped_on_eos', True) for r in vrows]
            agent_flipped[label] = sum(c and r['complete'] for c, r in zip(capped, vrows)) if flagged else None
            vrows = [r2.cap_failure(r) if c else r for r, c in zip(vrows, capped)]
            cohort[label] = r2._map(_scored, [(r, rooms[r['uid']], requests['joint'][r['uid']]) for r in vrows], workers)
        inputs[f'agent:{name}'] = {'MANIFEST': h.sha256(Path(root) / 'MANIFEST.json'), 'system': info.get('system'),
                                   'model': info.get('model')}
    provenance = {'inputs': inputs, 'stage': summary.get('stage', 'sft'),
                  'cap_rule': {'joint_placement': flipped,
                               'agents_flipped_complete': agent_flipped},
                  'cohort_rooms': len(uids) if uids else None}
    return test, cohort, provenance, [rooms[u].get('room_type') for u in expected]


# ---- output ---------------------------------------------------------------------------------------------------

def _f(value, signed=False):
    return '—' if value is None else (f'{value:+.3f}' if signed else f'{value:.3f}')


def _ci(item):
    if item is None or item['estimate'] is None:
        return '—'
    return _f(item['estimate'], True) + (f" [{item['ci95'][0]:+.3f}, {item['ci95'][1]:+.3f}]" if item['ci95'] else '')


def _level_lines(table):
    lines = ['| 范围 | 论文覆盖 | 系统 | 房间 | 房屋组 k | 完整率 | 检查器成功 | 下游接受 | B | 尺寸 log MAE | OOR | 标记 |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for scope, value in table.items():
        d = value['design']
        if not d['rooms']:
            continue
        seen = _paper(scope, '是', '否') if scope in FAMILIES else ''
        for label, m in value['systems'].items():
            lines.append(f"| {scope} | {seen} | {label} | {m['requests']} | {d['components']} | "
                         + ' | '.join(_f(m[k]) for k in FIELDS[:3]) + ' | ' + _f(m['bias_fp'], True) + ' | '
                         + _f(m['size_log_mae']) + ' | ' + _f(m['oor']) + f" | {d['flags'] or ''} |")
    return lines


def _contrast_lines(name, table):
    lines = [f'| 范围 | 房间 | k | ' + ' | '.join('Δ' + NAMES[f] for f in FIELDS) + ' | 标记 |',
             '|---|---:|---:|' + '---:|' * len(FIELDS) + '---|']
    for scope, value in table.items():
        if value['rooms']:
            lines.append(f"| {scope} | {value['rooms']} | {value['components']} | "
                         + ' | '.join(_ci(value['fields'].get(f)) for f in FIELDS) + f" | {value['flags'] or ''} |")
    return [f'### {name}', ''] + lines + ['']


def _source_lines(table):
    lines = ['| 范围 | 房间 | 房屋组 k | 来源（房间数，占比） |', '|---|---:|---:|---|']
    for scope, value in table.items():
        d = value['design']
        if d['rooms']:
            mix = sorted(d['sources'].items(), key=lambda x: (-x[1], x[0]))
            lines.append(f"| {scope} | {d['rooms']} | {d['components']} | "
                         + '; '.join(f'{s} {n} ({n / d["rooms"]:.0%})' for s, n in mix) + ' |')
    return lines


def report(result):
    p = result['provenance']
    lines = ['# 按房型族的结果（EXPLORATORY，事后分析，不属于预注册方案）', '',
             f"OptiScene 论文只覆盖 {' / '.join(PAPER_FAMILIES)} 四类房型（eval_data.PAPER_STRATA）；本测试集有 "
             f"{len(FAMILIES)} 个房型族。“论文覆盖 = 否”的族论文没有评测；OptiScene 公开的 SFT 和 DPO2 训练提示只有这四类，"
             'DPO1（7,951 对）另有 Library（study_office）和 Hallway + Corridor（circulation）各 16 对（各约 0.2%）。'
             '房型族由原始 room_type 经 scene.norm_room_type 后的确定性规则得到（mapping.md 列出每个原始字符串及房间数；'
             '规则见 room_families.py）。none = 没有 room_type（真实扫描，房型未知）；other = 有名称但无法归类'
             '（other room、misc.、studio、apartment 等）；二者论文覆盖记为“—”，不计入 other_families（与 E3 相同）。', '',
             '各族结果很大程度上反映房间来自哪个数据源（表 0）。已知局限：论文四族里也有 OptiScene 没见过的商业/办公变体，'
             '留在论文族里，因此也不在 E3 中（eval-data-v1 上约 411 间：酒店、宿舍和标准间 128，办公和公共卫生间 111，'
             '游戏、娱乐和休息室约 103，办公茶水间 69）。', '',
             f"系统：J（联合预测尺寸）、P（训练集尺寸先验 + 模型布局，交付模型）、GT（测试集真值布局本身）。模型阶段：{p['stage']}。"
             '完整率、检查器成功、下游接受以范围内全部请求为分母；B（足迹 log 偏差，负 = 偏小）、尺寸 log MAE、OOR（两两 XY 重叠率，'
             '不是越界率）只在各系统自己的完整房间上平均。检查器成功 = 冻结几何检查器；下游接受 = downstream.accept'
             '（EmbodiedGen Path A 规则移植，每扇门都作为入口），都在原始输出上计算。', '',
             f"差值为前者减后者，房屋连通组自举（{result['repeats']:,} 次，seed {result['seed']}，按来源签名分层，与 geometry.paired_intervals 相同的设计和抽样），"
             f"标量只在两者都完整的房间上配对。房屋组 k < {NO_CI} 时不给区间；k < {r2.FEW_CLUSTERS} 标记 few clusters，"
             f"房间 < {r2.SMALL_N} 标记 small n（report_v2._design）。"
             '区间警告（如 singleton_source_signature_strata）见 contrasts.csv。各族之间不做多重比较校正；所有数字仅作描述。', '',
             f"截断规则（修订 005，达上限且无 EOS 记为失败）：J/P 被翻转的原完整房间 {p['cap_rule']['joint_placement']}；"
             f"agent 按行（行内带标志时）{p['cap_rule']['agents_flipped_complete']}。", '',
             '## 表 0：各范围的来源构成（全部测试房间）', ''] + _source_lines(result['test']) + ['',
             '## 表 1：全部测试房间，按房型族', ''] + _level_lines(result['test']) + ['',
             '## 表 2：配对差值', ''] + _contrast_lines('J − P（全部测试房间）', result['contrasts']['J-P'])
    if 'cohort' in result:
        lines += [f"## 表 3：同输入 agent 队列（{p['cohort_rooms']} 个房间），按房型族", '',
                  '队列不代表整个测试集（各来源最多 30 个房间加论文分层）；agent 使用 J 的请求（无尺寸），P 额外拿到尺寸先验。', '']
        lines += _level_lines(result['cohort']) + ['']
        for name, table in result['contrasts'].items():
            if name != 'J-P':
                lines += _contrast_lines(name.replace('-', ' − ', 1) + '（队列）', table)
    return '\n'.join(lines) + '\n'


def _csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write(out, result, mapping):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    levels = [{'population': pop, 'scope': scope, 'system': label, **value['design'],
               'sources': json.dumps(value['design']['sources'], sort_keys=True), **m}
              for pop in ('test', 'cohort') if pop in result for scope, value in result[pop].items()
              for label, m in value['systems'].items()]
    contrasts = [{'contrast': name, 'scope': scope, 'rooms': v['rooms'], 'components': v['components'],
                  'flags': v['flags'], 'field': f, 'estimate': x['estimate'],
                  'ci_low': x['ci95'][0] if x['ci95'] else None, 'ci_high': x['ci95'][1] if x['ci95'] else None,
                  'paired_n': x['n'], 'warnings': ';'.join(x['warnings'])}
                 for name, table in result['contrasts'].items() for scope, v in table.items()
                 for f, x in v['fields'].items()]
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.' + out.name + '-', dir=out.parent) as tmp:
        dest = Path(tmp)
        h.write_json(dest / 'families.json', result)
        (dest / 'report.md').write_text(report(result))
        (dest / 'mapping.md').write_text(mapping)
        _csv(dest / 'levels.csv', levels)
        _csv(dest / 'contrasts.csv', contrasts)
        h.write_json(dest / 'MANIFEST.json', {'files': h.artifact_manifest(dest)})
        _publish(dest, out)


def run(data, geometry, out, runs, agents=None, s1=(), repeats=2000, workers=1):
    if Path(out).exists():
        raise FileExistsError(out)
    test, cohort, provenance, room_types = load(data, geometry, runs, agents, s1, workers)
    result = {'status': 'ROOM_FAMILIES_COMPLETE', 'label': 'EXPLORATORY', **build(test, cohort, repeats),
              'provenance': provenance, 'families': list(FAMILIES), 'paper_families': list(PAPER_FAMILIES),
              'family_counts': dict(collections.Counter(map(family, room_types))),
              'repeats': repeats, 'seed': SEED, 'small_n': r2.SMALL_N, 'few_clusters': r2.FEW_CLUSTERS,
              'no_ci_components': NO_CI, 'code_sha256': {Path(p).name: h.sha256(p) for p in (
                  __file__, scene.__file__, g.__file__, dn.__file__, r2.__file__)}}
    write(out, result, mapping_table(room_types))
    return result


def _pairs(values):
    return dict(v.split('=', 1) for v in values or ())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    table = sub.add_parser('mapping')
    table.add_argument('--rooms', type=Path, required=True)
    rep = sub.add_parser('report')
    for flag in ('data', 'geometry', 'out'):
        rep.add_argument('--' + flag, type=Path, required=True)
    rep.add_argument('--joint-run', type=Path, required=True)
    rep.add_argument('--placement-run', type=Path, required=True)
    rep.add_argument('--agent', action='append', metavar='NAME=DIR')
    rep.add_argument('--s1', action='append', type=Path, default=[])
    rep.add_argument('--repeats', type=int, default=2000)
    rep.add_argument('--workers', type=int, default=1)
    args = parser.parse_args(argv)
    if args.command == 'mapping':
        with args.rooms.open() as stream:
            sys.stdout.write(mapping_table(json.loads(line).get('room_type') for line in stream))
        return
    runs = {'joint': args.joint_run, 'placement': args.placement_run}
    result = run(args.data, args.geometry, args.out, runs, _pairs(args.agent), args.s1, args.repeats, args.workers)
    print(json.dumps({'status': result['status'], 'family_counts': result['family_counts']}))


if __name__ == '__main__':
    main()
