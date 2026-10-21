"""Render manifests and comparison sheets for S2-F-task (layouts reference / S2F / S2F-task), cohort rooms only.

python3 s2f_task_render.py manifests   20 cohort rooms whose layout S2-F-task changed, in
                                       sha256('render-s2ftask-20261021:'+uid) order; 2 chunks of 10 (s2f_task_render.sh),
                                       plus reserves #21.. in manifest-2.json (one per pick the renderer rejects)
python3 s2f_task_render.py compose     sheets from render-{0,1}/ledger.json into compare/ (+ 1400 px compare-small/)
"""
import hashlib
import json
import math
import sys
from pathlib import Path

P = Path('/Volumes/harddisk/fastfillout/unknownsize-20261013/probes-20261021')
O = P / 's2f-task'
CASES = Path('/Volumes/harddisk/fastfillout/unknownsize-20261013/visual-024/A/cases.jsonl')
RESERVE = 1        # added after the first render: pick #8 rejected (RenderReject no_asset: fixed_0001)
COLS = [('reference', 'GT'), ('S2F', 'S2-F'), ('S2Ftask', '+S2-F-task')]


def L(f):
    return [json.loads(l) for l in open(f)]


def manifests():
    s2f = {r['uid']: r for r in L(P / 's2f/cohort/final-s2f.jsonl')}
    new = {r['uid']: r for r in L(O / 'cohort/final-s2ftask.jsonl')}
    rows = {r['uid']: r for r in L(O / 'cohort/rows.jsonl')}
    cases = {c['uid']: c for c in L(CASES)}
    changed = sorted((u for u in s2f if new[u]['s1_layout'] != s2f[u]['s1_layout']),
                     key=lambda u: hashlib.sha256(('render-s2ftask-20261021:' + u).encode()).hexdigest())
    print('pool (S2-F-task changed the layout):', len(changed))
    info = []
    for k, u in enumerate(changed[:20 + RESERVE]):
        a, b = s2f[u]['s1_layout'], new[u]['s1_layout']
        moved = [(o['category'], round(math.dist(o['pos'][:2], q['pos'][:2]), 2)) for o, q in zip(a, b) if o['pos'] != q['pos']]
        x = rows[u]
        info.append({'k': k + 1, 'uid': u, 'source': s2f[u]['source'], 'objects': s2f[u]['objects'], 'cats': x['cats'],
                     'a3_only': x['a3_only'], 'task': x['task'], 'checker': x['checker'], 'strict': x['strict'],
                     'retention': x['retention'], 'moved': moved, 'steps': [m['step'] for m in x['info']['moves']]})
    (O / 'picks.json').write_text(json.dumps(info, indent=1, ensure_ascii=False))
    for i in (0, 1, 2):
        out = []
        for p in info[i * 10:(i + 1) * 10]:
            c = cases[p['uid']]
            out.append({**c, 'layouts': {'reference': c['layouts']['reference'], 'S2F': s2f[p['uid']]['s1_layout'],
                                         'S2Ftask': new[p['uid']]['s1_layout']}, 'stage': 's2ftask-changed'})
        (O / f'manifest-{i}.json').write_text(json.dumps({'cases': out}))
    for p in info:
        print(p['k'], p['uid'][:48], p['cats'], p['moved'])


def compose():
    from PIL import Image, ImageDraw, ImageFont
    info = {p['uid']: p for p in json.load(open(O / 'picks.json'))}
    FONT = '/System/Library/Fonts/STHeiti Medium.ttc'
    font, small = ImageFont.truetype(FONT, 26), ImageFont.truetype(FONT, 19)
    out, outs = O / 'compare', O / 'compare-small'
    out.mkdir(exist_ok=True)
    outs.mkdir(exist_ok=True)
    ok = lambda v: '过' if v else '不过'
    n = 0
    for i in (0, 1, 2):
        for case in json.load(open(O / f'render-{i}' / 'ledger.json'))['cases']:
            p = info[case['uid']]
            if case['status'] != 'rendered':
                print('not rendered', case['uid'], case.get('reason'))
                continue
            W, top = 512, 150
            sheet = Image.new('RGB', (W * 3, top + 40 + W * 2), 'white')
            d = ImageDraw.Draw(sheet)
            d.text((12, 8), f"#{p['k']:02d}  {p['source']}  {p['objects']} 件   {p['uid'][:52]}", fill='black', font=font)
            d.text((12, 46), f"任务物体: {', '.join(p['cats'])}    任务准入 S2-F {ok(p['task'][0])} → {ok(p['task'][1])};  checker "
                             f"{ok(p['checker'][0])} → {ok(p['checker'][1])};  搭配保持 {p['retention'][0]}/{p['retention'][1]}",
                   fill='#444', font=small)
            mv = ', '.join(f'{c[:22]} {m} m' for c, m in p['moved'])
            d.text((12, 76), f"移动 {len(p['moved'])} 件 ({'+'.join(sorted(set(p['steps'])))}): {mv[:150]}", fill='#444', font=small)
            if len(mv) > 150:
                d.text((12, 104), mv[150:300], fill='#444', font=small)
            for c, (key, label) in enumerate(COLS):
                d.text((c * W + 12, top + 6), label, fill='#b00' if key == 'S2Ftask' else 'black', font=font)
                for row, view in enumerate(('perspective', 'top')):
                    sheet.paste(Image.open(case['images'][key][view]).convert('RGB').resize((W, W)), (c * W, top + 40 + row * W))
            f = f"{p['k']:02d}.png"
            sheet.save(out / f)
            sheet.resize((1400, round(sheet.height * 1400 / sheet.width))).save(outs / f)
            n += 1
    print(n, 'sheets in', out)


if __name__ == '__main__':
    manifests() if sys.argv[1] == 'manifests' else compose()
