"""Render manifests and comparison sheets for S2-F (layouts reference / S1 / S2 / S2F).

python3 s2f_render.py manifests      set A = s2-render/picks.json rooms; set B = 20 cohort rooms (not in A) whose S1
                                     layout S2-F changed, in sha256('render-s2f-20261021:'+uid) order; 2 chunks of 10 each
python3 s2f_render.py compose A|B    sheets from render-{A,B}-{0,1}/ledger.json into compare-{A,B}/ (+ 1400 px -small)
"""
import hashlib
import json
import sys
from pathlib import Path

P = Path('/Volumes/harddisk/fastfillout/unknownsize-20261013/probes-20261021')
O = P / 's2f'
CASES = Path('/Volumes/harddisk/fastfillout/unknownsize-20261013/visual-024/A/cases.jsonl')
COLS = [('reference', 'GT'), ('S1', '+S1'), ('S2', '+S2'), ('S2F', '+S2-F')]


def L(f):
    return [json.loads(l) for l in open(f)]


def manifests():
    sys.path.insert(0, str(P / 'scripts'))
    import companions as C
    s1 = {r['uid']: r for r in L(P / 's2-render/cohort-s1.jsonl')}
    s2 = {r['uid']: r for r in L(O / 'cohort/merged-s2.jsonl')}
    s2f = {r['uid']: r for r in L(O / 'cohort/final-s2f.jsonl')}
    gt = {r['uid']: r for r in L(Path('/Volumes/harddisk/fastfillout/unknownsize-20261013/visual-final-selected-002/inputs/eval-data-v1/rooms.jsonl'))
          if r['uid'] in s1}
    cases = {c['uid']: c for c in L(CASES)}
    a = [(p['uid'], p['group']) for p in json.load(open(P / 's2-render/picks.json'))]
    changed = [u for u in s1 if s2f[u]['s1_layout'] != s1[u]['s1_layout'] and u not in dict(a)]
    changed.sort(key=lambda u: hashlib.sha256(('render-s2f-20261021:' + u).encode()).hexdigest())
    b = [(u, 's2f-changed') for u in changed[:20]]
    print('set B pool (S2-F changed S1, not in A):', len(changed))
    flags = lambda r: [r['s1']['geometry_success_s1'], r['s1']['strict_ok_s1']]
    for name, picks in (('A', a), ('B', b)):
        info = []
        for k, (u, group) in enumerate(picks):
            lay = {'S1': s1[u]['s1_layout'], 'S2': s2[u]['s1_layout'], 'S2F': s2f[u]['s1_layout']}
            info.append({'k': k + 1, 'uid': u, 'group': group, 'source': s1[u]['source'], 'objects': s1[u]['objects'],
                         **{arm: flags(r[u]) for arm, r in (('S1', s1), ('S2', s2), ('S2F', s2f))},
                         'pairs_S1': len(C.pairs(lay['S1'])),
                         'ret_S1_pairs': {arm: C.retention(lay['S1'], lay[arm]) for arm in ('S2', 'S2F')},
                         'gt_score': {arm: C.gt_score(gt[u]['objects'], lay[arm]) for arm in ('S1', 'S2', 'S2F')}})
        (O / f'picks-{name}.json').write_text(json.dumps(info, indent=1))
        for i in (0, 1):
            out = []
            for p in info[i * 10:(i + 1) * 10]:
                c = cases[p['uid']]
                lay = {'S1': s1[p['uid']]['s1_layout'], 'S2': s2[p['uid']]['s1_layout'], 'S2F': s2f[p['uid']]['s1_layout']}
                out.append({**c, 'layouts': {'reference': c['layouts']['reference'], **lay}, 'stage': p['group']})
            (O / f'manifest-{name}-{i}.json').write_text(json.dumps({'cases': out}))
        print(name, [(p['k'], p['uid'][:40], p['S1'], p['S2'], p['S2F']) for p in info])


def compose(name):
    from PIL import Image, ImageDraw, ImageFont
    info = {p['uid']: p for p in json.load(open(O / f'picks-{name}.json'))}
    FONT = '/System/Library/Fonts/STHeiti Medium.ttc'
    font, small = ImageFont.truetype(FONT, 26), ImageFont.truetype(FONT, 20)
    out, outs = O / f'compare-{name}', O / f'compare-{name}-small'
    out.mkdir(exist_ok=True)
    outs.mkdir(exist_ok=True)
    ok = lambda v: f"checker {'过' if v[0] else '不过'} / strict {'过' if v[1] else '不过'}"
    r = lambda t: f'{t[0]}/{t[1]}'
    n = 0
    for i in (0, 1):
        for case in json.load(open(O / f'render-{name}-{i}' / 'ledger.json'))['cases']:
            p = info[case['uid']]
            if case['status'] != 'rendered':
                print('not rendered', case['uid'], case.get('reason'))
                continue
            W, top = 512, 120
            sheet = Image.new('RGB', (W * 4, top + 40 + W * 2), 'white')
            d = ImageDraw.Draw(sheet)
            d.text((12, 8), f"{name}#{p['k']:02d}  {p['group']}   {p['source']}  {p['objects']} 件   {p['uid'][:60]}", fill='black', font=font)
            d.text((12, 48), f"S1: {ok(p['S1'])}    S2: {ok(p['S2'])}    S2-F: {ok(p['S2F'])}", fill='#444', font=small)
            d.text((12, 80), f"companion pairs in S1 {p['pairs_S1']}; kept by S2 {r(p['ret_S1_pairs']['S2'])}, by S2-F "
                             f"{r(p['ret_S1_pairs']['S2F'])};  GT pairs held S1 {r(p['gt_score']['S1'])}, S2 {r(p['gt_score']['S2'])}, "
                             f"S2-F {r(p['gt_score']['S2F'])}", fill='#444', font=small)
            for c, (key, label) in enumerate(COLS):
                d.text((c * W + 12, top + 6), label, fill='#b00' if key == 'S2F' else 'black', font=font)
                for row, view in enumerate(('perspective', 'top')):
                    sheet.paste(Image.open(case['images'][key][view]).convert('RGB').resize((W, W)), (c * W, top + 40 + row * W))
            f = f"{p['k']:02d}-{p['group']}.png"
            sheet.save(out / f)
            sheet.resize((1400, round(sheet.height * 1400 / sheet.width))).save(outs / f)
            n += 1
    print(n, 'sheets in', out)


if __name__ == '__main__':
    manifests() if sys.argv[1] == 'manifests' else compose(sys.argv[2])
