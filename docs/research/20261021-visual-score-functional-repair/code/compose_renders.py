"""One comparison PNG per rendered room: columns GT / 27B raw / S1 / S2, rows perspective / top, plus a header."""
import json, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

O = Path(sys.argv[1])
picks = {p['uid']: p for p in json.load(open(O / 'picks.json'))}
COLS = [('reference', 'GT'), ('raw', '27B 原始'), ('S1', '+S1'), ('S2', '+S2')]
FONT = '/System/Library/Fonts/STHeiti Medium.ttc'
font, small = ImageFont.truetype(FONT, 26), ImageFont.truetype(FONT, 20)
out = O / 'compare'
out.mkdir(exist_ok=True)
G = {'fix-checker': 'S2 修好 checker', 'fix-strict': 'S2 修好 strict', 'still-fail': 'S2 改了但仍失败'}
n = 0
for i in (0, 1):
    ledger = json.load(open(O / f'render-{i}' / 'ledger.json'))
    for case in ledger['cases']:
        p = picks[case['uid']]
        k = len(json.load(open(O / 'picks.json'))) and [x['uid'] for x in json.load(open(O / 'picks.json'))].index(case['uid'])
        if case['status'] != 'rendered':
            print('not rendered', case['uid'], case.get('reason'))
            continue
        W, Hh, top = 512, 512, 90
        sheet = Image.new('RGB', (W * 4, top + 40 + Hh * 2), 'white')
        d = ImageDraw.Draw(sheet)
        ok = lambda v: f"checker {'过' if v[0] else '不过'} / strict {'过' if v[1] else '不过'}"
        d.text((12, 8), f"#{k + 1:02d}  {G[p['group']]}   {p['source']}  {p['objects']} 件", fill='black', font=font)
        d.text((12, 48), f"S1: {ok(p['S1'])}    S2: {ok(p['S2'])}    {p['uid'][:60]}", fill='#444', font=small)
        for c, (key, label) in enumerate(COLS):
            d.text((c * W + 12, top + 6), label, fill='#b00' if key == 'S2' else 'black', font=font)
            for r, view in enumerate(('perspective', 'top')):
                sheet.paste(Image.open(case['images'][key][view]).convert('RGB').resize((W, Hh)), (c * W, top + 40 + r * Hh))
        sheet.save(out / f"{k + 1:02d}-{p['group']}.png")
        n += 1
print(n, 'sheets in', out)
