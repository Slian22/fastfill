"""CLIP scores for every rendered system on the visual cohort, Holodeck protocol (OpenCLIP ViT-L/14 laion2b_s32b_b82k,
'a top-down view of {room type}', top view, cosine x100) plus a furniture-list prompt on both views.
Same rooms, renderer, cameras and prompts for all systems. python3 clip_systems.py OUT"""
import collections, glob, json, os, sys
import numpy as np, torch
from PIL import Image
os.environ.setdefault('HF_HOME', '/Volumes/harddisk/fastfillout/hf-cache')
import open_clip

B = '/Volumes/harddisk/fastfillout/unknownsize-20261013/'
out = sys.argv[1]; os.makedirs(out, exist_ok=False)
cases = {json.loads(l)['uid']: json.loads(l) for l in open(B + 'visual-024/A/cases.jsonl')}
imgs = collections.defaultdict(dict)   # uid -> system -> {view: path}
for p in glob.glob(B + 'visual-final-selected-002/visual/chunks/*/render/ledger.json'):
    for c in json.load(open(p))['cases']:
        if c.get('status') == 'rendered':
            for k, v in c['images'].items():
                imgs[c['uid']][{'reference': 'GT'}.get(k, k)] = v
for arm, name in (('A', '27B-raw'), ('S', '27B-system')):
    for p in glob.glob(B + f'visual-024/{arm}/chunks/*/render/ledger.json'):
        for c in json.load(open(p))['cases']:
            if c.get('status') == 'rendered':
                imgs[c['uid']][name] = c['images'][arm]
rt = lambda u: cases[u]['stratum'].split('/')[-1].replace('_', ' ')
def furn(u):
    cnt = collections.Counter(o['category'].lower().replace('_', ' ') for o in cases[u]['layouts']['reference'])
    return ', '.join(f'{n} {c}' for c, n in cnt.most_common(8))
texts = {u: {'holodeck': f'a top-down view of {rt(u)}', 'furniture': f'a 3D rendering of a {rt(u)} with {furn(u)}'} for u in imgs}
dev = 'mps'
model, _, pre = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k', device=dev)
tok = open_clip.get_tokenizer('ViT-L-14'); model.eval()
ut = sorted({t for d in texts.values() for t in d.values()})
with torch.no_grad():
    te = torch.cat([model.encode_text(tok(ut[i:i + 64]).to(dev)) for i in range(0, len(ut), 64)])
    te = torch.nn.functional.normalize(te.float(), dim=-1)
ti = {t: i for i, t in enumerate(ut)}
items = [(u, s, view, p) for u, d in imgs.items() for s, v in d.items() for view, p in v.items()]
rows = []
with torch.no_grad():
    for i in range(0, len(items), 32):
        b = items[i:i + 32]
        x = torch.stack([pre(Image.open(p).convert('RGB')) for *_, p in b]).to(dev)
        ie = torch.nn.functional.normalize(model.encode_image(x).float(), dim=-1)
        for (u, s, view, p), e in zip(b, ie):
            rows.append({'uid': u, 'system': s, 'view': view,
                         **{k: round(100 * float(e @ te[ti[t]]), 4) for k, t in texts[u].items()}})
        if i % 1600 == 0: print(i, len(items), flush=True)
open(out + '/scores.jsonl', 'w').write(''.join(json.dumps(r) + '\n' for r in rows))
common = set.intersection(*[{r['uid'] for r in rows if r['system'] == s} for s in {r['system'] for r in rows}])
lines = [f'# CLIP 分（同一批 {len(common)} 间房、同一渲染、同一文字；OpenCLIP ViT-L/14 laion2b，×100）', '',
         '| 系统 | Holodeck 口径（俯视，"a top-down view of 房型"） | 家具清单文字，俯视 | 家具清单文字，透视 |', '|---|---:|---:|---:|']
for s in sorted({r['system'] for r in rows}):
    rs = [r for r in rows if r['system'] == s and r['uid'] in common]
    m = lambda view, k: np.mean([r[k] for r in rs if r['view'] == view])
    lines.append(f"| {s} | {m('top','holodeck'):.2f} | {m('top','furniture'):.2f} | {m('perspective','furniture'):.2f} |")
open(out + '/report.md', 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines))
