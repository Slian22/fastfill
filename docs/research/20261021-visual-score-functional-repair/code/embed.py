import json,glob,random,collections,sys,numpy as np,torch
from PIL import Image
B='/Volumes/harddisk/fastfillout/unknownsize-20261013/'
O=B+'probes-20261021/clip-metric/'
S=B+'probes-20261021/s2-render/'
# ---- rooms
rooms={}   # key -> dict(set,uid,rt,cats list)
items=[]   # image items
def rt_of(strat): return strat.split('/')[-1]
def cats(layout): return [o['category'] for o in layout]
man={}
for i in (0,1):
    for c in json.load(open(S+f'manifest-{i}.json'))['cases']: man[c['uid']]=c
picks=json.load(open(S+'picks.json'))
for n,p in enumerate(picks,1):
    c=man[p['uid']]; rooms['R20|'+p['uid']]=dict(set='R20',uid=p['uid'],idx=n,rt=rt_of(c['stratum']),cats=cats(c['layouts']['reference']))
for i in (0,1):
    for c in json.load(open(S+f'render-{i}/ledger.json'))['cases']:
        for lay,v in c['images'].items():
            for view,pth in v.items(): items.append(('R20',c['uid'],lay,view,pth))
cs={json.loads(l)['uid']:json.loads(l) for l in open(B+'visual-024/A/cases.jsonl')}
A={};R={}
for p in glob.glob(B+'visual-024/A/chunks/*/render/ledger.json'):
    for c in json.load(open(p))['cases']: A[c['uid']]=c
for p in glob.glob(B+'visual-final-selected-002/visual/chunks/*/render/ledger.json'):
    for c in json.load(open(p))['cases']: R[c['uid']]=c
for u,a in A.items():
    c=cs[u]; assert a['cameras']==R[u]['cameras'] and 'A' in a['images']
    rooms['G|'+u]=dict(set='G',uid=u,rt=rt_of(c['stratum']),cats=cats(c['layouts']['reference']))
    for view,pth in a['images']['A'].items(): items.append(('G',u,'raw',view,pth))
    for view,pth in R[u]['images']['reference'].items(): items.append(('G',u,'reference',view,pth))
print(len(rooms),len(items))
# ---- texts
def lst(cs_):
    cnt=collections.Counter(c.lower().replace('_',' ') for c in cs_); return ', '.join(f'{n} {c}' for c,n in cnt.most_common(8))
RT=sorted({r['rt'] for r in rooms.values()}); print(RT)
rng=random.Random(0)
pool={t:[k for k,r in rooms.items() if r['rt']==t] for t in RT}
allk=list(rooms)
texts={}
def T1(rt,l): return f'a 3D rendering of a {rt} with {l}'
for k,r in rooms.items():
    l=lst(r['cats']); t={}
    t['T1']=T1(r['rt'],l); t['T2']=f"a photo of a {r['rt']}"
    for w in RT:
        if w!=r['rt']: t['T1_wrongrt|'+w]=T1(w,l); t['T2_wrongrt|'+w]=f'a photo of a {w}'
    o=rng.choice([x for x in pool[r['rt']] if x!=k]); t['T1_shuf_same']=T1(r['rt'],lst(rooms[o]['cats']))
    o=rng.choice([x for x in allk if rooms[x]['rt']!=r['rt']]); t['T1_shuf_other']=T1(r['rt'],lst(rooms[o]['cats']))
    r['texts']=t
json.dump({k:{kk:vv for kk,vv in r.items()} for k,r in rooms.items()},open(O+'rooms.json','w'))
json.dump(items,open(O+'items.json','w'))
if len(sys.argv)>1 and sys.argv[1]=='prep': sys.exit()
# ---- embed
from transformers import CLIPModel,CLIPProcessor,AutoModel,AutoProcessor
dev='mps'
utexts=sorted({s for r in rooms.values() for s in r['texts'].values()})
for name,tag in (("openai/clip-vit-large-patch14",'clip'),("google/siglip-so400m-patch14-384",'siglip')):
    M=(CLIPModel if tag=='clip' else AutoModel).from_pretrained(name).to(dev).eval(); P=(CLIPProcessor if tag=='clip' else AutoProcessor).from_pretrained(name)
    pad='max_length' if tag=='siglip' else True
    def nrm(x): return torch.nn.functional.normalize(x.float(),dim=-1).cpu().numpy()
    with torch.no_grad():
        te=[]
        for i in range(0,len(utexts),64):
            b=P.tokenizer(utexts[i:i+64],padding=pad,truncation=True,max_length=77 if tag=='clip' else 64,return_tensors='pt').to(dev)
            te.append(nrm(M.get_text_features(**b)))
        te=np.concatenate(te); ie=[]
        for i in range(0,len(items),32):
            ims=[Image.open(x[4]).convert('RGB') for x in items[i:i+32]]
            ie.append(nrm(M.get_image_features(**P.image_processor(ims,return_tensors='pt').to(dev))))
            if i%640==0: print(tag,i,flush=True)
        ie=np.concatenate(ie)
    ls=float(M.logit_scale.exp()) ; bias=float(M.logit_bias) if hasattr(M,'logit_bias') else 0.
    np.savez(O+f'emb_{tag}.npz',img=ie,txt=te,ls=ls,bias=bias)
    json.dump(utexts,open(O+'utexts.json','w'))
    del M; torch.mps.empty_cache()
