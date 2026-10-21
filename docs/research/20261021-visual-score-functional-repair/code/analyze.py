import json,numpy as np,collections
from scipy.stats import spearmanr
B='/Volumes/harddisk/fastfillout/unknownsize-20261013/'
O=B+'probes-20261021/clip-metric/'
rooms=json.load(open(O+'rooms.json')); items=json.load(open(O+'items.json')); utexts=json.load(open(O+'utexts.json'))
tix={t:i for i,t in enumerate(utexts)}
G={}
for l in open(B+'a021-test-geometry-arm/placement-geometry.jsonl'):
    d=json.loads(l); G[d['uid']]=d
picks=json.load(open(B+'probes-20261021/s2-render/picks.json'))
better={2,3,11,14}; worse={1,5,6,9,12,15,17,18,19}
rng=np.random.default_rng(0)
img_ix={(s,u,l,v):i for i,(s,u,l,v,p) in enumerate(items)}
rows=[];out=[]
def auc(pos,neg):  # P(score_pos > score_neg)
    pos=np.asarray(pos);neg=np.asarray(neg)
    d=pos[:,None]-neg[None,:]; return float(((d>0).sum()+.5*(d==0).sum())/d.size)
def boot(x,n=5000):
    x=np.asarray(x); m=[x[rng.integers(0,len(x),len(x))].mean() for _ in range(n)]; return float(x.mean()),float(np.percentile(m,2.5)),float(np.percentile(m,97.5))
res={}
for tag in ('clip','siglip'):
    z=np.load(O+f'emb_{tag}.npz'); IE,TE=z['img'],z['txt']
    def sc(s,u,l,v,key):
        r=rooms[('R20|' if s=='R20' else 'G|')+u]; return 100*float(IE[img_ix[(s,u,l,v)]]@TE[tix[r['texts'][key]]])
    def isim(s,u,l,v): return 100*float(IE[img_ix[(s,u,l,v)]]@IE[img_ix[(s,u,'reference',v)]])
    for (s,u,l,v,p) in items:
        r=rooms[('R20|' if s=='R20' else 'G|')+u]; t=r['texts']
        row=dict(model=tag,set=s,uid=u,layout=l,view=v,room_type=r['rt'],T1=sc(s,u,l,v,'T1'),T2=sc(s,u,l,v,'T2'),
          T1_wrongrt=float(np.mean([sc(s,u,l,v,k) for k in t if k.startswith('T1_wrongrt|')])),
          T2_wrongrt=float(np.mean([sc(s,u,l,v,k) for k in t if k.startswith('T2_wrongrt|')])),
          T1_shuf_same=sc(s,u,l,v,'T1_shuf_same'),T1_shuf_other=sc(s,u,l,v,'T1_shuf_other'),
          T2_rt_correct=bool(sc(s,u,l,v,'T2')>max(sc(s,u,l,v,k) for k in t if k.startswith('T2_wrongrt|'))),
          imgsim_to_ref=None if l=='reference' else isim(s,u,l,v))
        if s=='R20': row['pick_idx']=r['idx']
        out.append(row)
    R=[r for r in out if r['model']==tag]
    D={(r['set'],r['uid'],r['layout'],r['view']):r for r in R}
    m={}
    # ---------- GTvsRAW
    uids=sorted({r['uid'] for r in R if r['set']=='G' and r['layout']=='raw'})
    viol=np.array([G[u]['main_collisions']+G[u]['oob_objects']+G[u]['fixed_blocking']+G[u]['support_fail'] for u in uids])
    ok=np.array([bool(G[u]['geometry_success']) for u in uids]); m['G_n']=len(uids); m['G_pass']=int(ok.sum()); m['G_viol_gt0']=int((viol>0).sum())
    rtv=np.array([rooms['G|'+u]['rt'] for u in uids])
    for view in ('perspective','top'):
        for key in ('T1','T2','T1_shuf_same','T1_shuf_other','T1_wrongrt'):
            g=np.array([D[('G',u,'reference',view)][key] for u in uids]); r_=np.array([D[('G',u,'raw',view)][key] for u in uids]); d=g-r_
            mn,lo,hi=boot(d)
            sp=spearmanr(d,viol)
            # room-type centred raw score for AUC
            rc=r_.copy()
            for t in set(rtv): rc[rtv==t]-=r_[rtv==t].mean()
            m[f'G|{view}|{key}']=dict(share_gt_gt_raw=float((d>0).mean()),mean_diff=mn,ci=[lo,hi],spearman_diff_viol=[float(sp[0]),float(sp[1])],
               auc_raw_pass_vs_fail=auc(r_[ok],r_[~ok]),auc_roomtype_centred=auc(rc[ok],rc[~ok]),mean_gt=float(g.mean()),mean_raw=float(r_.mean()))
        isimv=np.array([D[('G',u,'raw',view)]['imgsim_to_ref'] for u in uids]); sp=spearmanr(isimv,viol)
        m[f'G|{view}|imgsim']=dict(mean=float(isimv.mean()),spearman_sim_viol=[float(sp[0]),float(sp[1])],auc_pass_vs_fail=auc(isimv[ok],isimv[~ok]),
            mean_pass=float(isimv[ok].mean()),mean_fail=float(isimv[~ok].mean()))
        # variance decomposition of T1: within-room (GT vs raw) vs between room
        g=np.array([D[('G',u,'reference',view)]['T1'] for u in uids]); r_=np.array([D[('G',u,'raw',view)]['T1'] for u in uids])
        m[f'G|{view}|T1_sd']=dict(between_room=float(((g+r_)/2).std()),within_pair_sd=float(((g-r_)/np.sqrt(2)).std()))
    # ---------- negative control (all images in G, pooled)
    nc={}
    for view in ('perspective','top'):
        sub=[r for r in R if r['view']==view]
        nc[view]={k:float(np.mean([r[k] for r in sub])) for k in ('T1','T1_wrongrt','T1_shuf_same','T1_shuf_other','T2','T2_wrongrt')}
        nc[view]['roomtype_zero_shot_acc']=float(np.mean([r['T2_rt_correct'] for r in sub]))
        byrt={}
        for t in sorted(set(r['room_type'] for r in sub)): byrt[t]=float(np.mean([r['T2_rt_correct'] for r in sub if r['room_type']==t]))
        nc[view]['acc_by_roomtype']=byrt
    m['negctl']=nc
    # ---------- R20
    r20={}
    idx=[p['uid'] for p in picks]
    for view in ('perspective','top'):
        for key in ('T1','T2','imgsim_to_ref'):
            d=np.array([D[('R20',u,'S2',view)][key]-D[('R20',u,'S1',view)][key] for u in idx])
            lab=np.array([1 if (i+1) in better else -1 if (i+1) in worse else 0 for i in range(20)])
            agree=int(((np.sign(d)==lab)&(lab!=0)).sum()); nl=int((lab!=0).sum())
            tb=int(((d>0)&(lab==1)).sum()); tw=int(((d<0)&(lab==-1)).sum())
            sp=spearmanr(d[lab!=0],lab[lab!=0])
            r20[f'{view}|{key}']=dict(diff=[round(float(x),2) for x in d],agree=agree,n=nl,better_ok=f'{tb}/4',worse_ok=f'{tw}/9',spearman=float(sp[0]),
                same_mean_absdiff=float(np.abs(d[lab==0]).mean()),labelled_mean_absdiff=float(np.abs(d[lab!=0]).mean()),
                auc_better_vs_worse=auc(d[lab==1],d[lab==-1]),mean_d_better=float(d[lab==1].mean()),mean_d_worse=float(d[lab==-1].mean()))
        for key in ('T1','T2'):
            ranks=collections.defaultdict(list); top1=0
            for u in idx:
                s={l:D[('R20',u,l,view)][key] for l in ('reference','raw','S1','S2')}
                order=sorted(s,key=lambda l:-s[l]); top1+=order[0]=='reference'
                for l in s: ranks[l].append(order.index(l)+1)
            r20[f'{view}|{key}|rank']=dict(mean_rank={l:float(np.mean(v)) for l,v in ranks.items()},ref_top1=top1)
        ranks=collections.defaultdict(list)
        for u in idx:
            s={l:D[('R20',u,l,view)]['imgsim_to_ref'] for l in ('raw','S1','S2')}; order=sorted(s,key=lambda l:-s[l])
            for l in s: ranks[l].append(order.index(l)+1)
        r20[f'{view}|imgsim|rank']={l:float(np.mean(v)) for l,v in ranks.items()}
    m['R20']=r20
    res[tag]=m
json.dump(res,open(O+'results.json','w'),indent=1)
with open(O+'scores.jsonl','w') as f:
    for r in out: f.write(json.dumps({k:(round(v,4) if isinstance(v,float) else v) for k,v in r.items()})+'\n')
print(json.dumps(res,indent=1))
