"""Generic blind pairwise judge (prompt/composition identical to judge_s1_s2.py). Budget + call log shared via calls.jsonl."""
import base64, hashlib, io, json, time, urllib.request, datetime
from pathlib import Path
from PIL import Image
P = Path('/Volumes/harddisk/fastfillout/unknownsize-20261013/probes-20261021'); J = P / 'judge'; LOG = J / 'calls.jsonl'
C = json.load(open('/Users/slian/Desktop/3D/Worldedge/OptiScene/llm_config.json'))
MAX_OK, MAX_BUSY = 300, 300
PROMPT = ("You see two renderings of the SAME room with the SAME furniture, arranged two ways: Layout A (first image) and "
          "Layout B (second image). Each image shows a perspective view (left) and a top-down view (right). "
          "Judge which arrangement is more realistic and functional for people living there: furniture grouped sensibly "
          "(e.g. nightstands beside the head of the bed, TV or TV stand facing the bed or sofa, chairs at their desk/table, "
          "toilet and fixtures placed as in a real bathroom), clear walking paths and doors, and no furniture passing "
          "through other furniture. Reply with JSON only: {\"winner\": \"A\" | \"B\" | \"tie\", \"reason\": \"<= 30 words\"}.")

def counts():
    ok = busy = 0
    if LOG.exists():
        for l in LOG.read_text().splitlines():
            s = json.loads(l)['status']; ok += s == 'ok'; busy += s == 'busy'
    return ok, busy

def log(**r):
    r['time'] = datetime.datetime.now().isoformat(timespec='seconds')
    with open(LOG, 'a') as f: f.write(json.dumps(r, ensure_ascii=False) + '\n')

def post(content, kind, max_tokens=None):
    """One HTTP request. Returns (status, text, usage); status ok|busy|error. Logged."""
    body = {'model': C['model'], 'reasoning_effort': C['reasoning_effort'], 'messages': [{'role': 'user', 'content': content}]}
    req = urllib.request.Request(C['chat_completions_url'], data=json.dumps(body).encode(), headers={
        'Content-Type': 'application/json', 'Authorization': 'Bearer ' + C['api_key'], 'User-Agent': 'curl/8.7.1'})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=600))
    except Exception as e:
        log(kind=kind, status='error', detail=repr(e)[:200]); return 'error', None, None
    if r.get('type') == 'error' or 'choices' not in r:
        busy = 'busy' in json.dumps(r).lower() or r.get('error', {}).get('type') == 'upstream_error'
        log(kind=kind, status='busy' if busy else 'error', detail=json.dumps(r)[:200]); return ('busy' if busy else 'error'), None, None
    u = r.get('usage'); log(kind=kind, status='ok', usage=u)
    return 'ok', r['choices'][0]['message']['content'], u

def post_retry(content, kind):
    delay = 20
    while True:
        ok, busy = counts()
        if ok >= MAX_OK or busy >= MAX_BUSY: raise SystemExit(f'budget exhausted ok={ok} busy={busy}')
        st, text, u = post(content, kind)
        if st == 'ok': return text, u
        if st == 'error': time.sleep(10); st2 = None
        time.sleep(delay); delay = min(delay * 2, 300)

def panel(case, key):
    a, b = (Image.open(case['images'][key][v]).convert('RGB').resize((512, 512)) for v in ('perspective', 'top'))
    im = Image.new('RGB', (1024, 512)); im.paste(a, (0, 0)); im.paste(b, (512, 0))
    buf = io.BytesIO(); im.save(buf, 'PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()

def load_cases(names):
    out = {}
    for n in names:
        for c in json.load(open(P / n / 'ledger.json'))['cases']:
            if c['status'] == 'rendered': out[c['uid']] = c
    return out

def judge_pair(case, x, y, salt, flip, kind):
    swap = (int(hashlib.sha256((salt + case['uid']).encode()).hexdigest(), 16) % 2 == 1) ^ flip
    first, second = (y, x) if swap else (x, y)
    content = [{'type': 'text', 'text': PROMPT}] + [{'type': 'image_url', 'image_url': {'url': panel(case, k)}} for k in (first, second)]
    text, u = post_retry(content, kind)
    try: v = json.loads(text[text.find('{'):text.rfind('}') + 1])
    except Exception: v = {'winner': None, 'raw_text': text[:200]}
    w = {'A': first, 'B': second, 'tie': 'tie'}.get(v.get('winner'), 'invalid')
    return {'uid': case['uid'], 'x': x, 'y': y, 'first': first, 'flip': flip, 'winner': w, 'raw': v, 'usage': u}

def run(name, ledgers, x, y, uids=None, flip=False, salt=None):
    """Resumable; rows -> rows-<name>.jsonl"""
    cases = load_cases(ledgers); out = J / f'rows-{name}.jsonl'
    salt = salt or ('judge-20261021:' if (x, y) == ('S1', 'S2') else f'judge-20261021-{x}-{y}:')
    done = {json.loads(l)['uid'] for l in out.read_text().splitlines()} if out.exists() else set()
    for uid in (uids or list(cases)):
        if uid in done or uid not in cases: continue
        r = judge_pair(cases[uid], x, y, salt, flip, name)
        with open(out, 'a') as f: f.write(json.dumps(r, ensure_ascii=False) + '\n')
