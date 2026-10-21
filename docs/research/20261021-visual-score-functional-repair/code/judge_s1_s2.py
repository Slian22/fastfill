"""Blind paired judge: gpt-6.1-sol (medium) sees layout A and B (perspective + top, same room/furniture), A/B order
randomised per room by hash; answers which is the more realistic, functional arrangement. One call per room."""
import base64, hashlib, io, json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image

O = Path(sys.argv[1]); LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 0
C = json.load(open('/Users/slian/Desktop/3D/Worldedge/OptiScene/llm_config.json'))
PROMPT = ("You see two renderings of the SAME room with the SAME furniture, arranged two ways: Layout A (first image) and "
          "Layout B (second image). Each image shows a perspective view (left) and a top-down view (right). "
          "Judge which arrangement is more realistic and functional for people living there: furniture grouped sensibly "
          "(e.g. nightstands beside the head of the bed, TV or TV stand facing the bed or sofa, chairs at their desk/table, "
          "toilet and fixtures placed as in a real bathroom), clear walking paths and doors, and no furniture passing "
          "through other furniture. Reply with JSON only: {\"winner\": \"A\" | \"B\" | \"tie\", \"reason\": \"<= 30 words\"}.")


def panel(case, key):
    a, b = (Image.open(case['images'][key][v]).convert('RGB').resize((512, 512)) for v in ('perspective', 'top'))
    im = Image.new('RGB', (1024, 512)); im.paste(a, (0, 0)); im.paste(b, (512, 0))
    buf = io.BytesIO(); im.save(buf, 'PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def call(case):
    swap = int(hashlib.sha256(('judge-20261021:' + case['uid']).encode()).hexdigest(), 16) % 2 == 1
    first, second = ('S2', 'S1') if swap else ('S1', 'S2')
    body = {'model': C['model'], 'reasoning_effort': C['reasoning_effort'], 'messages': [{'role': 'user', 'content': [
        {'type': 'text', 'text': PROMPT},
        {'type': 'image_url', 'image_url': {'url': panel(case, first)}},
        {'type': 'image_url', 'image_url': {'url': panel(case, second)}}]}]}
    req = urllib.request.Request(C['chat_completions_url'], data=json.dumps(body).encode(), headers={
        'Content-Type': 'application/json', 'Authorization': 'Bearer ' + C['api_key'], 'User-Agent': 'curl/8.7.1'})
    t = time.time()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=600))
        text = r['choices'][0]['message']['content']
        v = json.loads(text[text.find('{'):text.rfind('}') + 1])
        w = {'A': first, 'B': second}.get(v.get('winner'), 'tie')
        return {'uid': case['uid'], 'first': first, 'raw': v, 'winner': w, 'usage': r.get('usage'), 's': round(time.time() - t, 1)}
    except Exception as e:
        return {'uid': case['uid'], 'first': first, 'error': repr(e)[:300], 's': round(time.time() - t, 1)}


cases = [c for i in (0, 1) for c in json.load(open(O / f'render-{i}' / 'ledger.json'))['cases'] if c['status'] == 'rendered']
cases = cases[:LIMIT] if LIMIT else cases
with ThreadPoolExecutor(4) as ex:
    rows = list(ex.map(call, cases))
out = O / 'judge' / ('rows-smoke.jsonl' if LIMIT else 'rows.jsonl')
out.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
for r in rows:
    print(r.get('winner'), r.get('raw', r.get('error')), r['s'])
