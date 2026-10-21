"""Judge v2 (protocol addendum A): pros/cons before verdict, both A/B orders; verdict = agreed winner else tie.
python3 judge_v2.py NAME X Y LEDGER [LEDGER ...]   (rows -> judge/rows-v2-NAME.jsonl, resumable)"""
import json, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, '/Volumes/harddisk/fastfillout/unknownsize-20261013/probes-20261021/scripts')
import judge_lib as J

PROMPT = ("You see two renderings of the SAME room with the SAME furniture, arranged two ways: Layout A (first image) and "
          "Layout B (second image). Each image shows a perspective view (left) and a top-down view (right). "
          "First list the main pros and cons of Layout A, then of Layout B, judging realism and function for people "
          "living there: furniture grouped sensibly (nightstands beside the head of the bed, TV or TV stand facing the bed "
          "or sofa, chairs at their desk/table, bathroom fixtures placed as in a real bathroom), clear walking paths and "
          "doors, and no furniture passing through other furniture. Then decide. End your answer with one line of JSON "
          "only: {\"winner\": \"A\" | \"B\" | \"tie\"}.")


def ask(case, first, second, kind):
    content = [{'type': 'text', 'text': PROMPT}] + [{'type': 'image_url', 'image_url': {'url': J.panel(case, k)}}
                                                    for k in (first, second)]
    text, u = J.post_retry(content, kind)
    try:
        v = json.loads(text[text.rfind('{'):text.rfind('}') + 1])
    except Exception:
        v = {'winner': None}
    return {'A': first, 'B': second, 'tie': 'tie'}.get(v.get('winner'), 'invalid'), text[-600:], u


def one(args):
    case, x, y, name = args
    w1, t1, _ = ask(case, x, y, f'v2-{name}')
    w2, t2, _ = ask(case, y, x, f'v2-{name}')
    verdict = w1 if w1 == w2 else 'tie'
    return {'uid': case['uid'], 'x': x, 'y': y, 'order_xy': w1, 'order_yx': w2, 'consistent': w1 == w2,
            'verdict': verdict, 'text_xy': t1, 'text_yx': t2}


def main():
    name, x, y, ledgers = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    out = J.J / f'rows-v2-{name}.jsonl'
    done = {json.loads(l)['uid'] for l in out.read_text().splitlines()} if out.exists() else set()
    cases = [c for c in J.load_cases(ledgers).values() if c['uid'] not in done]
    with ThreadPoolExecutor(4) as ex:
        for r in ex.map(one, [(c, x, y, name) for c in cases]):
            with open(out, 'a') as f:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    cons = [r for r in rows if r['consistent']]
    xw = sum(r['verdict'] == x for r in rows); yw = sum(r['verdict'] == y for r in rows)
    print(f"{name}: rooms {len(rows)}, order-consistent {len(cons)}/{len(rows)}, {x} wins {xw}, {y} wins {yw}, "
          f"ties/inconsistent {len(rows) - xw - yw}; {x} share of decisive {xw / max(1, xw + yw):.2f}")


if __name__ == '__main__':
    main()
