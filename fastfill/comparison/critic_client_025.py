#!/usr/bin/env python3
"""Amendment 025 critic call: server_api_judge's client with the critic's prompt, parser and effort.

Run only on the credential-owning server, normally one case at a time by critic_batch_025:
  python -m fastfill.comparison.critic_client_025 --manifest one-case.json --results-dir R --env-file ENV
Model gpt-6.1-sol and reasoning effort medium (as system L) are fixed; the manifest's expected_model must
be gpt-6.1-sol. Manifest schema, artifact/PNG checks, one POST without retries, SSE reading, model check,
redaction and the immutable attempt/result files are server_api_judge's (see its docstring). The result
keeps raw_final_text, returned_model, usage (incl. reasoning_tokens) and the parsed critic JSON under
"scores", or a fixed error code.

The answer must pass critic_025.parse_edits' whole-answer rule, with the same error codes (invalid_json,
invalid_keys, invalid_scores, invalid_edits): exactly {"scores": {"functionality": int 0-10, "layout": int 0-10},
"edits": [...]}. Each edit (op, fields, id, MAX_EDITS) is checked there against the room's ids and rejected alone,
so the client leaves edits opaque. One documented exception: a number that parses to a non-finite float (e.g.
1e400) cannot be stored, so the whole answer fails here as invalid_number (raw text kept, room unedited), where
parse_edits would reject only that edit.
"""
import argparse
import json
from pathlib import Path
import sys

from fastfill.comparison import server_api_judge as judge
from fastfill.comparison.server_api_judge import JudgeError

MODEL, EFFORT = 'gpt-6.1-sol', 'medium'
INSTRUCTIONS = (
    "You are an expert interior-layout critic. You review one furnished room and propose constrained edits; you "
    "never edit the room yourself. The text results in the prompt (objects, frozen checker, strict admission) take "
    "precedence over what you see in the images, and the boxes are the exact geometry, so overlapping boxes really "
    "collide. While any critical problem exists (a collision, an object outside the room, a blocked door or window, "
    "an object facing a wall or away from its partner), propose edits for critical problems only, the most "
    "important first. Use only the prompt's edit operations, fields and object ids. Return only one JSON object, "
    'with no markdown and no other keys: {"scores": {"functionality": INT, "layout": INT}, "edits": [...]}. '
    "Do not invent evidence that is absent from the supplied images and text."
)


def parse_critic(text):
    critic = judge.strict_json(text, 'invalid_json')
    if not isinstance(critic, dict) or set(critic) != {'scores', 'edits'}:
        raise JudgeError('invalid_keys')
    scores = critic['scores']
    if (not isinstance(scores, dict) or set(scores) != {'functionality', 'layout'}
            or any(type(v) is not int or not 0 <= v <= 10 for v in scores.values())):
        raise JudgeError('invalid_scores')
    if not isinstance(critic['edits'], list):
        raise JudgeError('invalid_edits')
    try:
        judge.canonical(critic)  # the result record must be storable (allow_nan=False)
    except ValueError:
        raise JudgeError('invalid_number') from None
    return critic


def evaluate(manifest, results_dir, env_file, limit=1, timeout=600):
    return judge.evaluate(manifest, results_dir, env_file, limit, timeout, MODEL, EFFORT, INSTRUCTIONS, parse_critic)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--results-dir', required=True, type=Path)
    parser.add_argument('--env-file', type=Path, default=Path('~/.fastfill_api.env'))
    parser.add_argument('--limit', type=int, default=1)
    parser.add_argument('--timeout', type=float, default=600)
    args = parser.parse_args(argv)
    try:
        records = evaluate(args.manifest, args.results_dir, args.env_file, args.limit, args.timeout)
    except JudgeError as exc:
        print('critic_error: ' + str(exc), file=sys.stderr)
        return 1
    except Exception:
        print('critic_error: unexpected_error', file=sys.stderr)
        return 1
    failed = sum(r['status'] != 'completed' for r in records)
    print(json.dumps({'examined': len(records), 'completed': len(records) - failed, 'failed': failed}))
    return int(bool(failed))


if __name__ == '__main__':
    sys.exit(main())
