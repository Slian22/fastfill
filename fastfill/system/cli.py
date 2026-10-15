"""fastfill-system: the amendment-022 delivered system (fastfill.system.pipeline; input format in its docstring).

python -m fastfill.system.cli run --requests ROOMS.jsonl --prior size-priors.json [--prior-sha256 HEX] \
    --placer hf|vllm --model TRAINING_BASE --adapter ADAPTER --escalate none|o2|l+o2 [--api-cap N] \
    [--api-max-rooms N] [--env-file ~/.fastfill_api.env] [--o2-model MODELS/Qwen3.8-27B] [--dry-run] [--workers N] \
    --out NEW_DIR

--placer hf: the exact policy on the first visible BF16 GPU; vllm: greedy BF16 LoRA (amendment 021 §3 engine rule);
either runs in a child process that frees its GPU before O2 loads. --escalate l+o2 needs --api-cap (every HTTP request
counts, retries included) and the 0600 env file; --api-max-rooms N stops before any API call when more than N rooms
need it. o2 and l+o2 need --o2-model and 2 visible GPUs (TP 2), checked before the placer runs. --dry-run: placer + S1
only, result.json lists the rooms that would escalate.
"""
import argparse
import json
import os
from pathlib import Path
import sys

from fastfill.system import pipeline


def main(argv=None):
    def count(text):
        value = int(text)
        if value < 0:
            raise argparse.ArgumentTypeError('must be >= 0')
        return value
    parser = argparse.ArgumentParser(prog='fastfill-system', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    run = parser.add_subparsers(dest='command', required=True).add_parser('run')
    for flag in ('requests', 'prior', 'model', 'adapter', 'out'):
        run.add_argument('--' + flag, type=Path, required=True)
    run.add_argument('--prior-sha256')
    run.add_argument('--placer', choices=('hf', 'vllm'), required=True)
    run.add_argument('--escalate', choices=pipeline.ESCALATE, required=True)
    run.add_argument('--escalate-on', choices=pipeline.TRIGGERS, default='checker',
                     help="escalate on checker failure (default) or on checker or strict-admission failure")
    run.add_argument('--api-cap', type=count)
    run.add_argument('--api-max-rooms', type=count)
    run.add_argument('--env-file', type=Path, default=Path(os.environ.get('FASTFILL_API_ENV', '~/.fastfill_api.env')))
    run.add_argument('--o2-model', type=Path)
    run.add_argument('--dry-run', action='store_true')
    run.add_argument('--workers', type=count, default=1)
    args = parser.parse_args(argv)
    placer = {'hf': pipeline.hf_placer, 'vllm': pipeline.vllm_placer}[args.placer](args.model, args.adapter)
    result = pipeline.run(pipeline.load_requests(args.requests), pipeline.load_prior(args.prior, args.prior_sha256),
                          placer, args.out, escalate=args.escalate, api_cap=args.api_cap,
                          api_max_rooms=args.api_max_rooms, env_file=args.env_file,
                          o2_model=args.o2_model, dry_run=args.dry_run, workers=max(1, args.workers),
                          trigger=args.escalate_on)
    print(json.dumps({k: result[k] for k in ('status', 'rooms', 'stages', 'would_escalate')}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
