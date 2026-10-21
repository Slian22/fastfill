"""fastfill-system: the amendment-022 delivered system (fastfill.system.pipeline; input format in its docstring).

python -m fastfill.system.cli run --requests ROOMS.jsonl [--prior size-priors.json] [--prior-sha256 HEX] \
    --placer hf|vllm --model TRAINING_BASE --adapter ADAPTER --escalate none|o2|l+o2 [--api-cap N] \
    [--api-max-rooms N] [--env-file ~/.fastfill_api.env] [--o2-model MODELS/Qwen3.8-27B] [--dry-run] [--workers N] \
    --out NEW_DIR

--placer hf: the exact policy on the first visible BF16 GPU; vllm: greedy BF16 LoRA (amendment 021 §3 engine rule);
either runs in a child process that frees its GPU before O2 loads. --escalate l+o2 needs --api-cap (every HTTP request
counts, retries included) and the 0600 env file; --api-max-rooms N stops before any API call when more than N rooms
need it. o2 and l+o2 need --o2-model and 2 visible GPUs (TP 2), checked before the placer runs. --dry-run: placer + S1
only, result.json lists the rooms that would escalate.

--size-fallback api enables size-only API estimates for objects with no eligible category/room or category prior.
It also allows an omitted or missing unpinned --prior. --size-api-cap is a separate hard HTTP-call limit (retries
included), and --size-api-max-rooms limits affected rooms. Missing sizes in --dry-run produce a plan only, without
loading models or calling APIs. Malformed/pinned prior files still fail. See docs/fastfill-system-size-api.md.
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
    for flag in ('requests', 'model', 'adapter', 'out'):
        run.add_argument('--' + flag, type=Path, required=True)
    run.add_argument('--prior', type=Path, help='size prior; optional with --size-fallback api')
    run.add_argument('--prior-sha256')
    run.add_argument('--size-fallback', choices=pipeline.sizing.POLICIES, default='prior',
                     help='api estimates sizes only when no eligible prior bucket exists (or prior is absent)')
    run.add_argument('--size-api-cap', type=count, help='separate hard HTTP-call limit for sizing, including retries')
    run.add_argument('--size-api-max-rooms', type=count, help='refuse sizing before calls if more rooms need it')
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
    if args.prior is None and args.size_fallback != 'api':
        parser.error('--prior is required unless --size-fallback api is enabled')
    requests = pipeline.load_requests(args.requests)
    prior = (pipeline.load_prior(args.prior, args.prior_sha256, allow_missing=True) if args.size_fallback == 'api'
             else pipeline.load_prior(args.prior, args.prior_sha256))
    size_plan = pipeline.sizing.plan(requests, prior[0]) if args.size_fallback == 'api' else None
    if size_plan is not None:
        pipeline.sizing.preflight(size_plan, args.size_api_cap, args.size_api_max_rooms, dry_run=args.dry_run)
    needs_sizes = size_plan is not None and bool(size_plan['requests'])
    placer = None if args.dry_run and needs_sizes else {
        'hf': pipeline.hf_placer, 'vllm': pipeline.vllm_placer}[args.placer](args.model, args.adapter)
    result = pipeline.run(requests, prior,
                          placer, args.out, escalate=args.escalate, api_cap=args.api_cap,
                          api_max_rooms=args.api_max_rooms, env_file=args.env_file,
                          o2_model=args.o2_model, dry_run=args.dry_run, workers=max(1, args.workers),
                          trigger=args.escalate_on, size_fallback=args.size_fallback,
                          size_api_cap=args.size_api_cap, size_api_max_rooms=args.size_api_max_rooms)
    print(json.dumps({k: result[k] for k in ('status', 'rooms', 'stages', 'would_escalate')}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
