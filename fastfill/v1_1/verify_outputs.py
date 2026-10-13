"""Independently CPU re-score completed outputs using their frozen scoring code.

This module never generates model text or changes original predictions/metrics.
The public CLI starts a separate interpreter pinned to --scoring-root; receipts
are separate from the original evaluation completion markers. All four cells of
a tag must already be complete. A verified receipt asserts reproducibility of
scoring from persisted output text, not model quality or generation authenticity.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
from itertools import zip_longest
import json
import multiprocessing
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

TAGS = ('sft', 'dpo1', 'final')
SPLITS = ('dev_rooms', 'dev_constrained_rooms', 'test_rooms', 'test_constrained_rooms')
MODELS = dict(zip(TAGS, ('sft_merged', 'dpo1_merged', 'final_merged')))


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'Expected a JSON object: {path}')
    return value


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _runtime() -> dict[str, Any]:
    versions = {}
    for package in ('shapely', 'numpy', 'transformers', 'torch', 'vllm'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {'python': platform.python_version(), 'sys_version': sys.version,
            'executable': str(Path(sys.executable).resolve()), 'packages': versions}


def _scoring_imports(scoring_root: Path):
    from fastfill.v1_1 import evaluate_v1_1 as evaluator, evaluation_protocol as protocol

    if Path(evaluator.__file__).resolve().parents[2] != scoring_root:
        raise ValueError('Imported scorer differs from the explicitly requested frozen scoring root')
    if Path(protocol.__file__).resolve().parents[2] != scoring_root:
        raise ValueError('Imported protocol differs from the frozen scoring root')
    return evaluator, protocol


def _preflight(output: Path, tags: tuple[str, ...], scoring_root: Path, evaluator, protocol) -> dict[str, Any]:
    runner, frozen = _read(output / 'runner.json'), _read(output / 'protocol.json')
    if Path(runner['root']).resolve() != scoring_root or Path(runner['output']).resolve() != output:
        raise ValueError('Frozen runner output/scoring root mismatch')
    if Path(runner['python']).resolve() != Path(sys.executable).resolve():
        raise ValueError('Verification must use the frozen runner Python interpreter')
    if runner.get('models') != MODELS or runner.get('splits') != list(SPLITS):
        raise ValueError('Frozen runner model/split contract mismatch')
    if evaluator.evaluation_code_hashes() != frozen['code_hashes']:
        raise ValueError('Frozen scoring code identity mismatch')
    actual = {split: protocol.dataset_identity(frozen['datasets'][split]['path']) for split in SPLITS}
    if actual != frozen['datasets']:
        raise ValueError('Frozen dataset identity mismatch')
    cells, states = {}, {}
    for tag in tags:
        state = _read(output / 'state' / f'{tag}.json')
        model = protocol.model_identity(Path(runner['run']) / MODELS[tag])
        if state.get('model_identity') != model:
            raise ValueError(f'Frozen model identity mismatch: {tag}')
        if state.get('status') != 'complete':
            raise ValueError(f'Model evaluation state is not complete: {tag}')
        states[tag] = protocol.file_identity(output / 'state' / f'{tag}.json')
        for split in SPLITS:
            identity = {'schema_version': frozen['schema_version'], 'dataset': actual[split],
                        'model': model, 'settings': frozen['settings'], 'code_hashes': frozen['code_hashes']}
            state_cell = state.get('cells', {}).get(split, {})
            if state_cell.get('status') != 'complete' or state_cell.get('identity') != identity:
                raise ValueError(f'Frozen cell state/identity mismatch: {tag}/{split}')
            target = output / tag / split
            if not protocol.evaluation_complete(target, identity):
                raise ValueError(f'Missing or invalid full-cohort completion: {tag}/{split}')
            cells[f'{tag}/{split}'] = {'identity': identity, 'artifacts': {
                name: protocol.file_identity(target / name)
                for name in ('completion.json', 'predictions.jsonl', 'metrics.json')}}
    return {'cells': cells, 'states': states,
            'protocol': protocol.file_identity(output / 'protocol.json'),
            'runner': protocol.file_identity(output / 'runner.json')}


def _tasks(rooms_path: Path, predictions_path: Path):
    with rooms_path.open(encoding='utf-8') as rooms, predictions_path.open(encoding='utf-8') as predictions:
        for index, pair in enumerate(zip_longest(rooms, predictions)):
            room_line, prediction_line = pair
            if room_line is None or prediction_line is None:
                raise ValueError(f'Prediction/dataset row count mismatch: {predictions_path}')
            room, prediction = json.loads(room_line), json.loads(prediction_line)
            if not isinstance(room, dict) or not isinstance(prediction, dict):
                raise ValueError(f'Invalid row at evaluation_row={index}')
            if prediction.get('uid') != room.get('uid') or type(prediction.get('evaluation_row')) is not int \
                    or prediction['evaluation_row'] != index:
                raise ValueError(f'Prediction ordered row identity mismatch at evaluation_row={index}')
            if 'output' not in prediction or (prediction['output'] is not None and not isinstance(prediction['output'], str)):
                raise ValueError(f'Prediction output type mismatch at evaluation_row={index}')
            yield room, prediction, index


def _rescore(task):
    from fastfill.v1_1.evaluate_v1_1 import score_v1_1

    room, prediction, index = task
    text = prediction['output']
    scored = score_v1_1(room, text, 'missing_generation' if text is None else None)
    saved = {key: value for key, value in prediction.items() if key not in ('output', 'evaluation_row')}
    if _canonical(scored) != _canonical(saved):
        changed = sorted(set(scored) ^ set(saved) | {
            key for key in scored.keys() & saved.keys() if _canonical(scored[key]) != _canonical(saved[key])})
        raise ValueError(f'Prediction scoring mismatch at evaluation_row={index}, fields={changed}')
    return scored


def _verify_cell(output: Path, name: str, bound: dict[str, Any], evaluator, executor) -> dict[str, Any]:
    target = output / name
    tasks = _tasks(Path(bound['identity']['dataset']['path']), target / 'predictions.jsonl')
    rows = list(map(_rescore, tasks) if executor is None else executor.map(_rescore, tasks, chunksize=16))
    if len(rows) != bound['identity']['dataset']['line_count']:
        raise ValueError(f'Recomputed cohort count mismatch: {name}')
    summary = evaluator.summarize_v1_1(rows)
    if _canonical(summary) != _canonical(_read(target / 'metrics.json')):
        raise ValueError(f'Recomputed metrics mismatch: {name}')
    print(f'[verify_outputs] independently re-scored {name}: {len(rows)} rows', flush=True)
    return {**bound, 'verified_rows': len(rows), 'scoring_fields_equal': True, 'metrics_equal': True}


def _tag_bindings(bindings: dict[str, Any], tag: str) -> dict[str, Any]:
    return {**bindings, 'states': {tag: bindings['states'][tag]},
            'cells': {name: cell for name, cell in bindings['cells'].items() if name.startswith(tag + '/')}}


def _cached_cells(receipt, tag, bindings, verifier, runtime):
    if not receipt or receipt.get('status') != 'complete' or receipt.get('tag') != tag:
        return None
    if receipt.get('bindings') != _tag_bindings(bindings, tag) or receipt.get('runtime') != runtime:
        return None
    if any(receipt.get('verifier', {}).get(key) != verifier[key] for key in ('sha256', 'size_bytes')):
        return None
    cells = receipt.get('cells')
    if not isinstance(cells, dict) or set(cells) != set(SPLITS):
        return None
    for split, cell in cells.items():
        expected = {**bindings['cells'][f'{tag}/{split}'],
                    'verified_rows': bindings['cells'][f'{tag}/{split}']['identity']['dataset']['line_count'],
                    'scoring_fields_equal': True, 'metrics_equal': True}
        if _canonical(cell) != _canonical(expected):
            return None
    return {f'{tag}/{split}': cell for split, cell in cells.items()}


def _verify_locked(output: Path, tags: tuple[str, ...], scoring_root: Path, workers: int,
                   evaluator, protocol, reuse_verified: bool):
    verifier = {'path': str(Path(__file__).resolve()), **protocol.file_identity(__file__)}
    started = _now()
    base = {'schema_version': 1, 'started_at': started, 'verifier': verifier,
            'scoring_root': str(scoring_root), 'runtime': _runtime(),
            'interpretation': 'Independent CPU scoring reproduction; not model quality or generation authenticity.'}
    previous = {}
    if reuse_verified:
        for tag in tags:
            try:
                previous[tag] = _read(output / 'verification' / f'{tag}.json')
            except (OSError, ValueError):
                previous[tag] = None
    for tag in tags:
        protocol.write_json_atomic(output / 'verification' / f'{tag}.json', {**base, 'tag': tag, 'status': 'running'})
    try:
        bindings = _preflight(output, tags, scoring_root, evaluator, protocol)
        cached = {}
        for tag in tags:
            cached.update(_cached_cells(previous.get(tag), tag, bindings, verifier, base['runtime']) or {})
        with ExitStack() as stack:
            pool = stack.enter_context(ProcessPoolExecutor(max_workers=workers,
                mp_context=multiprocessing.get_context('spawn'))) if workers > 1 else None
            cells = {name: cached[name] if name in cached else _verify_cell(output, name, bound, evaluator, pool)
                     for name, bound in bindings['cells'].items()}
        if _preflight(output, tags, scoring_root, evaluator, protocol) != bindings:
            raise ValueError('Evaluation inputs/artifacts identity changed during independent verification')
        if protocol.file_identity(__file__) != {key: verifier[key] for key in ('sha256', 'size_bytes')}:
            raise ValueError('Verifier source identity changed during independent verification')
        for tag in tags:
            receipt = {**base, 'tag': tag, 'status': 'complete', 'completed_at': _now(),
                       'bindings': _tag_bindings(bindings, tag),
                       'cells': {name.split('/')[1]: cell for name, cell in cells.items() if name.startswith(tag + '/')}}
            protocol.write_json_atomic(output / 'verification' / f'{tag}.json', receipt)
        return {'status': 'complete', 'verified_cells': len(cells), 'cached_cells': len(cached), 'tags': list(tags)}
    except BaseException as exc:
        for tag in tags:
            protocol.write_json_atomic(output / 'verification' / f'{tag}.json',
                {**base, 'tag': tag, 'status': 'failed', 'failed_at': _now(), 'error': f'{type(exc).__name__}: {exc}'})
        raise


def verify_worker(output: str | Path, tags, scoring_root: str | Path, workers: int = 1, *, reuse_verified=False):
    output, scoring_root, tags = Path(output).resolve(), Path(scoring_root).resolve(), tuple(tags)
    if not tags or len(set(tags)) != len(tags) or any(tag not in TAGS for tag in tags) or workers < 1:
        raise ValueError('Require distinct valid model tags and positive workers')
    evaluator, protocol = _scoring_imports(scoring_root)
    directory = output / 'verification'
    directory.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for tag in sorted(tags):
            stream = stack.enter_context((directory / f'.{tag}.lock').open('a'))
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _verify_locked(output, tags, scoring_root, workers, evaluator, protocol, reuse_verified)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--tag', choices=TAGS)
    choice.add_argument('--all', action='store_true')
    parser.add_argument('--scoring-root', required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--reuse-verified', action='store_true', help='Reuse only an identity- and runtime-matched independent receipt')
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._worker:
        result = verify_worker(args.output, TAGS if args.all else (args.tag,), args.scoring_root,
                               args.workers, reuse_verified=args.reuse_verified)
        print(json.dumps(result), flush=True)
        return
    root = Path(args.scoring_root).resolve()
    # -I removes ambient PYTHONPATH; only the explicit frozen root is prepended.
    bootstrap = ('import runpy,sys;sys.path.insert(0,sys.argv[1]);'
                 'sys.argv=[sys.argv[2],*sys.argv[3:]];runpy.run_path(sys.argv[0],run_name="__main__")')
    original_python = str(Path(_read(Path(args.output) / 'runner.json')['python']).resolve())
    command = [original_python, '-I', '-c', bootstrap, str(root), str(Path(__file__).resolve()),
               '--output', str(Path(args.output).resolve()), '--scoring-root', str(root),
               '--workers', str(args.workers), '--_worker']
    command += ['--all'] if args.all else ['--tag', args.tag]
    if args.reuse_verified:
        command += ['--reuse-verified']
    raise SystemExit(subprocess.run(command, cwd=root, check=False).returncode)


if __name__ == '__main__':
    main()
