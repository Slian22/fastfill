"""Freeze train-bound, independently verified preference pairs before DPO.

Only training references enter this audit. The legacy candidate size interval is
expanded just enough to contain its reference dimension, so thin rugs and large
objects do not have impossible acceptance intervals. No dev/test score selects
this policy. Existing pairs are filtered whole; no answers are generated or fixed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from fastfill.v1_1.build_dpo import _room_from_v1_1_messages
from fastfill.v1_1.scene_v1_1 import parse_v1_1
from fastfill.validate import check, holds

PROTOCOL = 'fastfill_v1.1_pair_sanitizer_v1'
POLICY = {
    'size_lower': 'min(gt, max(0.05, 0.45 * gt))',
    'size_upper': 'max(gt, min(6.0, 2.2 * gt))',
    'chosen': 'parse + geometry + all prompt constraints + reference-relative size',
    'rejected': 'at least one independently verified semantic defect',
    'reference': 'exact training uid, source, and prompt; reference geometry and constraints must pass',
    'selection_data': 'train only; no evaluation scores or generated replacement answers',
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def size_violations(layout: dict, reference: dict, *, legacy: bool = False) -> list[str]:
    """Return IDs outside the registered interval (including legacy diagnostics)."""
    bad = []
    for oid, obj in layout.items():
        for value, gt in zip(obj['size'], reference[oid]['size']):
            low, high = max(.05, .45 * gt), min(6., 2.2 * gt)
            if not legacy:
                low, high = min(gt, low), max(gt, high)
            if not low <= value <= high:
                bad.append(oid)
                break
    return bad


def _violations(prompt: list, text: str, reference: dict) -> tuple[list[str], dict]:
    parsed = _room_from_v1_1_messages(prompt, text)
    if parsed is None:
        return ['parse_failure'], {}
    room, _ = parsed
    ids = [obj['id'] for obj in room['objects']]
    layout, error = parse_v1_1(text, ids)
    if error:
        return ['parse_failure'], {}
    checked = check(room)
    defects = [name for name in ('oob', 'support_fail', 'ceiling', 'fixed_blocking') if checked[name]]
    if checked['collisions']['main']:
        defects.append('main_collision')
    if not all(holds(con, room['objects'], room['boundary']) for con in room['constraints']):
        defects.append('constraint_failure')
    if size_violations(layout, reference):
        defects.append('size_distortion')
    return sorted(defects), layout


def audit_pair(pair: dict, reference_row: dict) -> dict[str, Any]:
    """Verify identity strictly; mark semantic label failures for whole-pair removal."""
    for key in ('uid', 'source'):
        if pair.get(key) != reference_row.get(key) or not isinstance(pair.get(key), str):
            raise ValueError(f'Pair {key} identity differs from training reference')
    messages = reference_row['messages']
    if len(messages) != 3 or pair.get('prompt') != messages[:2]:
        raise ValueError(f'Pair prompt identity differs for {pair["uid"]}')
    if not all(isinstance(pair.get(key), str) for key in ('chosen', 'rejected')):
        raise ValueError(f'Pair answers must be strings for {pair["uid"]}')
    ids = [o['id'] for o in json.loads(messages[1]['content'])['objects']]
    reference, error = parse_v1_1(messages[2]['content'], ids)
    if error:
        raise ValueError(f'Training reference does not parse for {pair["uid"]}: {error}')
    reference_errors, _ = _violations(messages[:2], messages[2]['content'], reference)
    chosen_errors, chosen = _violations(pair['prompt'], pair['chosen'], reference)
    rejected_errors, rejected = _violations(pair['prompt'], pair['rejected'], reference)
    reasons = [f'reference:{x}' for x in reference_errors] + [f'chosen:{x}' for x in chosen_errors]
    if not rejected_errors:
        reasons.append('rejected:no_verified_defect')
    if chosen and rejected and chosen == rejected:
        reasons.append('identical_semantic_answers')
    return {
        'keep': not reasons, 'drop_reasons': sorted(reasons),
        'chosen_violations': chosen_errors, 'rejected_violations': rejected_errors,
        'chosen_is_reference': chosen == reference,
        'chosen_legacy_size_fail': bool(chosen and size_violations(chosen, reference, legacy=True)),
        'chosen_corrected_size_fail': 'size_distortion' in chosen_errors,
        'rejected_legacy_size_fail': bool(rejected and size_violations(rejected, reference, legacy=True)),
        'rejected_corrected_size_fail': 'size_distortion' in rejected_errors,
    }


def _read_pairs(path: Path) -> list[tuple[str, dict]]:
    rows, seen = [], set()
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            pair = json.loads(line)
            uid = pair.get('uid')
            if not isinstance(uid, str) or not uid:
                raise ValueError(f'Missing string uid in {path}')
            if uid in seen:
                raise ValueError(f'duplicate uid {uid} in {path}')
            seen.add(uid)
            rows.append((line if line.endswith('\n') else line + '\n', pair))
    return rows


def _references(path: Path, wanted: set[str]) -> dict[str, dict]:
    refs = {}
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('uid') in wanted:
                if row['uid'] in refs:
                    raise ValueError(f'duplicate training uid {row["uid"]}')
                refs[row['uid']] = row
    missing = wanted - refs.keys()
    if missing:
        raise ValueError(f'Pairs lack training membership: {sorted(missing)[:5]}')
    return refs


def _identity(paths: dict[str, Path]) -> dict:
    package = Path(__file__).resolve().parents[1]
    modules = ('v1_1/audit_pairs.py', 'v1_1/build_dpo.py', 'v1_1/scene_v1_1.py',
               'validate.py', 'scene.py', 'anchors.py')
    return {
        'protocol': PROTOCOL, 'policy': POLICY,
        'inputs': {key: {'path': str(path.resolve()), 'sha256': sha256(path)} for key, path in paths.items()},
        'code_sha256': {name: sha256(package / name) for name in modules},
    }


def _existing(out: Path, identity: dict) -> dict | None:
    if not out.exists():
        return None
    receipt_file = out / 'audit.json'
    if not receipt_file.is_file():
        raise ValueError(f'Existing output has no identity receipt: {out}')
    receipt = json.loads(receipt_file.read_text())
    if any(receipt.get(key) != value for key, value in identity.items()):
        raise ValueError(f'Existing output identity differs: {out}')
    for stage in ('stage1', 'stage2'):
        record = receipt['outputs'][stage]
        file = out / f'{stage}.cleaned.jsonl'
        if record['file'] != file.name or not file.is_file() or sha256(file) != record['sha256']:
            raise ValueError(f'Existing {stage} output hash differs: {out}')
    return receipt


def _audit_task(task: tuple[dict, dict]) -> dict:
    return audit_pair(*task)


def _write_stage(stage: str, rows: list, refs: dict, temporary: Path, executor: Any) -> dict:
    target = temporary / f'{stage}.cleaned.jsonl'
    tasks = ((pair, refs[pair['uid']]) for _, pair in rows)
    results = map(_audit_task, tasks) if executor is None else executor.map(_audit_task, tasks, chunksize=32)
    reasons, diagnostics, dropped, kept = Counter(), Counter(), [], 0
    with target.open('w', encoding='utf-8') as stream:
        for index, ((line, pair), result) in enumerate(zip(rows, results), 1):
            for key in ('chosen_is_reference', 'chosen_legacy_size_fail', 'chosen_corrected_size_fail',
                        'rejected_legacy_size_fail', 'rejected_corrected_size_fail'):
                diagnostics[key] += int(result[key])
            if result['keep']:
                stream.write(line)
                kept += 1
            else:
                reasons.update(result['drop_reasons'])
                dropped.append({'line': index, 'uid': pair['uid'], 'reasons': result['drop_reasons']})
    if not kept:
        raise ValueError(f'No verified preference pairs remain in {stage}')
    return {'file': target.name, 'sha256': sha256(target), 'input_rows': len(rows),
            'kept': kept, 'dropped': len(dropped), 'drop_reasons': dict(reasons),
            'diagnostics': dict(diagnostics), 'dropped_rows': dropped}


def sanitize_pairs(train: Path, stage1: Path, stage2: Path, out: Path, *, workers: int = 4) -> dict:
    if workers < 1 or workers > 16:
        raise ValueError('workers must be between 1 and 16')
    paths = {'train': Path(train), 'stage1': Path(stage1), 'stage2': Path(stage2)}
    out = Path(out)
    identity = _identity(paths)
    prior = _existing(out, identity)
    if prior is not None:
        return prior
    data = {stage: _read_pairs(paths[stage]) for stage in ('stage1', 'stage2')}
    refs = _references(paths['train'], {pair['uid'] for rows in data.values() for _, pair in rows})
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{out.name}.tmp.', dir=out.parent))
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        outputs = {stage: _write_stage(stage, rows, refs, temporary, executor) for stage, rows in data.items()}
        if identity != _identity(paths):
            raise ValueError('Input or audit code identity changed during audit')
        receipt = {**identity, 'outputs': outputs}
        (temporary / 'audit.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
        if out.exists():
            raise ValueError(f'Output appeared during audit: {out}')
        os.rename(temporary, out)
        return receipt
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-jsonl', required=True, type=Path)
    parser.add_argument('--stage1', required=True, type=Path)
    parser.add_argument('--stage2', required=True, type=Path)
    parser.add_argument('--out-dir', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    receipt = sanitize_pairs(args.train_jsonl, args.stage1, args.stage2, args.out_dir, workers=args.workers)
    print(json.dumps({stage: {key: value for key, value in info.items() if key != 'dropped_rows'}
                      for stage, info in receipt['outputs'].items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()
