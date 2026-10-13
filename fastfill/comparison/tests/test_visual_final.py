"""Final visual config from the committed template and a saved sacct listing (no ssh, no launch)."""
import json
from pathlib import Path

import pytest

from fastfill.comparison import overnight
from fastfill.comparison import visual_final as V

TEMPLATE = (Path(V.__file__).parents[2] / 'docs/research/20261013-unknown-size-comparison'
            / 'visual-final-dpo2-template.json')


def _sacct(tmp_path, **states):
    jobs = {k: v['job_id'] for k, v in json.loads(TEMPLATE.read_text())['candidate_agents'].items()}
    lines = [f"{jobs[k]}|{states.get(k, 'COMPLETED')}|{'1:0' if states.get(k) == 'FAILED' else '0:0'}"
             for k in jobs if states.get(k) != 'absent']
    lines += [f'{job}.batch|COMPLETED|0:0' for job in jobs.values()]  # Step lines are ignored.
    path = tmp_path / 'sacct.txt'
    path.write_text('\n'.join(lines) + '\n')
    return path


def _build(tmp_path, sacct, remote='/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/snapshots/main-x'):
    return V.build(TEMPLATE, sacct, tmp_path / 'config.json', tmp_path / 'code', remote, tmp_path / 'visual-out')


def test_config_renders_completed_agents_and_judges_strongest_open_model(tmp_path):
    config = _build(tmp_path, _sacct(tmp_path, O6='FAILED'))
    selection = config['visual_final_selection']
    assert selection['included'] == ['L', 'O1a', 'O1b', 'O2', 'O3', 'O4', 'O5', 'O7']
    assert selection['excluded'] == {'O6': {'state': 'FAILED', 'exit_code': '1:0'}}
    assert selection['judged'] == ['J', 'P', 'L', 'O5'] and config['agents']['L'] == 'llm-eval-L-dpo2-v1'
    loaded = overnight.load_config(tmp_path / 'config.json')
    assert loaded['geometry_name'] == 'geometry-v4' and loaded['agent_job_ids']['O5'] == '7389190'
    assert loaded['api_timeout_seconds'] == 86400 and loaded['final_tables'] is True


@pytest.mark.parametrize('states, remote, match', [
    ({'L': 'FAILED'}, None, 'API agent L'),
    ({'O3': 'RUNNING'}, None, 'has not ended'),
    ({'O7': 'absent'}, None, 'missing from the sacct output'),
    ({}, '<FILL: server snapshot>', 'placeholder')])
def test_refusals(tmp_path, states, remote, match):
    sacct = _sacct(tmp_path, **states)
    with pytest.raises(ValueError, match=match):
        _build(tmp_path, sacct, *([remote] if remote else []))
    assert not (tmp_path / 'config.json').exists()
