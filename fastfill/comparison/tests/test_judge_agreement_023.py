import json

import pytest

from fastfill.comparison.judge_agreement_023 import compare


def records(path, shift):
    rows = []
    for room in range(4):
        for k, arm in enumerate(('J', 'L', 'O6', 'P')):
            s = 4 + k + shift * (arm == 'L') + room % 2
            rows.append({'id': f'{room}-{arm}', 'uid': f'r{room}', 'arm': arm, 'status': 'completed',
                         'scores': {m + '_score': s for m in ('functionality', 'layout', 'aesthetics', 'overall')}})
    rows[0] = {**rows[0], 'status': 'failed', 'scores': None}
    path.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    return path


def test_identical_judges_agree_and_l_margin_tracks_shift(tmp_path):
    a = records(tmp_path / 'a.jsonl', 0)
    same = compare(a, a, ['x', 'y'])
    assert same['correlation']['overall']['spearman'] > 0.999 and same['same_ranking']
    assert same['room_kendall_tau'] == {'rooms': 3, 'mean': 1.0}  # room 0 lost J under both judges
    shifted = compare(a, records(tmp_path / 'b.jsonl', 3), ['x', 'y'])
    assert shifted['y']['L_margin_common'] - shifted['x']['L_margin_common'] == 3
    assert shifted['common_completed'] == 15


@pytest.mark.parametrize('mode', ['no_overall', 'none_completed', 'one_completed', 'constant', 'only_l'])
def test_missing_or_insufficient_scores_have_defined_missingness(tmp_path, mode):
    path = records(tmp_path / 'records.jsonl', 0)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for index, row in enumerate(rows):
        if mode == 'none_completed' or (mode == 'one_completed' and index != 1):
            row['status'], row['scores'] = 'failed', None
        elif row['status'] == 'completed':
            if mode == 'no_overall':
                row['scores'].pop('overall_score')
            if mode == 'constant':
                row['scores'] = {key: 5 for key in row['scores']}
    if mode == 'only_l':
        rows = [row for row in rows if row['arm'] == 'L']
    path.write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
    result = compare(path, path, ['a', 'b'])
    json.dumps(result, allow_nan=False)
    if mode in {'no_overall', 'none_completed'}:
        assert result['correlation']['overall'] == {'n': 0, 'spearman': None, 'pearson': None}
        assert result['a']['ranking_common'] is None
        assert result['same_ranking'] is None
    if mode in {'one_completed', 'constant'}:
        assert result['correlation']['overall']['pearson'] is None
        assert result['correlation']['overall']['spearman'] is None
    if mode in {'no_overall', 'none_completed', 'one_completed', 'only_l'}:
        assert result['a']['L_margin_common'] is None
    assert result['a']['means_common']['L']['metric_n']['overall'] == sum(
        row['arm'] == 'L' and row['status'] == 'completed' and 'overall_score' in row['scores'] for row in rows)
