import json

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
