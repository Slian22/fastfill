"""Room families: the frozen mapping on the full test list, and per-family tables from synthetic rows."""
import collections
import json
from pathlib import Path

import pytest

from fastfill.comparison import geometry as g
from fastfill.comparison import llm_agent as L
from fastfill.comparison import room_families as rf
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.tests.test_llm_agent import eval_data, target, write_rows

TEST_ROOMS = Path('/Volumes/harddisk/fastfill_data/v3/test_rooms.jsonl')  # = RUN/eval-data-v1 rooms (8,647)
MAPPING = {  # family: {raw room_type: test rooms}, the 346 strings of the test set (8,647 rooms)
    'bedroom': {'bedroom': 1625, 'master bedroom': 458, "children's bedroom": 131, 'elderly room': 57, 'kids room': 48,
                'bedroom with home office': 30, 'bedroom / hotel': 23, 'primary bedroom': 20, 'dorm room 1': 15,
                'dorm room 2': 15, 'standard room 1': 15, 'standard room 2': 15, 'standard room 3': 15,
                'standard room 4': 15, 'secondary bedroom': 14, 'dorm room 3': 12, 'guest room': 8, 'nursery': 8,
                'child room': 5, "children's room": 3, 'master_bedroom': 3, 'kids_bedroom': 2, 'Bedroom / Hotel': 1,
                'bedroom with study room': 1, 'children_room': 1, 'hotel_room': 1, 'kids_room': 1, 'on call room': 1,
                'primary_bedroom': 1},
    'living': {'living room': 845, 'living dining room': 169, 'living room with dining area': 146, 'living_room': 62,
               'living and dining room': 37, 'livingroom': 31, 'recreation room': 23, 'game room': 10,
               'living room / lounge': 10, 'living_dining_room': 9, 'leisure corner': 8, 'media room': 6,
               'playroom': 6, 'gaming_room': 5, 'family lounge': 4, 'game room 1': 4, 'lounge break area 4': 4,
               'lounge collaboration 2': 4, 'open_plan_living_kitchen': 4, 'staff_lounge_1': 4, 'family room': 3,
               'lounge': 3, 'lounge area': 3, 'break_room': 1, 'client_lounge': 1, 'common_room': 1,
               'living room with kitchen': 1, 'media_room': 1, 'music room': 1, 'music_room': 1, 'rec/game': 1,
               'recreation_room': 1, 'tv': 1},
    'dining': {'diningroom': 198, 'dining room': 106, 'dining_room': 34, 'dining booth': 2,
               'dining room with kitchen': 2, 'dining room with living room': 1},
    'kitchen': {'kitchen': 705, 'pantry 1': 24, 'pantry break area': 9, 'pantry and lounge': 8, 'pantry nook 2': 8,
                'pantry': 7, 'pantry 2': 7, 'pantry break area 1': 4, 'residential_kitchen': 3,
                'commercial kitchen': 1, 'commercial_kitchen': 1, 'kitchen with dining area': 1},
    'bathroom': {'bathroom': 932, 'restroom': 16, 'restroom 2': 15, 'restroom female': 15, 'restroom male': 15,
                 'powder room': 12, 'toilet': 9, 'public restrooms': 7, 'restroom 1': 7, 'pantry restroom 1': 4,
                 'pantry restroom 4': 4, 'private restroom 1': 4, 'restroom 4': 4, 'staff restroom': 4,
                 'primary bathroom': 3, 'private restroom with shower': 3, 'restroom f1': 3, 'restroom f2': 3,
                 'restroom f3': 3, 'staff restrooms': 3, 'guest bathroom': 1, 'locker room showers': 1},
    'study_office': {'study room': 201, 'home office': 96, 'office': 84, 'library': 66, 'hub_room': 65, 'study': 23,
                     'focus booth 1': 17, 'focus booth 2': 17, 'collaboration zone 1': 12, 'copy print area 1': 12,
                     'open_plan_workstations': 9, 'collaboration zone 2': 8, 'open_office_2': 8,
                     'open_office_area_1': 8, 'open_office_area_2': 8, 'open_office_main': 8, 'copy/mail room': 4,
                     'executive assistant area 1': 4, 'open_office_3': 4, 'open_office_area_4': 4,
                     'open_office_support': 4, 'assistant workstation area': 3, 'copy room': 3, 'hub_and_lounge': 3,
                     'hub_and_reception': 3, 'hub_room_f2': 3, 'open_office_area': 3, 'phone booth 1': 3,
                     'phone booth 2': 3, 'Library': 1, 'office_space': 1, 'open_office': 1, 'study_room': 1},
    'meeting': {'conference room': 19, 'department meeting room 1': 12, 'meeting room small 1': 12, 'huddle room 1': 9,
                'huddle room 2': 8, 'large meeting room 1': 8, 'meeting room': 8, 'meeting room medium 1': 8,
                'board room': 4, 'boardroom 1': 4, 'large conference room': 4, 'main conference room': 4,
                'meeting room 1': 4, 'client meeting room': 3, 'executive boardroom': 3, 'large meeting room': 3,
                'medium conference room 1': 3, 'medium conference room 2': 3, 'conference': 2, 'meeting_room': 2,
                'conference_room': 1, 'small meeting room': 1},
    'balcony': {'balcony': 166, 'outdoor': 6, 'porch/terrace/deck/driveway': 5, 'greenhouse': 1},
    'laundry_utility': {'laundry room': 85, 'utility room': 11, 'laundromat': 9, 'it server room 1': 4,
                        'laundryroom/mudroom': 4, 'clean utility room': 3, 'soiled utility room': 3, 'server_room': 2,
                        'clean utility': 1, 'laundry_room': 1, 'mudroom': 1, 'server it room': 1, 'soiled utility': 1,
                        'utilityroom/toolroom': 1},
    'garage': {'garage': 66, 'workshop': 2},
    'circulation': {'corridor': 28, 'hallway': 23, 'lobby': 23, 'foyer': 6, 'aisle': 4, 'entrance hub f1': 4,
                    'entrance_hall': 4, 'landing_hallway_f2': 4, 'lobby_1': 4, 'upper_hallway': 4,
                    'entryway/foyer/lobby': 3, 'main_lobby': 3, 'vestibule': 2, 'back_lobby': 1, 'corridor_wing': 1,
                    'entry_vestibule': 1, 'lobby_and_waiting_area': 1, 'staff_hall': 1, 'victorian corridor': 1},
    'storage_closet': {'walk in closet': 19, 'closet': 16, 'linen closet': 15, 'store room': 12, 'cloakroom': 9,
                       'print storage room': 9, 'storage room 1': 8, 'storage room': 7, 'walk_in_closet': 6,
                       'wine cellar': 5, 'luggage storage': 3, 'medical supply storage': 3, 'storage': 3,
                       'storage_room': 3, 'dressing_room': 2, 'walk-in closet': 2, 'archive room': 1,
                       'linen_storage': 1, 'props_storage': 1, 'wine bar': 1, 'wine_cellar': 1},
    'public_commercial': {'restaurant_dining_area': 20, 'classroom': 17, 'bar_pub': 12, 'gym_fitness_center': 12,
                          'exam_room': 9, 'grocery store': 9, 'reception_lobby': 8, 'bakery': 7, 'clothing store': 7,
                          'hair salon': 7, 'reception': 7, 'warehouse': 7, 'art studio': 6, 'bookstore / library': 6,
                          'hospital room': 6, 'waiting_room': 6, 'grocery_store': 5, 'gym': 5, 'waiting room': 5,
                          'activity_space': 4, 'bar': 4, 'bookstore': 4, 'exhibition hall 1': 4, 'locker room': 4,
                          'prison cell': 4, 'reception desk area 1': 4, 'activity_area': 3, 'cafe bar': 3,
                          'computer lab': 3, 'consultation room 1': 3, 'consultation room 2': 3,
                          'consultation room 3': 3, 'consultation room 4': 3, 'consultation room 5': 3,
                          'creative studio': 3, 'emergency treatment room 1': 3, 'emergency treatment room 2': 3,
                          'emergency triage room': 3, 'exhibition_room': 3, 'fast_food_restaurant': 3,
                          'hospital patient room': 3, 'kindergarten classroom': 3, 'lab support': 3,
                          'laboratory sample collection': 3, 'locker_room': 3, 'main exhibition space': 3,
                          'main lab': 3, 'main_waiting_area': 3, 'minor procedure room': 3, 'pharmacy': 3,
                          'reception and registration': 3, 'reception area': 3, 'shoe store': 3,
                          'video rental store': 3, 'buffet_restaurant': 2, 'clothing_store': 2, 'computer_lab': 2,
                          'delicatessen': 2, 'fast food restaurant': 2, 'florist_shop': 2, 'lab': 2,
                          'music_recording_studio': 2, 'patient room': 2, 'reception_and_lobby': 2,
                          'restaurant dining area': 2, 'showroom': 2, 'toy store': 2, 'waiting_area': 2,
                          'activity_room_art': 1, 'activity_room_crafts': 1, 'activity_room_pingpong': 1,
                          'activity_room_yoga': 1, 'art_studio': 1, 'barber_shop': 1, 'cafe': 1, 'casino floor': 1,
                          'casino_floor': 1, 'clinic_corridor_waiting': 1, 'consulting room 1': 1,
                          'consulting room 2': 1, 'dance_studio': 1, 'daycare_room': 1, 'dentist_office': 1,
                          'diagnostic lab': 1, 'editing_room': 1, 'electronics_store': 1, 'examination room': 1,
                          'fitting_room': 1, 'florist shop': 1, 'flower_shop': 1, 'gallery': 1, 'grooming_salon': 1,
                          'hospital_room': 1, 'imaging room': 1, 'jewelry store': 1, 'laboratory': 1, 'main_hall': 1,
                          'museum_gallery': 1, 'nail_salon': 1, 'operating room': 1, 'pet_store': 1,
                          'physical_therapy_room': 1, 'pottery_store': 1, 'reception and records': 1,
                          'reception_area': 1, 'reception_waiting': 1, 'restaurant_dining': 1, 'salon_floor': 1,
                          'shooting_area': 1, 'spa_treatment_room': 1, 'surgery_room': 1, 'tattoo_parlor': 1,
                          'toy_store': 1, 'treatment room 1': 1, 'treatment room 2': 1, 'treatment_room': 1,
                          'treatment_vestibule': 1, 'triage station': 1, 'tv_studio': 1, 'warehouse studio': 1,
                          'workout_area': 1, 'xray_room': 1, 'yoga_room': 1, 'yoga_studio': 1},
    'other': {'other room': 29, 'apartment': 6, 'studio': 4, 'misc.': 2},
    'none': {None: 673},
}


@pytest.mark.parametrize('raw, expected', [
    (None, 'none'), ('  ', 'none'), ('other room', 'other'), ('misc.', 'other'), ('studio', 'other'),
    ('apartment', 'other'), ('Bedroom / Hotel', 'bedroom'), ('bedroom with home office', 'bedroom'),
    ('standard room 3', 'bedroom'), ("children's room", 'bedroom'), ('living dining room', 'living'),
    ('living and dining room', 'living'), ('dining room with living room', 'dining'), ('diningroom', 'dining'),
    ('kitchen with dining area', 'kitchen'), ('pantry 1', 'kitchen'), ('pantry restroom 4', 'bathroom'),
    ('restroom f1', 'bathroom'), ('hub_room', 'study_office'), ('home office', 'study_office'),
    ('focus booth 2', 'study_office'), ('conference room', 'meeting'), ('boardroom 1', 'meeting'),
    ('walk-in closet', 'storage_closet'), ('store room', 'storage_closet'), ('grocery store', 'public_commercial'),
    ('reception_lobby', 'public_commercial'), ('lobby_and_waiting_area', 'circulation'),
    ('entrance hub f1', 'circulation'), ('laundromat', 'laundry_utility'), ('utilityroom/toolroom', 'laundry_utility'),
    ('porch/terrace/deck/driveway', 'balcony'), ('workshop', 'garage'), ('game room 1', 'living'),
    ('wine bar', 'storage_closet'), ('main_hall', 'public_commercial')])
def test_family_rules(raw, expected):
    assert rf.family(raw) == expected


def test_every_test_string_keeps_its_frozen_family():
    frozen = {raw: (family, n) for family, strings in MAPPING.items() for raw, n in strings.items()}
    assert len(frozen) == 346 and sum(n for _, n in frozen.values()) == 8647 and list(MAPPING) == list(rf.FAMILIES)
    assert {raw: rf.family(raw) for raw in frozen} == {raw: family for raw, (family, _) in frozen.items()}
    raws = [raw for raw, (_, n) in frozen.items() for _ in range(n)]
    assert '| none | — | 673 | 1 | (none) (673) |' in rf.mapping_table(raws)
    assert '| total | | 8647 | 346 | |' in rf.mapping_table(raws)


@pytest.mark.skipif(not TEST_ROOMS.is_file(), reason='local data copy not mounted')
def test_the_frozen_strings_are_the_test_set():
    raws = collections.Counter(json.loads(line).get('room_type') for line in TEST_ROOMS.open())
    assert raws == {raw: n for strings in MAPPING.values() for raw, n in strings.items()}


def _views(values, groups):
    return [{'uid': f'u{i:02d}', 'source': 'ST'[i % 2], 'group': f'g{i % groups}', 'family': 'bedroom',
             'complete': v is not None, 'geometry_success': bool(v), 'downstream': bool(v),
             'bias_fp': v, 'size_log_mae': v, 'oor': v, 'size_iou': v, 'parent_accuracy': v, 'oor_repaired': v,
             'geometry_success_repaired': False} for i, v in enumerate(values)]


def test_contrast_is_paired_intervals_and_suppressed_below_ten_components():
    first = _views([1, 0, 1, 1, None, 0, 1, 1, 0, 1, 1, 1, 0, 1], 12)
    second = _views([0, 0, 1, 0, 1, None, 0, 1, 0, 0, 1, 0, 0, 1], 12)
    mine, ref = rf.contrast(first, second, 300), g.paired_intervals(first, second, 300)
    assert mine['components'] == 12 and mine['flags'] == 'few clusters; small n'
    assert mine['sources'] == {'S': 7, 'T': 7}
    for field in ('complete', 'geometry_success', 'size_log_mae', 'oor'):
        assert mine['fields'][field]['estimate'] == ref[field]['estimate_joint_minus_placement']
        assert mine['fields'][field]['ci95'] == ref[field]['ci95'] is not None
    few = rf.contrast(first[:5], second[:5], 300)
    assert few['components'] == 5 < rf.NO_CI
    assert all(v['ci95'] is None for v in few['fields'].values())


def _sealed(root, files, summary):
    for name, rows in files.items():
        write_rows(root / name, rows)
    h.write_json(root / 'summary.json', summary)
    h.write_json(root / 'MANIFEST.json', {'files': h.artifact_manifest(root)})
    return root


def test_report_end_to_end(tmp_path, monkeypatch):
    data = eval_data(tmp_path / 'data')
    manifest = json.loads((data / 'MANIFEST.json').read_text())
    expected, data_sha = manifest['expected_uids'], h.sha256(data / 'MANIFEST.json')
    rooms = {r['uid']: r for r in map(json.loads, (data / 'rooms.jsonl').read_text().splitlines())}
    requests = {arm: {r['uid']: r for r in map(json.loads, (data / f'{arm}/requests.jsonl').read_text().splitlines())}
                for arm in h.ARMS}
    missing = {'joint': {'x-00', 'x-01'}, 'placement': {'x-01'}}
    rows = {arm: [g.score(rooms[u], requests[arm][u]['messages'],
                          None if u in missing[arm] else target(None, arm, rooms[u]), arm) for u in expected]
            for arm in h.ARMS}
    runs = {'joint': 'j' * 64, 'placement': 'p' * 64}
    code = {'geometry.py': h.sha256(g.__file__)}
    geometry = _sealed(tmp_path / 'geometry-v5', {arm + '-geometry.jsonl': r for arm, r in rows.items()},
                       {'status': 'GEOMETRY_COMPLETE', 'stage': 'selected', 'data_manifest_sha256': data_sha,
                        'inference_run_sha256': runs, 'code_sha256': code})
    s1_rows = [{**r, 's1': {'geometry_success_s1': r['complete'], 'strict_ok_s1': r['complete'],
                            'geometry_s1': {'oor': 0.0}}} for r in rows['joint']]
    s1 = _sealed(tmp_path / 's1', {'joint-geometry-s1.jsonl': s1_rows},
                 {'status': 'S1_COMPLETE', 'requests_arm': 'joint',
                  'inputs': {'data_manifest_sha256': data_sha, 'geometry': 'joint-geometry.jsonl',
                             'geometry_manifest_sha256': h.sha256(geometry / 'MANIFEST.json')}})
    cohort = expected[:20]
    agent_rows = [{**r, 'arm': 'llm_raw', 'reached_token_cap': r['uid'] == 'il-bed-0', 'stopped_on_eos': False}
                  for r in (g.score(rooms[u], requests['joint'][u]['messages'], target(None, 'joint', rooms[u]),
                                    'joint') for u in cohort)]
    agent = _sealed(tmp_path / 'llm-eval', {'llm_raw-geometry.jsonl': agent_rows,
                                            'llm_harness-geometry.jsonl': [{**r, 'arm': 'llm_harness',
                                                                            'reached_token_cap': False}
                                                                           for r in agent_rows]},
                    {'status': 'LLM_COMPARISON_COMPLETE', 'data_manifest_sha256': data_sha, 'system': 'O9',
                     'model': 'm', 'inference_run_sha256': runs, 'code_sha256': code,
                     'cohort': {'uids_sha256': L._digest(json.dumps(cohort))}})
    capped = {'x-02', 'x-01'}  # x-01: a cap hit on an incomplete row flips nothing
    monkeypatch.setattr(rf.e, '_inference', lambda root, d, uids: (
        [{'uid': u, 'reached_token_cap': u in capped, 'stopped_on_eos': False} for u in uids],
        {'run_sha256': runs[Path(root).name]}))
    out = tmp_path / 'families'
    rf.main(['report', '--data', str(data), '--geometry', str(geometry), '--joint-run', str(tmp_path / 'joint'),
             '--placement-run', str(tmp_path / 'placement'), '--agent', f'O9={agent}', '--s1', str(s1),
             '--out', str(out), '--repeats', '200'])
    result = json.loads((out / 'families.json').read_text())
    assert result['family_counts'] == {'bedroom': 3, 'living': 1, 'kitchen': 1, 'study_office': 33, 'none': 1}
    assert result['provenance']['cap_rule'] == {'joint_placement': {'joint': 1, 'placement': 1},
                                                'agents_flipped_complete': {'O9-raw': 1, 'O9-harness': 0}}
    test = result['test']['all']['systems']
    assert test['J']['complete'] == 36 / 39 and test['P']['complete'] == 37 / 39 and test['GT']['complete'] == 1
    assert test['J']['bias_fp'] == test['GT']['bias_fp'] == test['GT']['size_log_mae'] == 0
    assert test['J+S1']['geometry_success'] == test['J+S1']['downstream'] == 36 / 39 and test['J+S1']['oor'] == 0
    office = result['test']['study_office']
    assert {k: office['design'][k] for k in ('rooms', 'components', 'flags')} == {
        'rooms': 33, 'components': 33, 'flags': 'small n'}
    assert office['systems']['GT']['downstream'] == 1 and office['systems']['J']['downstream'] == 30 / 33
    assert result['test']['bedroom']['design']['flags'] == 'few clusters; small n'
    assert result['test']['paper_families']['design']['rooms'] == 5
    assert result['test']['other_families']['design']['rooms'] == 33  # none and other are not other families
    assert all(v['ci95'] is None for v in result['contrasts']['J-P']['bedroom']['fields'].values())
    assert result['contrasts']['J-P']['all']['fields']['complete']['ci95'] is not None
    assert set(result['contrasts']) == {'J-P', 'P-O9-raw', 'P-O9-harness'}
    assert result['cohort']['all']['systems']['O9-raw']['complete'] == 19 / 20
    assert set(json.loads((out / 'MANIFEST.json').read_text())['files']) == {
        'families.json', 'report.md', 'levels.csv', 'contrasts.csv', 'mapping.md'}
    assert 'EXPLORATORY' in (out / 'report.md').read_text() and '| study_office |' in (out / 'mapping.md').read_text()
    with pytest.raises(FileExistsError):
        rf.run(data, geometry, out, runs)
    with pytest.raises(SystemExit):  # the cap rule is not optional
        rf.main(['report', '--data', str(data), '--geometry', str(geometry), '--out', str(tmp_path / 'no-runs')])
    with pytest.raises(ValueError, match='second S1'):
        rf.main(['report', '--data', str(data), '--geometry', str(geometry), '--joint-run', str(tmp_path / 'joint'),
                 '--placement-run', str(tmp_path / 'placement'), '--s1', str(s1), '--s1', str(s1),
                 '--out', str(tmp_path / 'twice'), '--repeats', '200'])
