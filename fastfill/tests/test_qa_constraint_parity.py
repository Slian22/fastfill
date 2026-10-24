"""QA containment must use the same hard-constraint protection as evaluation."""
import copy

from fastfill.evaluate import score
from fastfill.scene import messages
from fastfill.tools.qa_report import _train_metrics


def test_training_qa_cannot_claim_repair_by_breaking_a_satisfied_constraint():
    room = {
        'uid': 'qa-parity', 'source': 'synthetic', 'group': 'g',
        'boundary': [[0, 0], [4, 0], [4, 4], [0, 4]], 'boundary_type': 'polygon', 'height': 3,
        'objects': [
            {'id': 'chair_1', 'category': 'chair', 'size': [.5, .5, .9],
             'pos': [.25, .22, 0], 'yaw': 0, 'parent': None},
            {'id': 'box_1', 'category': 'box', 'size': [.06, .06, .3],
             'pos': [.60, .03, 0], 'yaw': 0, 'parent': None}],
        'constraints': [['faces', 'chair_1', 'box_1']],
    }
    row = {'messages': messages(room, room['constraints'])}
    original = copy.deepcopy(row)
    qa, runtime = _train_metrics([row]), score(copy.deepcopy(room), None)
    assert qa['constraint_reference_validation']['train'] == '1/1'
    assert runtime['inside_1mm_repaired'] is False
    assert qa['train reference inside walls within 1 mm']['repaired'] == '0/1'
    assert row == original
