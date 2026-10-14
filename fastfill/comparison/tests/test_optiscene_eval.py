"""Amendment 014 on CPU: toy 3D-SynthPlace scenes and training rows, toy eval-data, a fake vLLM, fake training dirs."""
import difflib
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
from shapely.geometry import Polygon

from fastfill import scene
from fastfill.adapters.optiscene_holodeck import raw_furniture
from fastfill.adapters.unified import convert_room
from fastfill.comparison import contract, geometry
from fastfill.comparison import optiscene_eval as oe
from fastfill.comparison.tests import test_infer_shard as T
from fastfill.validate import check

CODE = Path(__file__).resolve().parents[3]
SCRIPTS = CODE / 'fastfill/comparison'
ORIGINAL = CODE / 'OptiScene_Original'
INSTR = 'Design the room.\n\n'
FLOOR = [[-2, -1.5], [-2, 1.5], [2, 1.5], [2, -1.5]]
SHORT_SFT_INPUT = (  # The shortest released SFT input, verbatim.
    '[Task Room Type]\nbathroom\n[/Task Room Type]\n\n[Task Room Area]\n6.0\n[/Task Room Area]\n\n'
    '[Task Objects and Bounding Box Size]\n{"objects":\n[\n{\n    "object": "1 modern freestanding white bathtub",\n'
    '    "bounding box": [\n        {\n            "h": 0.75,\n            "w": 0.77,\n            "d": 1.74\n'
    '        }\n    ]\n},\n{\n    "object": "1 compact modern white toilet",\n    "bounding box": [\n        {\n'
    '            "h": 0.68,\n            "w": 0.56,\n            "d": 0.8\n        }\n    ]\n},\n{\n'
    '    "object": "1 small wooden bath stool",\n    "bounding box": [\n        {\n            "h": 0.42,\n'
    '            "w": 0.54,\n            "d": 0.54\n        }\n    ]\n}\n]\n}\n[/Task Objects and Bounding Box Size]\n\n')
ESCAPED_INPUT = (
    '[Task Room Type]\nliving room\n[/Task Room Type]\n\n[Task Room Area]\n21.37\n[/Task Room Area]\n\n'
    '[Task Objects and Bounding Box Size]\n{"objects":\n[\n{\n    "object": "2 caf\\u00e9 chair",\n'
    '    "bounding box": [\n        {\n            "h": 0.9,\n            "w": 0.45,\n            "d": 0.5\n        },\n'
    '        {\n            "h": 0.9,\n            "w": 0.45,\n            "d": 0.5\n        }\n    ]\n}\n]\n}\n'
    '[/Task Objects and Bounding Box Size]\n\n')


def sha12(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]


def design(groups, room_type='bedroom'):
    """[Design] text from [(name, [(x, y, z, angle), ...])]."""
    return '[Design]\n' + json.dumps({'room type': room_type, 'objects': [
        {'object': name, 'coordinates': [{'x': x, 'y': y, 'z': z} for x, y, z, _ in cs],
         'rotate': [{'angle': a} for *_, a in cs]} for name, cs in groups]}, indent=4) + '\n[/Design]'


def wardrobe(x, y=-1.2, bbox=(200, 200, 60), desc='a tall wardrobe'):
    return {'object_name': 'wardrobe-0', 'description': desc, 'bbox': list(bbox),
            'position': {'x': x, 'y': 0, 'z': y}, 'rotation': {'x': 0, 'y': 0, 'z': 0}}


def hf_scene(source, objects, room='bedroom'):
    return {'floor': FLOOR, 'room': room, 'objects': objects, 'source': source}


def training_row(uid, x, desc):
    box = [[2.0, 2.0, 0.6]]
    return {'uid': uid, 'instruction': INSTR, 'input': oe.render_input('bedroom', 12.0, [(desc, box)]),
            'output': design([(f'1 {desc}', [(x, -1.2, 0, 0.0)])])}


def write_official(root, sft, dpo1, dpo2, monkeypatch):
    root.mkdir(parents=True, exist_ok=True)
    for name, rows in (('sft_prompts.json', sft), ('dpo_prompts.json', dpo1), ('dpo2_prompts.json', dpo2)):
        (root / name).write_text(json.dumps(rows))
    monkeypatch.setattr(oe, 'DATA_FILES', {k: (v[0], '') for k, v in oe.DATA_FILES.items()})
    monkeypatch.setattr(oe, 'INSTRUCTION', (len(INSTR), sha12(INSTR)))
    return root


def pairs(prefix, xs):
    return [{'directory': 'd', 'combined_prompts': [training_row(f'{prefix}-{k}', x, f'{prefix} chest {k}'),
                                                     training_row(f'{prefix}-{k}', x + .5, f'{prefix} chest {k}')]}
            for k, x in enumerate(xs)]


@pytest.fixture
def world(tmp_path, monkeypatch):
    """HF: bedroom-1 seen (an SFT row copies it), bedroom-2/-3 unseen, bedroom-4 (3D-FRONT) has a NaN pose."""
    scenes = {'bedroom-1': hf_scene('HOLODECK', [wardrobe(-0.5)]), 'bedroom-2': hf_scene('HOLODECK', [wardrobe(0.0)]),
              'bedroom-3': hf_scene('HOLODECK', [wardrobe(0.5)]),
              'bedroom-4': hf_scene('3DFRONT', [wardrobe(float('nan'))])}
    hf = tmp_path / 'hf'
    (hf / 'scenes').mkdir(parents=True)
    for name, s in scenes.items():
        (hf / 'scenes' / f'{name}.json').write_text(json.dumps(s))
    xs = {'bedroom-1': -0.5, 'bedroom-2': 0.0, 'bedroom-3': 0.5, 'bedroom-4': 9.0}  # gprompt's design output
    (hf / 'prompts_all_scenes.json').write_text(json.dumps(
        [{'uid': f"{s['source']}-{n}", 'instruction': 'x', 'input': 'x',
          'output': design([('1 tall wardrobe', [(xs[n], -1.2, 0, 0.0)])])} for n, s in scenes.items()]))
    sft = [{'uid': 'HOLODECK-a_bedroom-1', 'instruction': INSTR, 'input': SHORT_SFT_INPUT,
            'output': design([('1 tall wardrobe', [(-0.5, -1.2, 0, 0.0)])])}]
    sft += [training_row(f'HOLODECK-a_bedroom-{k}', 1.0, f'oak dresser {k}') for k in range(2, 6)]
    data = write_official(tmp_path / 'data', sft, pairs('d1', [1, 1, 1, 1, 1]), pairs('d2', [1, 1, 1, 1, 1]),
                          monkeypatch)
    return SimpleNamespace(hf=hf, data=data, tmp=tmp_path)


# ---- vendored code and the README command --------------------------------------------------------------------------

def test_vendored_files_differ_from_upstream_only_by_f2_f3_and_f5():
    upstream = json.loads((oe.VENDORED / 'UPSTREAM.json').read_text())
    assert upstream['commit'] == oe.UPSTREAM_COMMIT and set(upstream['fixes']) == {'F1', 'F2', 'F3', 'F4', 'F5'}
    for name, v in upstream['files'].items():
        assert oe._sha(oe.VENDORED / name) == v['vendored_sha256']
        assert (v['upstream_sha256'] == v['vendored_sha256']) == (not v['fixes']), name
    assert {n for n, v in upstream['files'].items() if v['fixes']} == {'main.py', 'scripts/dpo_train.py',
                                                                       'scripts/sft_train.py'}
    if not (ORIGINAL / '.git').exists():
        pytest.skip('no OptiScene_Original clone')
    expected = {'main.py': ({'from dataclasses import dataclass', '@dataclass',
                             'class OptiSceneScriptArguments(ScriptArguments):  # F2 (FastFill amendment 014, see VENDORED.md)',
                             '    dataset_name: str = "local-json"  # unused; TRL 0.15.2 gives it no default',
                             '    dataset_file: str = None', '',
                             '        dpo_parser = TrlParser((OptiSceneScriptArguments, DPOConfig, ModelConfig))'},
                            {'        dpo_parser = TrlParser((ScriptArguments, DPOConfig, ModelConfig))'}),
                'scripts/dpo_train.py': (
                    {'from transformers.trainer_utils import get_last_checkpoint  # F3',
                     '    trainer.train(resume_from_checkpoint=get_last_checkpoint(training_args.output_dir))  # F3'},
                    {'    trainer.train()'}),
                'scripts/sft_train.py': (
                    {'        gradient_checkpointing=True,  # F5 (FastFill amendment 014, see VENDORED.md)'}, set())}
    for name, v in upstream['files'].items():
        original = subprocess.run(['git', '-C', str(ORIGINAL), 'show', f'{oe.UPSTREAM_COMMIT}:{name}'],
                                  capture_output=True, check=True).stdout
        assert hashlib.sha256(original).hexdigest() == v['upstream_sha256'], name
        diff = [d for d in difflib.ndiff(original.decode().splitlines(), (oe.VENDORED / name).read_text().splitlines())
                if d[:2] in ('+ ', '- ')]
        added, removed = {d[2:] for d in diff if d[0] == '+'}, {d[2:] for d in diff if d[0] == '-'}
        assert (added, removed) == expected.get(name, (set(), set())), name


def readme_command(title):
    text = (oe.VENDORED / 'README.md').read_text()
    block = text.split(title, 1)[1].split('```bash', 1)[1].split('```', 1)[0]
    return shlex.split(block.replace('\\\n', ' ').split('python main.py', 1)[1])


def test_stage_argv_is_the_readme_command_plus_f1_and_operational_flags():
    sft = readme_command('### 1. Supervised Fine-Tuning (SFT)')
    sft = [a.replace('dataset/', 'DATA/').replace('Qwen/Qwen2.5-7B-Instruct', 'BASE')
           .replace('outputs/Qwen-7B-SFT', 'OUT') for a in sft]
    assert oe.stage_argv('sft', 'DATA', 'BASE', 'OUT') == sft
    for stage, title, data, base, out in (
            ('dpo1', '### 2. DPO Stage 1', 'dataset/dpo1_prompts.json', 'outputs/Qwen-7B-SFT-merged',
             'outputs/Qwen-7B-DPO-Stage1'),
            ('dpo2', '### 3. DPO Stage 2', 'dataset/dpo2_prompts.json', 'outputs/Qwen-7B-DPO-Stage1-merged',
             'outputs/Qwen-7B-DPO-Stage2')):
        readme = readme_command(title)
        readme = [{data: 'DATA/' + oe.DATA_FILES[stage][0], base: 'BASE', out: 'OUT'}.get(a, a) for a in readme]
        assert oe.stage_argv(stage, 'DATA', 'BASE', 'OUT') == readme + oe.OPS_ARGV
    assert 'dpo1_prompts' not in ' '.join(oe.stage_argv('dpo1', 'DATA', 'BASE', 'OUT'))  # F1


@pytest.mark.skipif(importlib.util.find_spec('trl') is None, reason='trl is not installed (training venv only)')
def test_exact_readme_dpo_command_parses_with_the_vendored_parser():
    code = ('import sys; sys.path.insert(0, ""); import main; from trl import TrlParser, DPOConfig, ModelConfig\n'
            f'argv = {oe.dpo_argv("DATA/dpo_prompts.json", "BASE", "OUT")[2:]!r}\n'
            'argv[argv.index("--eval_strategy")] = "--evaluation_strategy"\n'
            's, t, m = TrlParser((main.OptiSceneScriptArguments, DPOConfig, ModelConfig)).parse_args_and_config(args=argv)\n'
            'assert (s.dataset_file, t.learning_rate, t.num_train_epochs, m.lora_r, m.torch_dtype) == '
            '("DATA/dpo_prompts.json", 5e-6, 10, 32, None)\n')
    subprocess.run([sys.executable, '-B', '-c', code], cwd=oe.VENDORED, check=True, env={**os.environ, 'USE_TF': '0'})


@pytest.mark.skipif(importlib.util.find_spec('trl') is None, reason='trl is not installed (training venv only)')
def test_f3_resume_past_a_partial_checkpoint_equals_an_uninterrupted_run(tmp_path, monkeypatch):
    """A tiny Qwen2 through the vendored main.py with the exact DPO argv on CPU: 60 pairs, 48 train, 10 epochs / 4."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    out = lambda x: '[Design] {"objects": [{"object": "1 bed", "coordinates": [{"x": %.2f, "y": 0}]}]} [/Design]' % x
    tok = Tokenizer(models.BPE(unk_token='<unk>'))
    tok.pre_tokenizer, tok.decoder = pre_tokenizers.ByteLevel(add_prefix_space=False), decoders.ByteLevel()
    tok.train_from_iterator([out(k / 7) + ' room design' for k in range(200)], trainers.BpeTrainer(
        vocab_size=300, special_tokens=['<unk>', '<|im_start|>', '<|im_end|>', '<|endoftext|>'],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token='<|im_end|>', pad_token='<|endoftext|>',
                                  unk_token='<unk>')
    tok.chat_template = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
                         "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
    base = tmp_path / 'base'
    Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                 num_attention_heads=4, num_key_value_heads=2)).save_pretrained(base)
    tok.save_pretrained(base)
    row = lambda k, x: {'uid': f'HOLODECK-a_bedroom-{k}', 'instruction': 'Design.\n', 'input': f'room {k}',
                        'output': out(x)}
    (tmp_path / 'dpo.json').write_text(json.dumps([{'directory': 'd', 'combined_prompts': [row(k, .1 * k),
                                                                                         row(k, -.1 * k)]}
                                                   for k in range(60)]))
    env = {**os.environ, 'USE_TF': '0', 'CUDA_VISIBLE_DEVICES': '', 'HF_DATASETS_CACHE': str(tmp_path / 'cache')}

    def train(out_dir, ok=True):
        run = subprocess.run([sys.executable, '-B', 'main.py', *oe.dpo_argv(tmp_path / 'dpo.json', base, out_dir)],
                             cwd=oe.VENDORED, env=env, capture_output=True, text=True)
        assert (run.returncode == 0) == ok, run.stderr[-3000:]
        return run.stderr
    monkeypatch.setitem(oe.STEPS, 'dpo1', 120)
    train(tmp_path / 'full')
    full = oe.checkpoint_rule('dpo1', tmp_path / 'full')
    cut = tmp_path / 'cut'  # Killed after checkpoint-100, while writing checkpoint-110.
    shutil.copytree(tmp_path / 'full' / 'checkpoint-100', cut / 'checkpoint-100')
    (cut / 'checkpoint-110').mkdir()
    assert "Can't find a valid checkpoint" in train(cut, ok=False)  # F3 resumes from the last checkpoint dir
    assert oe.quarantine_partial(cut, '7') == [str(cut / '.partial-checkpoint-110-job7')]
    train(cut)
    assert oe.checkpoint_rule('dpo1', cut)['adapter_sha256'] == full['adapter_sha256']


def test_numpy_split_matches_the_datasets_library():
    datasets = pytest.importorskip('datasets')
    for n in (5, 10, 37, 101):
        split = datasets.Dataset.from_dict({'i': list(range(n))}).train_test_split(test_size=0.2, seed=42)
        assert oe.split(n) == (split['train']['i'], split['test']['i'])


def test_sft_renderer_reproduces_released_inputs_byte_for_byte():
    for text in (SHORT_SFT_INPUT, ESCAPED_INPUT):
        room_type, area, groups, boxes = oe.parse_input(text)
        at, named = 0, []
        for name, n in groups:
            named.append((name.split(' ', 1)[1], boxes[at:at + n]))
            at += n
        assert oe.render_input(room_type, area, named) == text
    released = ORIGINAL / 'dataset/sft_prompts.json'
    if released.is_file():  # Local check against the release: every 97th row.
        for row in json.loads(released.read_text())[::97]:
            room_type, area, groups, boxes = oe.parse_input(row['input'])
            at, named = 0, []
            for name, n in groups:
                named.append((name.split(' ', 1)[1], boxes[at:at + n]))
                at += n
            assert oe.render_input(room_type, area, named) == row['input']


# ---- frames, OOR, parsing ------------------------------------------------------------------------------------------

def iou(a, b):
    pa, pb = Polygon(scene.footprint(a)), Polygon(scene.footprint(b))
    return pa.intersection(pb).area / pa.union(pb).area


def test_ir_t_round_trip_swap_control_and_holodeck_adapter_agree():
    for k in range(40):
        obj = {'size': [0.4 + .05 * k, 1.9 - .03 * k, .8], 'pos': [k * .1, -k * .05, 0.0], 'yaw': k * .37}
        t = oe.ir_to_t(obj, (1.5, -2.0))
        back = oe.t_to_ir('o', 'x', t['x'], t['y'], t['z'], t['angle'], t['box'], (1.5, -2.0))
        assert iou(obj, back) > .999 and back['size'] == obj['size']
        if abs(obj['size'][0] - obj['size'][1]) > .1:  # the swap is the box turned by 90 degrees
            assert iou(obj, oe.swapped([back])[0]) < .95
    raw = {'position': {'x': 1.25, 'y': 0.0, 'z': -0.4}, 'rotation': {'x': 0, 'y': 90, 'z': 0}, 'bbox': [75, 77, 174]}
    ours = oe.t_to_ir('o', 'bathtub', raw['position']['x'], raw['position']['z'], raw['position']['y'],
                      math.radians(raw['rotation']['y']), [b / 100 for b in raw['bbox']])
    adapter = convert_room({'room_boundary': [[-2, -2], [2, -2], [2, 2], [-2, 2]], 'furniture': [raw_furniture(
        {'furniture_category': 'bathtub', 'source_fields': {'layout_object': raw}})]},
        source='t', uid='t', group='g', boundary_type='polygon', front_offset_deg=-90)['objects'][0]
    assert adapter['size'] == ours['size'] == [1.74, 0.77, 0.75]
    assert abs(adapter['yaw'] - ours['yaw']) < 1e-9
    assert [round(a - b, 9) for a, b in zip(adapter['pos'], ours['pos'])] == [2.0, 2.0, 0.0]  # boundary min shift
    assert iou(adapter, {**ours, 'pos': [p + s for p, s in zip(ours['pos'], (2, 2, 0))]}) > .999


def test_oor_paper_standard_cases_and_validate_agreement():
    box = lambda x, y, w=1.0, d=1.0: {'pos': [x, y, 0], 'size': [d, w, 1], 'yaw': 0.0}
    assert oe.oor_paper([box(0, 0), box(5, 5)]) == 0
    assert oe.oor_paper([box(0, 0), box(.5, 0)]) == .25
    assert oe.oor_paper([box(0, 0), box(0, 0)]) == .5
    assert oe.oor_paper([box(0, 0)] * 4) == 1.5
    objs = [box(0, 0, w=2.0, d=.5), box(.75, 0, w=1.0, d=1.0)]
    assert oe.oor_paper(objs) != oe.oor_ours(objs)
    room = {'objects': [{**o, 'id': str(k), 'category': 'cabinet'} for k, o in enumerate(objs)]}
    assert oe.oor_ours(objs) == pytest.approx(check(room)['oor'])


GROUPS = [['1 bed', 1], ['2 nightstand', 2]]


@pytest.mark.parametrize('text, code', [
    ('no block here', 'no_design'),
    ('[Design]{"objects": [}[/Design]', 'bad_json'),
    ('[Design]{"objects": [{"object": "1 bed", "coordinates": [{"x": NaN, "y": 0}], "rotate": [{"angle": 0}]}]}'
     '[/Design]', 'nonfinite'),
    (design([('1 bed', [(0, 0, 0, 0)])]), 'missing_group'),
    (design([('1 bed', [(0, 0, 0, 0)]), ('2 nightstand', [(1, 1, 0, 0)] * 2), ('1 lamp', [(0, 0, 0, 0)])]),
     'extra_group'),
    (design([('1 bed', [(0, 0, 0, 0)]), ('2 nightstand', [(1, 1, 0, 0)])]), 'count_mismatch'),
    (design([('1 bed', [(0, 0, 0, 0)]), ('2 nightstand', [(1, 1, 0, 0), (1e999, 1, 0, 0)])]), 'nonfinite'),
    (design([('1 bed', [(0, 0, 0, 0)]), ('2 nightstand', [(1, '1', 0, 0), (1, 1, 0, 0)])]), 'bad_json')])
def test_parse_failure_codes(text, code):
    assert oe.parse_design(text, GROUPS) == (None, code)


def test_parse_takes_the_last_design_block_in_box_order():
    good = design([('2 nightstand', [(1, 2, 0, 0.5), (3, 4, 0, 1.5)]), ('1 bed', [(0, 0, 0, 3.14)])])
    coords, err = oe.parse_design('<reasoning>[Design]{}[/Design]</reasoning>\n' + good, GROUPS)
    assert err is None and coords == [(0, 0, 0, 3.14), (1, 2, 0, .5), (3, 4, 0, 1.5)]
    assert oe.PARSE_ERRORS == ('no_design', 'bad_json', 'nonfinite', 'missing_group', 'extra_group', 'count_mismatch')


def test_derive_on_floor_and_supporting_top():
    table = {'id': 't', 'size': [1, 1, .75], 'pos': [0, 0, 0], 'yaw': 0.0}
    lamp = {'id': 'l', 'size': [.2, .2, .4], 'pos': [.3, .3, .77], 'yaw': 0.0}
    away = {'id': 'a', 'size': [.2, .2, .4], 'pos': [3, 3, .77], 'yaw': 0.0}
    rug = {'id': 'r', 'size': [2, 2, .01], 'pos': [0, 0, 0], 'yaw': 0.0}
    assert oe.derive_on([table, lamp, away, rug]) == [None, 't', None, None]


# ---- E1 plan, gate, scoring ----------------------------------------------------------------------------------------

def test_strata_caps_rank_and_shortfalls():
    picked, receipt = oe.select_strata({'A': ['u3', 'u1', 'u2'], 'B': ['v1'], 'Z': ['z']}, {'A': 2, 'B': 3, 'C': 1})
    assert len(picked['A']) == 2 and set(picked['A']) < {'u1', 'u2', 'u3'} and picked['B'] == ['v1']
    assert picked['A'] == sorted(['u1', 'u2', 'u3'], key=lambda u: (oe._rank(u), u))[:2]
    assert receipt['A'] == {'population': 3, 'cap': 2, 'selected': 2, 'shortfall': 0}
    assert receipt['B']['shortfall'] == 2 and receipt['C'] == {'population': 0, 'cap': 1, 'selected': 0, 'shortfall': 1}
    assert receipt['_outside_caps'] == {'Z': 1}


def test_plan_e1_unseen_heldout_gate_and_write_once(world):
    out = world.tmp / 'e1'
    m = oe.plan_e1(world.hf, world.data, out)
    rows = oe._jsonl(out / 'requests.jsonl')
    assert m['unseen_by_stratum'] == {'HOLODECK/bedroom': 2, '3DFRONT/bedroom': 1}
    assert m['excluded'] == [{'uid': '3DFRONT-bedroom-4', 'stratum': '3DFRONT/bedroom', 'reason': 'nonfinite_pose'}]
    assert m['strata']['HOLODECK/bedroom'] == {'population': 2, 'cap': 50, 'selected': 2, 'shortfall': 48}
    assert m['strata']['3DFRONT/bedroom']['shortfall'] == 423
    assert m['gate']['geometry_success'] == 1.0 and m['gate']['swap_geometry_success'] == 0.0
    assert m['heldout20']['sft_eval_rows'] == 1 and m['counts'] == {'e1-unseen': 2, 'e1-heldout20': 1}
    unseen = [r for r in rows if r['population'] == 'e1-unseen']
    assert {r['uid'] for r in unseen} == {'HOLODECK-bedroom-2', 'HOLODECK-bedroom-3'}
    user = unseen[0]['messages']['sft'][0]['content']
    assert user.startswith(INSTR) and '"object": "1 tall wardrobe"' in user and '[Task Room Area]\n12.0\n' in user
    assert unseen[0]['messages']['reasoning'][0]['content'].startswith(oe.floor_plan_prompts())
    assert unseen[0]['room']['objects'][0]['size'] == [0.6, 2.0, 2.0] and unseen[0]['groups'] == [['1 tall wardrobe', 1]]
    assert json.loads((out / 'hf-coverage.json').read_text()) == {
        'HOLODECK-bedroom-1': 1.0, 'HOLODECK-bedroom-2': 0.0, 'HOLODECK-bedroom-3': 0.0, '3DFRONT-bedroom-4': 0.0}
    with pytest.raises(FileExistsError):
        oe.plan_e1(world.hf, world.data, out)
    (out / 'requests.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='hash mismatch'):
        oe._verified(out)


def test_plan_e1_excludes_rooms_whose_exact_input_was_trained_on(world):
    s = json.loads((world.hf / 'scenes' / 'bedroom-3.json').read_text())
    s['objects'][0]['bbox'] = [190, 200, 60]  # bedroom-3's input now differs from bedroom-2's
    (world.hf / 'scenes' / 'bedroom-3.json').write_text(json.dumps(s))
    load = lambda name: json.loads((world.data / name).read_text())
    sft, dpo1, dpo2 = load('sft_prompts.json'), load('dpo_prompts.json'), load('dpo2_prompts.json')
    held = sft[oe.split(len(sft))[1][0]]
    k = oe.split(len(dpo1))[0][0]  # A DPO1 training pair whose chosen prompt is the held-out input
    dpo1[k]['combined_prompts'][0] = {**dpo1[k]['combined_prompts'][0], 'input': held['input']}
    dpo2[0]['combined_prompts'][1] = training_row('d2-0', 1.0, 'tall wardrobe')  # bedroom-2's input, rejected side
    for name, rows in (('dpo_prompts.json', dpo1), ('dpo2_prompts.json', dpo2)):
        (world.data / name).write_text(json.dumps(rows))
    m = oe.plan_e1(world.hf, world.data, world.tmp / 'e1')
    assert [e for e in m['excluded'] if e['reason'] == 'seen_input'] == [
        {'uid': 'HOLODECK-bedroom-2', 'stratum': 'HOLODECK/bedroom', 'reason': 'seen_input'},
        {'uid': held['uid'], 'stratum': 'e1-heldout20', 'reason': 'seen_input'}]
    assert m['counts'] == {'e1-unseen': 1} and m['strata']['HOLODECK/bedroom']['population'] == 1


def test_plan_e1_gate_aborts_when_the_swap_control_passes(world):
    for name in ('bedroom-2', 'bedroom-3'):  # Square footprints: swapping w and d changes nothing.
        s = json.loads((world.hf / 'scenes' / f'{name}.json').read_text())
        s['objects'][0]['bbox'] = [200, 60, 60]
        (world.hf / 'scenes' / f'{name}.json').write_text(json.dumps(s))
    with pytest.raises(ValueError, match='gate failed'):
        oe.plan_e1(world.hf, world.data, world.tmp / 'e1')
    assert not (world.tmp / 'e1').exists()


def gt_design(row):
    """Their output for a plan row placed exactly at its GT (E1 room or E2 bound room)."""
    room, center = row['room'], row.get('center', (0.0, 0.0))
    by = {o['id']: o for o in room['objects']}
    ts = [oe.ir_to_t(by[i], center) for i in row.get('ids') or [o['id'] for o in room['objects']]]
    groups, at = [], 0
    for name, n in row['groups']:
        groups.append((name, [(t['x'], t['y'], t['z'], t['angle']) for t in ts[at:at + n]]))
        at += n
    return design(groups)


def test_score_e1_keeps_failed_rooms_in_denominators(world):
    oe.plan_e1(world.hf, world.data, world.tmp / 'e1')
    rows = oe._jsonl(world.tmp / 'e1' / 'requests.jsonl')
    unseen = [r for r in rows if r['population'] == 'e1-unseen']
    scored = [oe.score_e1(unseen[0], {'text': gt_design(unseen[0])}), oe.score_e1(unseen[1], {'text': 'nothing'}),
              oe.score_e1(unseen[1], None)]
    assert scored[0]['complete'] and scored[0]['geometry_success'] and scored[0]['oor_paper'] == 0
    assert [s['error'] for s in scored] == [None, 'no_design', 'missing_output']
    rates = oe._rates(scored)
    assert rates['requests'] == 3 and rates['complete_rate'] == rates['geometry_success_rate'] == 1 / 3
    assert rates['oor_paper'] == 0 and rates['errors'] == {'no_design': 1, 'missing_output': 1}
    held = [r for r in rows if r['population'] == 'e1-heldout20'][0]
    h = oe.score_e1(held, {'text': design([(g, [(k, 0, 0, 0)] * n) for k, (g, n) in enumerate(held['groups'])])})
    assert h['complete'] and h['oor'] is not None and not h['geometry_success']


# ---- E2 plan, v1 ids, scoring --------------------------------------------------------------------------------------

HOUSE = '0f3c9e7a-1111-4222-8333-444455556666'


def ir_rooms():
    holo = {'uid': 'optiscene:HOLODECK-bedroom-7', 'source': 'OptiScene_holodeck', 'group': 'holodeck:x',
            'room_type': 'bedroom', 'boundary': [[0, 0], [4, 0], [4, 3], [0, 3]], 'boundary_type': 'polygon',
            'height': 2.8, 'fixed': [{'id': 'door', 'category': 'door', 'size': [0.1, 0.9, 2.0],
                                      'pos': [3.95, 2.0, 0.0], 'yaw': 0.0, 'parent': None}],
            'objects': [{'id': 'w', 'category': 'wardrobe', 'size': [0.6, 2.0, 2.0], 'pos': [2.0, 0.3, 0.0],
                         'yaw': 3 * math.pi / 2, 'parent': None},
                        {'id': 'n', 'category': 'nightstand', 'size': [0.4, 0.4, 0.5], 'pos': [0.3, 2.6, 0.0],
                         'yaw': 0.0, 'parent': None},
                        {'id': 'n2', 'category': 'nightstand', 'size': [0.4, 0.4, 0.5], 'pos': [1.0, 2.6, 0.0],
                         'yaw': 0.0, 'parent': None},
                        {'id': 'l', 'category': 'table lamp', 'size': [0.2, 0.2, 0.4], 'pos': [0.3, 2.6, 0.5],
                         'yaw': 0.0, 'parent': 'n'}]}
    il3d = {'uid': 'il3d:abc', 'source': 'IL3D_3dfront', 'group': f'3dfront:{HOUSE}', 'room_type': 'LivingRoom',
            'boundary': [[0, 0], [5, 0], [5, 4], [0, 4]], 'boundary_type': 'polygon', 'height': None, 'fixed': [],
            'objects': [{'id': 's', 'category': 'sofa', 'size': [0.9, 2.2, 0.8], 'pos': [2.5, 0.5, 0.0],
                         'yaw': math.pi / 2, 'parent': None},
                        {'id': 't', 'category': 'coffee table', 'size': [0.6, 1.2, 0.45], 'pos': [2.5, 2.0, 0.0],
                         'yaw': 0.0, 'parent': None}]}
    return holo, il3d


@pytest.fixture
def e2world(tmp_path, monkeypatch):
    labels, files = [], {}
    for k, room in enumerate(ir_rooms()):
        bound, _ = contract.bind_room(room)
        labels.append({**bound, 'group': f'c:{k}', 'component': f'c:{k}', 'split': 'test'})
    prior = {r['uid']: {o['id']: [round(v * 1.1, 2) for v in o['size']] for o in r['objects']} for r in labels}
    meta = lambda r: {k: r[k] for k in ('uid', 'source', 'group', 'component', 'split', 'room_type')}
    files['rooms.jsonl'] = ''.join(json.dumps(r) + '\n' for r in labels)
    files['placement/requests.jsonl'] = ''.join(json.dumps({**meta(r), 'messages': contract.messages(
        r, 'placement', estimated_sizes=prior[r['uid']])[:2]}) + '\n' for r in labels)
    files['size-estimates.jsonl'] = ''.join(json.dumps({'uid': u, 'estimated_sizes': s}) + '\n' for u, s in prior.items())
    files['gpt-selection.json'] = json.dumps({'strata': {'IL3D_3dfront/bedroom': {'population': 0},
                                                         'IL3D_3dfront/living room': {'population': 1},
                                                         'OptiScene_holodeck/bedroom': {'population': 1}}})
    files['inventory-sidecar.json'] = json.dumps({r['uid']: {'original_group': r['group']} for r in ir_rooms()})
    data = tmp_path / 'eval-data-v1'
    for name, text in files.items():
        (data / name).parent.mkdir(parents=True, exist_ok=True)
        (data / name).write_text(text)
    (data / 'MANIFEST.json').write_text(json.dumps({'files': {n: oe._sha(data / n) for n in files}}))
    holo = labels[0]
    request = oe.optiscene_request(holo, {o['id']: o['size'] for o in holo['objects']}, INSTR)
    seen = {'uid': 'HOLODECK-a_bedroom-70', 'instruction': INSTR,
            'input': request['messages'][0]['content'][len(INSTR):],
            'output': gt_design({'room': holo, **request})}
    sft = [seen] + [training_row(f'HOLODECK-a_bedroom-{k}', 1.0, f'oak dresser {k}') for k in range(2, 6)]
    sft[1] = {**sft[1], 'uid': '3DFRONT-99999999-aaaa-4bbb-8ccc-dddddddddddd-living room'}
    official = write_official(tmp_path / 'data-e2', sft, pairs('d1', [1] * 5), pairs('d2', [1] * 5), monkeypatch)
    return SimpleNamespace(data=data, official=official, labels=labels, prior=prior, tmp=tmp_path)


def v1_text(row, room):
    """FastFill v1 answer at GT, in canonical ids."""
    canonical = {b: c for c, b in row['id_map'].items()}
    by = {o['id']: o for o in room['objects']}
    items = []
    for c in row['order']:
        o = by[row['id_map'][c]]
        items.append({'id': c, **({'on': canonical[o['parent']]} if o.get('parent') else {}),
                      'pos': o['pos'], 'yaw': round(math.degrees(o['yaw'])) % 360})
    return json.dumps({'placements': items})


def test_plan_e2_requests_exposure_and_v1_ids_round_trip(e2world):
    e1 = e1_stub(e2world.tmp / 'e1', {'HOLODECK-bedroom-7': 0.0})
    m = oe.plan_e2(e2world.data, e2world.official, e1, e2world.tmp / 'e2')
    rows = oe._jsonl(e2world.tmp / 'e2' / 'requests.jsonl')
    exposure = json.loads((e2world.tmp / 'e2' / 'exposure.json').read_text())
    assert m['rooms'] == 2 and len(rows) == 8 and m['population_by_stratum'] == {
        'IL3D_3dfront/living room': 1, 'OptiScene_holodeck/bedroom': 1}
    assert exposure['optiscene:HOLODECK-bedroom-7']['seen'] is True
    assert exposure['il3d:abc'] == {'rule': 'house', 'house': HOUSE, 'seen': False}
    holo = e2world.labels[0]
    opt = {r['mode']: r for r in rows if r['uid'] == holo['uid'] and r['format'] == 'optiscene'}
    assert sorted(opt['gt']['groups']) == [['1 table lamp', 1], ['1 wardrobe', 1], ['2 nightstand', 2]]
    k = opt['gt']['ids'].index(next(o['id'] for o in holo['objects'] if o['category'] == 'wardrobe'))
    assert opt['gt']['boxes'][k] == [2.0, 2.0, 0.6] and opt['prior']['boxes'][k] == [2.2, 2.2, 0.66]
    assert opt['gt']['center'] == [2.0, 1.5]
    assert '[Task Room Type]\nbedroom\n' in opt['gt']['messages']['sft'][0]['content']
    for r in rows:
        room = next(x for x in e2world.labels if x['uid'] == r['uid'])
        request = contract.messages(room, 'placement', estimated_sizes={o['id']: o['size'] for o in room['objects']})[:2]
        text = v1_text(r, room) if r['format'] == 'v1' else gt_design({'room': room, **r})
        raw, err = oe.placements({**r, '_order': [o['id'] for o in room['objects']]}, text)
        assert err is None and [p['id'] for p in raw['placements']] == [o['id'] for o in room['objects']]
        assert [p.get('on') for p in raw['placements']] == [o.get('parent') for o in room['objects']]
        scored = geometry.score(room, request, json.dumps(raw), 'placement')
        assert scored['complete'] and scored['geometry_success'], (r['key'], scored)
        assert max(math.dist(a['pos'], b['pos']) for a, b in zip(scored['raw_layout'], room['objects'])) < 1e-9
    v1 = next(r for r in rows if r['format'] == 'v1' and r['uid'] == holo['uid'])
    user = json.loads(v1['messages']['v1'][1]['content'])
    assert v1['messages']['v1'][0]['content'] == scene.SYSTEM_PROMPT and 'pos' not in user['objects'][0]
    assert [o['id'] for o in user['objects']] == v1['order'] and 'desc' not in user['objects'][0]
    with pytest.raises(FileExistsError):
        oe.plan_e2(e2world.data, e2world.official, e1, e2world.tmp / 'e2')


def test_plan_e2_refuses_a_population_mismatch_and_a_tampered_input(e2world):
    e1 = e1_stub(e2world.tmp / 'e1', {})
    sel = e2world.data / 'gpt-selection.json'
    sel.write_text(json.dumps({'strata': {'OptiScene_holodeck/bedroom': {'population': 2}}}))
    with pytest.raises(ValueError, match='hash mismatch'):
        oe.plan_e2(e2world.data, e2world.official, e1, e2world.tmp / 'e2')
    manifest = json.loads((e2world.data / 'MANIFEST.json').read_text())
    manifest['files']['gpt-selection.json'] = oe._sha(sel)
    (e2world.data / 'MANIFEST.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='differ from gpt-selection'):
        oe.plan_e2(e2world.data, e2world.official, e1, e2world.tmp / 'e2')
    assert oe.plan_e2(e2world.data, e2world.official, e1, e2world.tmp / 'smoke', limit=1)['rooms'] == 1
    with pytest.raises(ValueError, match='not a plan-e1'):
        oe.plan_e2(e2world.data, e2world.official, e2world.tmp / 'smoke', e2world.tmp / 'other', limit=1)


def test_e2_exposure_snaps_float_noise_and_uses_the_hf_scene_coverage(e2world, monkeypatch):
    assert round(1.2 - 0.65, 1) == 0.5 and oe._q(1.2 - 0.65) == oe._q(0.55) == 0.6  # 0.5499999999999999
    obj = {'pos': [1.2, 0.3, 0.0], 'size': [0.6, 2.0, 2.0]}
    row = training_row('HOLODECK-a_bedroom-9', 0.55, 'tall wardrobe')
    row['output'] = row['output'].replace('-1.2', '0.3')
    assert oe.holodeck_keys([obj], (0.65, 0.0)) == oe._training_box_keys(row) == {(0.6, 0.3, 2.0, 0.6)}
    monkeypatch.setattr(oe, 'exposure_index', lambda rows: (set(), oe.Coverage([])))  # no xywd match at all
    uid = 'optiscene:HOLODECK-bedroom-7'
    for name, coverage, seen in (('hi', {'HOLODECK-bedroom-7': 0.6}, True), ('lo', {'HOLODECK-bedroom-7': 0.4}, False),
                                 ('none', {}, False)):
        m = oe.plan_e2(e2world.data, e2world.official, e1_stub(e2world.tmp / f'e1-{name}', coverage),
                       e2world.tmp / f'e2-{name}')
        exposure = json.loads((e2world.tmp / f'e2-{name}' / 'exposure.json').read_text())[uid]
        assert (exposure['seen'], exposure['hf_scene_coverage'], m['hf_scene_missing']) == (
            seen, coverage.get('HOLODECK-bedroom-7'), int(not coverage))


def e1_stub(path, coverage):
    oe._publish_files(path, {'requests.jsonl': b'', 'hf-coverage.json': json.dumps(coverage).encode()}, {'plan': 'e1'})
    return path


def publish_gen(gen, name, plan, records):
    oe._publish_files(gen / name, {'records.jsonl': oe._jsonl_bytes(records)},
                      {'plan_manifest_sha256': oe._sha(plan / 'MANIFEST.json'), 'limit': None})


def test_score_end_to_end_with_fake_generations_and_p(world, e2world):
    tmp = world.tmp
    oe.plan_e1(world.hf, world.data, tmp / 'e1')
    oe.plan_e2(e2world.data, e2world.official, tmp / 'e1', tmp / 'e2')
    rooms = {r['uid']: r for r in e2world.labels}
    plans = {k: oe._jsonl(tmp / k / 'requests.jsonl') for k in ('e1', 'e2')}
    for model, plan, prompt in oe.GENS:
        records = []
        for r in plans[plan]:
            if prompt not in r['messages']:
                continue
            if plan == 'e1':
                text = gt_design(r) if r['population'] == 'e1-unseen' else 'no answer'
            elif r['format'] == 'v1':
                text = v1_text(r, rooms[r['uid']])
            else:
                text = gt_design({'room': rooms[r['uid']], **r}) if r['mode'] == 'gt' or model != 'sft' else '[Design]{[/Design]'
            records.append({'key': r['key'], 'uid': r['uid'], 'text': text, 'finish_reason': 'stop', 'error': None})
        publish_gen(tmp / 'gen', f'{model}-{plan}-{prompt}', tmp / plan, records)
    p_rows, p_records = [], []
    for r in e2world.labels:
        request = contract.messages(r, 'placement', estimated_sizes=e2world.prior[r['uid']])
        p_rows.append(geometry.score(r, request[:2], request[2]['content'], 'placement'))
        p_records.append({'uid': r['uid'], 'reached_token_cap': r['source'] == 'IL3D_3dfront', 'stopped_on_eos': False})
    oe._publish_files(tmp / 'geometry-v5', {'placement-geometry.jsonl': oe._jsonl_bytes(p_rows),
                                            'summary.json': json.dumps({'inference_run_sha256': {
                                                'placement': 'r' * 64}}).encode()}, {})
    oe._publish_files(tmp / 'infer-placement-005', {'records.jsonl': oe._jsonl_bytes(p_records),
                                                    'run_receipt.json': json.dumps({'run_sha256': 'r' * 64}).encode()},
                      {})
    m = oe.score(tmp / 'e1', tmp / 'e2', tmp / 'gen', e2world.data, tmp / 'geometry-v5', tmp / 'infer-placement-005',
                 tmp / 'score', workers=1, repeats=2)
    summary = json.loads((tmp / 'score' / 'summary.json').read_text())
    e1 = summary['e1']['dpo2-e1-reasoning']
    assert e1['e1-unseen']['pooled']['geometry_success_rate'] == 1.0 and e1['e1-unseen']['pooled']['requests'] == 2
    held = e1['e1-heldout20']['pooled']
    assert held['requests'] == 1 and held['complete_rate'] == 0.0 and held['geometry_success_rate'] is None
    full = summary['e2']['dpo2-e2-sft']['prior']['all']
    assert full['full']['pooled']['requests'] == 2 and full['full']['pooled']['geometry_success_rate'] == 1.0
    assert full['floor']['pooled']['requests'] == 2 and full['floor']['pooled']['geometry_success_rate'] == 1.0
    assert summary['e2']['dpo2-e2-sft']['prior']['unseen']['full']['pooled']['requests'] == 1  # IL3D only
    sft_prior = summary['e2']['sft-e2-sft']['prior']['all']['full']['pooled']
    assert sft_prior['complete_rate'] == 0.0 and sft_prior['errors'] == {'parse:bad_json': 2}
    p = summary['e2']['P']['prior']['all']['full']['pooled']
    assert p['requests'] == 2 and p['complete_rate'] == 0.5  # the IL3D P output hit the token cap
    assert summary['paired_vs_P']['8b-e2-v1']['geometry_success']['replicates'] == 2
    assert summary['training_exposure']['8b-e2-v1'] == oe.EIGHT_B_EXPOSURE and len(summary['training_exposure']) == 4
    assert m['label'] == oe.LABEL and set(m['files']) == {'e1-rows.jsonl', 'e2-rows.jsonl', 'p-rows.jsonl',
                                                          'summary.json'}
    with pytest.raises(FileExistsError):
        oe.score(tmp / 'e1', tmp / 'e2', tmp / 'gen', e2world.data, tmp / 'geometry-v5', tmp / 'infer-placement-005',
                 tmp / 'score', workers=1, repeats=2)
    with pytest.raises(ValueError, match='another e1 plan'):
        oe.score(e1_stub(tmp / 'e1-other', {}), tmp / 'e2', tmp / 'gen', e2world.data, tmp / 'geometry-v5',
                 tmp / 'infer-placement-005', tmp / 'score-2', workers=1, repeats=2)


# ---- generation (fake vLLM) ----------------------------------------------------------------------------------------

class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kw):
        assert tokenize is False and add_generation_prompt is True
        self.kwargs = kw
        return '|'.join(m['content'] for m in messages)

    def __call__(self, text, add_special_tokens=False):
        return {'input_ids': [0] * (len(text) // 4)}


class FakeLLM:
    def __init__(self, answer):
        self.tok, self.answer, self.params = FakeTokenizer(), answer, None

    def get_tokenizer(self):
        return self.tok

    def generate(self, prompts, params):
        self.params = params
        return [SimpleNamespace(outputs=[SimpleNamespace(text=self.answer(p), finish_reason='stop', token_ids=[1, 2])])
                for p in prompts]


def test_generate_records_prompt_too_long_gate_and_refusals(world, monkeypatch):
    oe.plan_e1(world.hf, world.data, world.tmp / 'e1')
    oe._publish_files(world.tmp / 'model', {'model.safetensors': b'w'}, {})
    rows = oe._jsonl(world.tmp / 'e1' / 'requests.jsonl')
    llm = FakeLLM(lambda p: gt_design(next(r for r in rows if r['messages']['sft'][0]['content'] == p))
                  if 'tall wardrobe' in p and 'oak' not in p else 'no')
    monkeypatch.setattr(oe, '_engine', lambda engine: (llm, lambda **kw: kw))
    lengths = [len(r['messages']['sft'][0]['content']) // 4 for r in rows]
    monkeypatch.setattr(oe, 'MAX_MODEL_LEN', max(lengths))  # The longest prompts leave no room.
    m = oe.generate(world.tmp / 'e1', 'sft', world.tmp / 'model', world.tmp / 'gen')
    records = oe._jsonl(world.tmp / 'gen' / 'records.jsonl')
    assert [r['key'] for r in records] == [r['key'] for r in rows] and m['requests'] == 3
    assert [r['error'] == 'prompt_too_long' for r in records] == [n == max(lengths) for n in lengths]
    assert [r['text'] is None for r in records] == [n == max(lengths) for n in lengths]
    assert all(p['temperature'] == 0 and p['seed'] == 0 and 0 < p['max_tokens'] <= oe.MAX_TOKENS for p in llm.params)
    assert m['readme_sampling'] == oe.README_SAMPLING
    assert llm.tok.kwargs == {} and m['engine']['seed'] == 0 and m['plan_manifest_sha256'] == oe._sha(
        world.tmp / 'e1' / 'MANIFEST.json')
    with pytest.raises(FileExistsError):
        oe.generate(world.tmp / 'e1', 'sft', world.tmp / 'model', world.tmp / 'gen')
    with pytest.raises(ValueError, match='strict parse rate'):
        oe.generate(world.tmp / 'e1', 'sft', world.tmp / 'model', world.tmp / 'gated', gate_parse=0.9)
    assert not (world.tmp / 'gated').exists()
    (world.tmp / 'model' / 'model.safetensors').write_bytes(b'x')
    with pytest.raises(ValueError, match='hash mismatch'):
        oe.generate(world.tmp / 'e1', 'sft', world.tmp / 'model', world.tmp / 'other')


def test_eight_b_identity_checks(tmp_path):
    model = tmp_path / '8b'
    model.mkdir()
    (model / 'config.json').write_text(json.dumps(oe.EIGHT_B))
    (model / 'tokenizer_config.json').write_text(json.dumps({'chat_template': '{% if enable_thinking %}'}))
    (model / 'generation_config.json').write_text('{}')
    assert set(oe.check_eight_b(model)['files']) == {'config.json', 'tokenizer_config.json', 'generation_config.json'}
    (model / 'README.md').write_text('card')
    with pytest.raises(ValueError, match='README'):
        oe.check_eight_b(model)


# ---- training: checkpoint rule, train, merge -----------------------------------------------------------------------

def make_stage(out, stage, step=None, max_steps=None, final=b'a'):
    step = step or oe.STEPS[stage]
    ckpt = out / f'checkpoint-{step}'
    ckpt.mkdir(parents=True)
    (ckpt / 'trainer_state.json').write_text(json.dumps({'global_step': step, 'max_steps': max_steps or step}))
    (ckpt / 'adapter_model.safetensors').write_bytes(b'a')
    if stage != 'sft' and final is not None:
        (out / 'adapter_model.safetensors').write_bytes(final)
    return ckpt


def test_checkpoint_rule_raises_on_missing_or_mismatched_steps(tmp_path):
    with pytest.raises(ValueError, match='last checkpoint None'):
        oe.checkpoint_rule('sft', tmp_path / 'none')
    make_stage(tmp_path / 'ok', 'sft')
    assert oe.checkpoint_rule('sft', tmp_path / 'ok')['global_step'] == 1226
    make_stage(tmp_path / 'short', 'dpo1', step=15800)
    with pytest.raises(ValueError, match='is not checkpoint-15900'):
        oe.checkpoint_rule('dpo1', tmp_path / 'short')
    make_stage(tmp_path / 'state', 'dpo1', max_steps=16000)
    with pytest.raises(ValueError, match='max_steps'):
        oe.checkpoint_rule('dpo1', tmp_path / 'state')
    make_stage(tmp_path / 'copy', 'dpo2', final=b'b')
    with pytest.raises(ValueError, match='save_model copy'):
        oe.checkpoint_rule('dpo2', tmp_path / 'copy')
    make_stage(tmp_path / 'nosave', 'dpo2', final=None)
    assert not oe.stage_complete('dpo2', tmp_path / 'nosave')


@pytest.fixture
def trainer(tmp_path, monkeypatch):
    env = tmp_path / 'env.json'
    env.write_text(json.dumps({'ok': True, 'sys_prefix': sys.prefix, 'model': {'path': str(tmp_path / 'base'),
                                                                                'files': {}}}))
    (tmp_path / 'base').mkdir()
    sacct = tmp_path / 'bin' / 'sacct'
    sacct.parent.mkdir()
    sacct.write_text('#!/bin/sh\necho "$SACCT_STATE"\n')
    sacct.chmod(0o755)
    monkeypatch.setenv('PATH', f"{sacct.parent}:{os.environ['PATH']}")
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.setenv('SLURM_JOB_ID', '77')
    data = tmp_path / 'data'
    data.mkdir()
    for name, _ in oe.DATA_FILES.values():
        (data / name).write_text('[]')
    monkeypatch.setattr(oe, 'DATA_FILES', {k: (v[0], '') for k, v in oe.DATA_FILES.items()})
    calls, real = [], subprocess.run

    def fake(argv, **kw):
        if argv[0] == 'sacct':
            return real(argv, **kw)
        calls.append((argv, kw))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(oe.subprocess, 'run', fake)
    return SimpleNamespace(env=env, data=data, run=tmp_path / 'run', base=tmp_path / 'base', calls=calls)


def test_train_previous_job_state_single_gpu_resume_and_completion(trainer, monkeypatch):
    t = trainer
    monkeypatch.setenv('SACCT_STATE', 'FAILED')
    with pytest.raises(ValueError, match='ended FAILED'):
        oe.train('sft', t.run, t.data, t.base, t.env, prev_job='5')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1')
    with pytest.raises(ValueError, match='exactly one visible GPU'):
        oe.train('sft', t.run, t.data, t.base, t.env)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    assert oe.train('sft', t.run, t.data, t.base, t.env) == 0
    argv, kw = t.calls[-1]
    assert argv[2:] == ['main.py'] + oe.stage_argv('sft', t.data, t.base, t.run / 'sft' / 'out') and kw['cwd'] == oe.VENDORED
    start = json.loads((t.run / 'sft' / 'receipts' / 'job-77-start.json').read_text())
    assert start['fixes'] == oe.FIXES and start['kept_as_released'] == list(oe.KEPT)
    (t.run / 'sft' / 'out' / 'partial').mkdir()
    with pytest.raises(ValueError, match='cannot resume'):
        oe.train('sft', t.run, t.data, t.base, t.env)
    make_stage(t.run / 'dpo1' / 'out', 'dpo1')
    monkeypatch.setenv('SACCT_STATE', 'TIMEOUT')
    n = len(t.calls)
    assert oe.train('dpo1', t.run, t.data, t.base, t.env, prev_job='9') == 0 and len(t.calls) == n  # complete


def test_previous_job_state_is_polled_while_sacct_lags(trainer, monkeypatch):
    sleeps = []
    monkeypatch.setattr(oe.time, 'sleep', sleeps.append)
    states = iter([None, 'RUNNING', 'COMPLETING', 'TIMEOUT'])
    monkeypatch.setattr(oe, '_job_state', lambda job: next(states))
    assert oe._final_state('5') == 'TIMEOUT' and sleeps == [10, 10, 10]
    monkeypatch.setattr(oe, '_job_state', lambda job: 'RUNNING')
    with pytest.raises(ValueError, match='ended RUNNING'):
        oe.train('dpo1', trainer.run, trainer.data, trainer.base, trainer.env, prev_job='5')
    assert len(sleeps) == 3 + 59  # about 10 minutes, then refuse
    monkeypatch.undo()
    (trainer.base.parent / 'bin' / 'sacct').write_text('#!/bin/sh\nexit 1\n')
    monkeypatch.setenv('PATH', f"{trainer.base.parent / 'bin'}:{os.environ['PATH']}")
    assert oe._job_state('5') is None  # a failing sacct counts as not yet recorded


def test_partial_checkpoints_are_quarantined_not_deleted(tmp_path):
    make_stage(tmp_path, 'dpo1', step=100)
    (tmp_path / 'checkpoint-200').mkdir()
    assert oe.quarantine_partial(tmp_path, '7') == [str(tmp_path / '.partial-checkpoint-200-job7')]
    assert (tmp_path / 'checkpoint-100').is_dir() and (tmp_path / '.partial-checkpoint-200-job7').is_dir()


def test_merge_applies_the_checkpoint_rule_and_publishes_once(trainer, monkeypatch):
    t = trainer
    with pytest.raises(ValueError, match='last checkpoint'):
        oe.merge('sft', t.run, t.base, t.env)
    make_stage(t.run / 'sft' / 'out', 'sft')
    calls = []

    def fake(argv, **kw):
        calls.append((argv, kw))
        Path(argv[argv.index('--output_path') + 1], 'model.safetensors').write_bytes(b'merged')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(oe.subprocess, 'run', fake)
    m = oe.merge('sft', t.run, t.base, t.env)
    assert m['receipt']['checkpoint_rule']['global_step'] == 1226 and calls[0][1]['env']['CUDA_VISIBLE_DEVICES'] == ''
    assert set(m['files']) == {'model.safetensors'} and oe._verified(t.run / 'sft-merged')['receipt']
    with pytest.raises(FileExistsError):
        oe.merge('sft', t.run, t.base, t.env)


# ---- shell scripts -------------------------------------------------------------------------------------------------

def test_scripts_parse():
    for name in ('isambard_014.sbatch', 'submit_014.sh', 'setup_optiscene_env.sh'):
        subprocess.run(['bash', '-n', str(SCRIPTS / name)], check=True)
    setup = (SCRIPTS / 'setup_optiscene_env.sh').read_text()
    assert 'trl==0.15.2 transformers==4.49.0 peft==0.14.0 accelerate==1.4.0 datasets==3.3.2 modelscope==1.23.2' in setup
    assert '--system-site-packages' in setup and 'PIP_CACHE_DIR="$PROJECT_ROOT/.cache-jobs/pip"' in setup


def test_setup_resumes_an_interrupted_download_at_its_pinned_revision(tmp_path):
    project = tmp_path / 'project'
    for name, _ in oe.DATA_FILES.values():
        (project / 'data/optiscene-official' / name).parent.mkdir(parents=True, exist_ok=True)
        (project / 'data/optiscene-official' / name).write_text('[]')
    venv = project / 'envs/optiscene-014/bin'
    venv.mkdir(parents=True)
    stubs = {'python': 'case "$*" in *HfApi*) cat "$T/rev" ;; *env-check*) echo env-check >> "$T/log" ;; *) echo t ;; esac',
             'huggingface-cli': 'echo "$*" >> "$T/log"; test ! -e "$T/fail"'}
    for name, body in stubs.items():
        (venv / name).write_text(f'#!/bin/sh\n{body}\n')
        (venv / name).chmod(0o755)
    line = 'PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill\n'
    text = (SCRIPTS / 'setup_optiscene_env.sh').read_text()
    assert text.count(line) == 1
    (tmp_path / 'setup.sh').write_text(text.replace(line, f'PROJECT_ROOT={project}\n'))
    run = lambda: subprocess.run(['bash', str(tmp_path / 'setup.sh'), str(CODE)], capture_output=True, text=True,
                                 env={**os.environ, 'T': str(tmp_path)})
    base = project / 'models/Qwen2.5-7B-Instruct'
    (tmp_path / 'rev').write_text('rev1\n')
    (tmp_path / 'fail').touch()
    assert run().returncode != 0 and not Path(f'{base}.revision').exists()  # interrupted download
    assert Path(f'{base}.revision.pending').read_text() == 'rev1\n'
    (tmp_path / 'rev').write_text('rev2\n')  # the hub moved on
    (tmp_path / 'fail').unlink()
    assert run().returncode == 0 and Path(f'{base}.revision').read_text() == 'rev1\n'
    assert run().returncode == 0
    log = (tmp_path / 'log').read_text().splitlines()
    assert [x.split('--revision ')[1].split()[0] for x in log if x.startswith('download')] == ['rev1', 'rev1']
    assert log.count('env-check') == 2 and not Path(f'{base}.revision.pending').exists()


@pytest.mark.parametrize('args, ok', [
    (['train', 'sft', '-'], True), (['train', 'dpo1', '123'], True), (['train', 'dpo3', '-'], False),
    (['train', 'dpo1', 'x1'], False), (['merge', 'dpo2'], True), (['merge'], False), (['gen', '8b'], True),
    (['gen', 'base'], False), (['plan-e2'], True), (['score'], True), (['score', 'x'], False), (['other'], False)])
def test_sbatch_argument_checks(tmp_path, args, ok):
    run = subprocess.run(['bash', str(SCRIPTS / 'isambard_014.sbatch'), str(CODE), *args],
                         env=T.stub_sbatch(tmp_path), capture_output=True, text=True)
    assert (run.returncode, 'Slurm allocation required' in run.stderr) == ((1, True) if ok else (2, False))


def submit_project(tmp_path):
    project = tmp_path / 'project'
    run = project / 'runs/unknownsize-20261013'
    for path in ('eval-data-v1/MANIFEST.json', 'dev-select-v1/MANIFEST.json', 'geometry-v5/MANIFEST.json',
                 'dev-eval-010/selection.json'):
        (run / path).parent.mkdir(parents=True, exist_ok=True)
        (run / path).write_text('{}')
    (project / 'models/OptiScene-Qwen-8B-Final').mkdir(parents=True)
    (project / 'models/OptiScene-Qwen-8B-Final/config.json').write_text('{}')
    (project / 'envs').mkdir()
    (project / 'envs/optiscene-014.env.json').write_text(json.dumps({'ok': True, 'model': {'files': {'a': 'b'}}}))
    (project / 'data/optiscene-official').mkdir(parents=True)
    for name, _ in oe.DATA_FILES.values():
        (project / 'data/optiscene-official' / name).symlink_to(ORIGINAL / 'dataset' / name)
    plan = project / 'data/optiscene-014-e1'
    plan.mkdir()
    (plan / 'requests.jsonl').write_text('{}\n')
    (plan / 'MANIFEST.json').write_text(json.dumps({'plan': 'e1', 'implementation_sha256': oe._code(), 'files': {
        'requests.jsonl': oe._sha(plan / 'requests.jsonl')}}))
    for py in ('env/bin/python', 'envs/optiscene-014/bin/python', 'vllm/bin/python'):
        (project / py).parent.mkdir(parents=True, exist_ok=True)
        (project / py).write_text(f'#!/bin/sh\nexec {sys.executable} "$@"\n')
        (project / py).chmod(0o755)
    script = tmp_path / 'submit_014.sh'
    text = (SCRIPTS / 'submit_014.sh').read_text()
    for line, new in (('PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill\n', f'PROJECT_ROOT={project}\n'),
                      ('VLLM_PY=/home/u6uv/shanliantian.u6uv/miniforge3/envs/vllm/bin/python\n',
                       f'VLLM_PY={project}/vllm/bin/python\n')):
        assert text.count(line) == 1
        text = text.replace(line, new)
    script.write_text(text)
    return project, script


@pytest.mark.skipif(not (ORIGINAL / 'dataset/sft_prompts.json').is_file(), reason='released dataset not present')
def test_submit_chain_holds_releases_and_refuses(tmp_path):
    project, script = submit_project(tmp_path)
    env = {**T.stub_sbatch(tmp_path), 'USER': 'me'}
    (tmp_path / 'squeue').write_text('#!/bin/sh\ntest "$*" = "-h -u me -o %i %j" || exit 3\nprintf "%s" "$SQUEUE_OUT"\n')
    (tmp_path / 'squeue').chmod(0o755)
    run = subprocess.run(['bash', str(script), str(CODE)], env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    log = [x.replace(f'{SCRIPTS}/isambard_014.sbatch {CODE} ', 'B ') for x in (tmp_path / 'log').read_text().splitlines()]
    kill = '--parsable --kill-on-invalid-dep=yes --dependency='
    gpu = '--gpus-per-node=1 --cpus-per-task=16 --mem=110G'
    cpu = '--cpus-per-task=16 --mem=64G'
    assert log == [
        f'--parsable --hold --job-name=optiscene-014-plan-e2 {cpu} --time=02:00:00 B plan-e2',
        f'--parsable --hold --job-name=optiscene-014-sft {gpu} --time=06:00:00 B train sft -',
        f'{kill}afterok:102 --job-name=optiscene-014-merge-sft {cpu} --time=02:00:00 B merge sft',
        f'{kill}afterok:103:101 --job-name=optiscene-014-gen-sft {gpu} --time=02:00:00 B gen sft',
        f'{kill}afterok:103 --job-name=optiscene-014-dpo1-s1 {gpu} --time=24:00:00 B train dpo1 -',
        f'{kill}afterany:105 --job-name=optiscene-014-dpo1-s2 {gpu} --time=24:00:00 B train dpo1 105',
        f'{kill}afterany:106 --job-name=optiscene-014-merge-dpo1 {cpu} --time=02:00:00 B merge dpo1',
        f'{kill}afterok:107:101 --job-name=optiscene-014-gen-dpo1 {gpu} --time=02:00:00 B gen dpo1',
        f'{kill}afterok:107 --job-name=optiscene-014-dpo2-s1 {gpu} --time=24:00:00 B train dpo2 -',
        f'{kill}afterany:109 --job-name=optiscene-014-dpo2-s2 {gpu} --time=24:00:00 B train dpo2 109',
        f'{kill}afterany:110 --job-name=optiscene-014-merge-dpo2 {cpu} --time=02:00:00 B merge dpo2',
        f'{kill}afterok:111:101 --job-name=optiscene-014-gen-dpo2 {gpu} --time=02:00:00 B gen dpo2',
        f'{kill}afterok:101 --job-name=optiscene-014-gen-8b {gpu} --time=02:00:00 B gen 8b',
        f'{kill}afterok:104:108:112:113 --job-name=optiscene-014-score {cpu} --time=04:00:00 B score']
    assert (tmp_path / 'log.scontrol').read_text() == 'scontrol release 101,102\n'
    hours = sum(int(x.split('--time=')[1][:2]) for x in log if '--gpus-per-node=1' in x)
    assert hours == 110  # The reserved GPU cap.
    failed = tmp_path / 'failed'
    failed.mkdir()
    (failed / 'squeue').write_bytes((tmp_path / 'squeue').read_bytes())
    (failed / 'squeue').chmod(0o755)
    bad = subprocess.run(['bash', str(script), str(CODE)], env={**T.stub_sbatch(failed, fail_on=5), 'USER': 'me'},
                         capture_output=True, text=True)
    assert bad.returncode == 1 and "sbatch failed for 'dpo1-s1'" in bad.stderr and 'scancel 101 102 103 104' in bad.stderr
    assert not (failed / 'log.scontrol').exists()
    queued = subprocess.run(['bash', str(script), str(CODE)], env={**env, 'SQUEUE_OUT': '9001 optiscene-014-sft\n'},
                            capture_output=True, text=True)
    assert queued.returncode == 1 and 'scancel 9001' in queued.stderr
    (project / 'runs/unknownsize-20261013/optiscene-014').mkdir()
    exists = subprocess.run(['bash', str(script), str(CODE)], env=env, capture_output=True, text=True)
    assert exists.returncode == 1 and 'optiscene-014 exists' in exists.stderr
    (project / 'runs/unknownsize-20261013/optiscene-014').rmdir()
    (project / 'envs/optiscene-014.env.json').write_text(json.dumps({'ok': False}))
    refused = subprocess.run(['bash', str(script), str(CODE)], env=env, capture_output=True, text=True)
    assert refused.returncode == 1 and 'nothing submitted' in refused.stderr
    (project / 'envs/optiscene-014.env.json').write_text(json.dumps({'ok': True, 'model': {'files': {'a': 'b'}}}))
    manifest = json.loads((project / 'data/optiscene-014-e1/MANIFEST.json').read_text())
    manifest['implementation_sha256']['optiscene_eval.py'] = '0' * 64
    (project / 'data/optiscene-014-e1/MANIFEST.json').write_text(json.dumps(manifest))
    stale = subprocess.run(['bash', str(script), str(CODE)], env=env, capture_output=True, text=True)
    assert stale.returncode == 1 and 'built by other code' in stale.stderr
    (project / 'vllm/bin/python').chmod(0o644)
    novllm = subprocess.run(['bash', str(script), str(CODE)], env=env, capture_output=True, text=True)
    assert novllm.returncode == 1 and 'no executable' in novllm.stderr
    assert len((tmp_path / 'log').read_text().splitlines()) == 14  # Nothing new was submitted.
