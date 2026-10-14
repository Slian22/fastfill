"""Amendment 010: RPO term and per-epoch adapter saves in train_dpo (rpo_alpha 0 = unchanged DPO)."""
import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from fastfill.v1_1 import train_dpo


def test_rpo_alpha_zero_is_plain_dpo_and_alpha_adds_chosen_mean_token_nll():
    pol_c, pol_r = torch.tensor([-6.0, -2.0], requires_grad=True), torch.tensor([-5.0, -4.0])
    ref_c, ref_r, tokens = torch.tensor([-5.0, -3.0]), torch.tensor([-5.0, -5.0]), torch.tensor([3, 2])
    plain = -F.logsigmoid(0.1 * ((pol_c - ref_c) - (pol_r - ref_r))).mean()
    loss, dpo, nll, diff = train_dpo.dpo_loss(pol_c, pol_r, ref_c, ref_r, 0.1, 0.0, tokens)
    assert torch.equal(loss, plain) and torch.equal(dpo, plain)
    assert math.isclose(nll.item(), (6 / 3 + 2 / 2) / 2)  # Per-sequence mean token NLL, then batch mean.
    assert torch.equal(diff, torch.tensor([-1.0, 0.0]))
    rpo, _, _, _ = train_dpo.dpo_loss(pol_c, pol_r, ref_c, ref_r, 0.1, 1.0, tokens)
    assert math.isclose(rpo.item(), plain.item() + 1.5, rel_tol=1e-6)
    rpo.backward()  # The NLL term pushes the chosen log-probability up: d/d pol_c includes -1/(2 * tokens).
    grad = pol_c.grad.clone()
    pol_c.grad = None
    plain.backward()
    assert torch.allclose(grad - pol_c.grad, -1 / (2 * tokens.float()))
    assert train_dpo.dpo_loss(pol_c, pol_r, ref_c, ref_r, 0.1)[2] is None
    with pytest.raises(ValueError, match='chosen token counts'):
        train_dpo.dpo_loss(pol_c, pol_r, ref_c, ref_r, 0.1, 1.0)


class Saver:
    def __init__(self, name):
        self.name = name

    def save_pretrained(self, path, **kwargs):
        (Path(path) / self.name).write_text(self.name)


def test_epoch_adapter_is_published_by_rename_and_a_resumed_epoch_replaces_it(tmp_path):
    target = tmp_path / 'epoch-01'
    train_dpo.save_epoch_adapter(target, Saver('adapter_model.safetensors'), Saver('tokenizer.json'), {'epoch': 1, 'step': 5})
    assert sorted(p.name for p in target.iterdir()) == ['adapter_model.safetensors', 'epoch.json', 'tokenizer.json']
    assert json.loads((target / 'epoch.json').read_text()) == {'epoch': 1, 'step': 5}
    (target / 'stale').write_text('x')
    train_dpo.save_epoch_adapter(target, Saver('adapter_model.safetensors'), Saver('tokenizer.json'), {'epoch': 1, 'step': 5})
    assert not (target / 'stale').exists() and [p.name for p in tmp_path.iterdir()] == ['epoch-01']

