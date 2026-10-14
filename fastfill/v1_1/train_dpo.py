"""Pure-PyTorch + Transformers + PEFT Distributed DPO Trainer for FastFill v1.1.

Why this trainer instead of `trl.DPOTrainer` from old OptiScene (`scripts/dpo_train.py`):
1. Old `scripts/dpo_train.py` had a random 80/20 row split leak, 3200-token silent truncation, and required a duplicate
   8B `ref_model` or `trl` version-specific APIs (`transformers 5.14` vs `trl`).
2. Here we load the merged SFT (or Stage-1 DPO) model, attach LoRA (`r=32, alpha=64`), and compute exact reference
   sequence log-probabilities `log pi_ref(y|x)` inside `with model.disable_adapter(), torch.no_grad():` — zero extra
   VRAM for a second 8B model and zero `trl` dependency!
3. Prompt tokens are strictly masked (`labels = -100` for all `system` + `user` tokens), so DPO log-ratios are summed
   exclusively over the `assistant` JSON tokens.
4. `--init_adapter PATH` (amendment 013) continues a trained stage adapter instead of a new LoRA: PATH is loaded twice
   on `--model` (its own training base), as the trainable policy (PEFT "default", initial weights = PATH) and as a
   frozen "reference" (= PATH, dropout off) that replaces the disabled-adapter reference. Only the policy is saved, in
   the plain adapter layout, so the result is again "training base + one FP32 adapter". Without the flag nothing changes.
5. `--sft_only` (amendment 016 RFT ablation): the loss is the chosen response's mean token NLL alone (RPO's NLL term,
   no DPO term); neither the reference nor the rejected response is run. Works with `--init_adapter`. Without the
   flag nothing changes.
6. Amendment 021 (the 27B student), each optional; without them nothing changes:
   `--sft_token_mean` (with --sft_only): loss = summed answer-token NLL of the optimizer window over all ranks /
   that window's supervised-token count (all-reduced), instead of the mean of per-sequence means.
   `--lora_regex R`: LoRA targets = the modules whose full name matches R (PEFT fullmatch), e.g. the language model's
   linear layers only; after the first optimizer step every trainable parameter must have a gradient.
   `--answer_logits`: the LM head runs only from the position before the first supervised token (logits_to_keep).
   `--check_render`: tok.eos_token must be the inference stop token <|im_end|> and every row's training prompt
   render must be infer.prompt_ids (thinking off) token for token and a prefix of its sequences (raise, never skip).
   `--base_receipt`: rank 0 rehashes --model against its pinned receipt (a021_train.base_receipt: 27B or 8B).
   `--image_text_model`: load --model with AutoModelForImageTextToText (a multimodal checkpoint's own class and
   module names, so adapter keys match the checkpoint; inputs stay text only).
   `--perf_out DIR`: per rank, per optimizer step wall seconds and input tokens plus peak CUDA memory (the smoke).
   `--warmup_ratio F`: cosine warmup steps = max(1, int(F x total steps)) instead of 5% (P's SFT used 3%).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from peft import LoraConfig, get_peft_model
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup, set_seed

from fastfill.scene import prompt_text

POLICY, REFERENCE = "default", "reference"  # PEFT saves the "default" adapter in the plain (root) layout.
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
INIT_FIELDS = ("peft_type", "task_type", "r", "lora_alpha", "lora_dropout", "bias", "use_rslora", "use_dora",
               "modules_to_save", "layers_to_transform", "rank_pattern", "alpha_pattern")
STOP = "<|im_end|>"  # The inference stop token (infer / sample_vllm stop on it only).
OPTIONAL = ("sft_token_mean", "lora_regex", "answer_logits", "check_render", "base_receipt", "image_text_model",
            "perf_out", "warmup_ratio")  # Amendment 021: absent from run_manifest unless set.


def lora_config(r: int, alpha: int, regex: str | None = None) -> LoraConfig:
    return LoraConfig(task_type="CAUSAL_LM", r=r, lora_alpha=alpha, lora_dropout=0.05,
                      target_modules=TARGET_MODULES if regex is None else regex)


def init_policy(base: Any, path: Path, expected: LoraConfig) -> Any:
    """--init_adapter: PATH as the trainable policy ("default") and as the frozen reference, on BASE.

    PEFT autocasts both to FP32 on a BF16 base (as get_peft_model does); the reference's dropout is Identity, so its
    log-probabilities are the exact policy PATH on BASE also while the model is in train mode."""
    from peft import PeftConfig, PeftModel
    from peft.tuners.lora import LoraLayer

    found = PeftConfig.from_pretrained(str(path))
    if (any(getattr(found, k, None) != getattr(expected, k) for k in INIT_FIELDS)
            or set(found.target_modules) != set(expected.target_modules)):
        raise ValueError(f"--init_adapter LoRA config differs from this run's (r, alpha, dropout, targets): {path}")
    # torch_device='cpu': safetensors maps a bare 'cuda' to cuda:0, so every DDP rank would open GPU 0 and ranks 1-3 fail
    # with 'device busy' on nodes in exclusive-process mode (013/018 trains); the model moves to its rank's GPU later.
    model = PeftModel.from_pretrained(base, str(path), adapter_name=POLICY, is_trainable=True, torch_device="cpu")
    model.load_adapter(str(path), adapter_name=REFERENCE, is_trainable=False, torch_device="cpu")
    for module in model.modules():
        if isinstance(module, LoraLayer):
            module.lora_dropout[REFERENCE] = torch.nn.Identity()
    return model


@contextlib.contextmanager
def _reference_adapter(model: Any):
    model.set_adapter(REFERENCE)
    try:
        yield
    finally:
        model.set_adapter(POLICY)  # Also restores requires_grad: policy trainable, reference frozen.


def reference_policy(model: Any, init_adapter: Path | None) -> Any:
    """Context for the frozen reference: the training base alone (adapter disabled), or base + the --init_adapter copy."""
    return model.disable_adapter() if init_adapter is None else _reference_adapter(model)


def check_render(tok: Any, lines: list[dict], p_ids_list: list, c_ids_list: list, r_ids_list: list) -> None:
    """--check_render: the appended EOS is the inference stop token, and each training prompt render is infer.prompt_ids
    (chat template, thinking off) token for token and a prefix of both sequences, which end on that token."""
    from fastfill.comparison.infer import prompt_ids

    if tok.eos_token != STOP:
        raise ValueError(f"training EOS {tok.eos_token!r} is not the inference stop token {STOP}")
    stop = tok([STOP], add_special_tokens=False)["input_ids"][0]
    if len(stop) != 1:
        raise ValueError(f"{STOP} is not a single token")
    for row, p_ids, c_ids, r_ids in zip(lines, p_ids_list, c_ids_list, r_ids_list):
        if (prompt_ids(tok, {"messages": row["prompt"]}) != p_ids
                or any(ids[:len(p_ids)] != p_ids or ids[-1] != stop[0] for ids in (c_ids, r_ids))):
            raise ValueError(f"training render differs from infer.prompt_ids or is not a prefix: {row.get('uid')}")


class DPOPairDataset(Dataset):
    def __init__(self, tok: Any, path: Path, max_len: int = 8192, render_check: bool = False) -> None:
        self.rows: list[dict[str, list[int]]] = []
        skipped = 0
        lines = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        prompts = [prompt_text(tok, r["prompt"]) for r in lines]
        chosen_full = [p + r["chosen"] + tok.eos_token for p, r in zip(prompts, lines)]
        rejected_full = [p + r["rejected"] + tok.eos_token for p, r in zip(prompts, lines)]

        p_ids_list = tok(prompts, add_special_tokens=False)["input_ids"]
        c_ids_list = tok(chosen_full, add_special_tokens=False)["input_ids"]
        r_ids_list = tok(rejected_full, add_special_tokens=False)["input_ids"]
        if render_check:
            check_render(tok, lines, p_ids_list, c_ids_list, r_ids_list)

        for p_ids, c_ids, r_ids in zip(p_ids_list, c_ids_list, r_ids_list):
            plen = len(p_ids)
            if len(c_ids) > max_len or len(r_ids) > max_len or plen >= len(c_ids) or plen >= len(r_ids):
                skipped += 1
                continue
            if c_ids[:plen] != p_ids or r_ids[:plen] != p_ids:
                skipped += 1
                continue
            self.rows.append(
                {
                    "c_input_ids": c_ids,
                    "c_labels": [-100] * plen + c_ids[plen:],
                    "r_input_ids": r_ids,
                    "r_labels": [-100] * plen + r_ids[plen:],
                }
            )
        if int(os.environ.get("RANK", 0)) == 0:
            print(f"[DPOPairDataset] Loaded {len(self.rows)} pairs from {path} (skipped {skipped} > {max_len})", flush=True)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.rows[idx]


def collate_dpo(batch: list[dict[str, list[int]]], pad_id: int) -> dict[str, torch.Tensor]:
    def pad_seq(seqs: list[list[int]], pad_val: int) -> torch.Tensor:
        max_l = max(len(s) for s in seqs)
        return torch.tensor([s + [pad_val] * (max_l - len(s)) for s in seqs], dtype=torch.long)

    c_ids = [b["c_input_ids"] for b in batch]
    c_lab = [b["c_labels"] for b in batch]
    r_ids = [b["r_input_ids"] for b in batch]
    r_lab = [b["r_labels"] for b in batch]
    return {
        "c_input_ids": pad_seq(c_ids, pad_id),
        "c_attention_mask": pad_seq([[1] * len(s) for s in c_ids], 0),
        "c_labels": pad_seq(c_lab, -100),
        "r_input_ids": pad_seq(r_ids, pad_id),
        "r_attention_mask": pad_seq([[1] * len(s) for s in r_ids], 0),
        "r_labels": pad_seq(r_lab, -100),
    }


def sequence_logps(model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor,
                   answer_only: bool = False) -> torch.Tensor:
    """Compute summed token log-probabilities over non-masked (`labels != -100`) completion tokens in FP32.

    answer_only (--answer_logits): logits only for the positions that predict a supervised token of some row, i.e.
    from the position before the batch's first supervised label (logits_to_keep); the sum is the same."""
    if answer_only:
        first = max(1, int((labels != -100).int().argmax(dim=1).min()))
        keep = input_ids.shape[1] - first + 1
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
                       logits_to_keep=keep).logits[:, :-1, :]
        target = labels[:, first:]
    else:
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits[:, :-1, :]
        target = labels[:, 1:]
    bsz, seq_len, vocab = logits.shape
    total = torch.zeros(bsz, dtype=torch.float32, device=logits.device)
    chunk_t = 1024
    for s in range(0, seq_len, chunk_t):
        e = min(seq_len, s + chunk_t)
        sub_logits = logits[:, s:e, :].float()
        sub_target = target[:, s:e]
        sub_loss = F.cross_entropy(
            sub_logits.reshape(-1, vocab),
            sub_target.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(bsz, e - s)
        total = total - sub_loss.sum(dim=-1)
    return total


def dpo_loss(pol_c, pol_r, ref_c, ref_r, beta, rpo_alpha=0.0, chosen_tokens=None):
    """Sigmoid DPO; with rpo_alpha > 0 (RPO) plus alpha x the chosen response's mean token NLL under the policy.

    Returns (loss, dpo term, chosen mean token NLL or None, logits_diff); rpo_alpha 0 is plain DPO, bit for bit.
    """
    logits_diff = (pol_c - ref_c) - (pol_r - ref_r)
    dpo = -F.logsigmoid(beta * logits_diff).mean()
    if chosen_tokens is None:
        if rpo_alpha:
            raise ValueError("RPO needs the chosen token counts")
        return dpo, dpo, None, logits_diff
    nll = (-pol_c / chosen_tokens).mean()
    return (dpo + rpo_alpha * nll if rpo_alpha else dpo), dpo, nll, logits_diff


def sft_loss(pol_c, chosen_tokens):
    """--sft_only: the chosen response's mean token NLL per sequence, then the batch mean (dpo_loss's NLL term)."""
    return (-pol_c / chosen_tokens).mean()


def token_mean_loss(pol_c, window_tokens, world_size):
    """--sft_token_mean: this micro-batch's summed answer NLL x world_size / the supervised tokens of the whole optimizer
    window on all ranks (all-reduced). DDP averages the rank gradients, so the update is the gradient of
    (sum of answer-token NLL over the window on every rank) / (their supervised-token count)."""
    return -pol_c.sum() * world_size / window_tokens


def window_tokens(batch_tokens, grad_accum):
    """Per micro-batch of one rank's epoch: the supervised tokens of its whole accumulation window on this rank."""
    return [sum(batch_tokens[b - b % grad_accum:b - b % grad_accum + grad_accum]) for b in range(len(batch_tokens))]


def missing_grads(model):
    """--lora_regex: trainable parameters that got no gradient (checked after the first optimizer step)."""
    return [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]


def save_epoch_adapter(target: Path, raw_peft: Any, tok: Any, state: dict[str, int]) -> None:
    """Amendment 010: the adapter at an epoch end, published by rename (a resumed epoch replaces it)."""
    import shutil

    tmp = target.with_name(f"{target.name}.tmp.{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    raw_peft.save_pretrained(tmp, selected_adapters=[POLICY])
    tok.save_pretrained(tmp)
    (tmp / "epoch.json").write_text(json.dumps(state) + "\n", encoding="utf-8")
    if target.exists():
        shutil.rmtree(target)
    os.replace(tmp, target)


def accumulation_window_size(batch_index: int, batches: int, grad_accum: int) -> int:
    """Normalize the final partial window by its actual microbatch count.

    DistributedSampler gives each rank the same batch count, including padding,
    so every rank uses the same divisor before DDP averages its gradients.
    """
    if grad_accum < 1 or not 0 <= batch_index < batches:
        raise ValueError("Invalid gradient accumulation window")
    window_start = (batch_index // grad_accum) * grad_accum
    return min(grad_accum, batches - window_start)


def backward_finite_loss(loss: torch.Tensor) -> None:
    """Fail before contaminating gradients; torchrun terminates peer ranks."""
    if not torch.isfinite(loss).all().item():
        raise FloatingPointError("Non-finite DPO loss; refusing backward and optimizer update")
    loss.backward()


def _validate_dpo_checkpoint(path: Path) -> None:
    from fastfill.v1_1.evaluation_protocol import _validate_safetensors

    required = ("adapter_config.json", "adapter_model.safetensors", "trainer_state.pt")
    if not path.is_dir() or any(not (path / name).is_file() or (path / name).stat().st_size == 0 for name in required):
        raise ValueError(f"Incomplete DPO checkpoint; refusing a silent restart: {path}")
    try:
        if not isinstance(json.loads((path / "adapter_config.json").read_text()), dict):
            raise ValueError("Adapter config must be an object")
        _validate_safetensors(path / "adapter_model.safetensors")
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid DPO checkpoint; inspect preserved artifacts: {path}") from exc


def recover_dpo_checkpoint(ckpt_dir: Path) -> Path | None:
    """Recover a killed rename transaction; never silently adopt a partial save.

    Only rank zero calls this mutating recovery helper before the DDP barrier.
    Existing checkpoint serialization remains compatible with earlier runs.
    """
    previous = ckpt_dir.with_name(f"{ckpt_dir.name}.previous")
    if ckpt_dir.exists():
        _validate_dpo_checkpoint(ckpt_dir)
        return ckpt_dir
    if previous.exists():
        _validate_dpo_checkpoint(previous)
        os.replace(previous, ckpt_dir)
        return ckpt_dir
    if any(ckpt_dir.parent.glob(f"{ckpt_dir.name}.tmp.*")):
        raise ValueError(f"Interrupted initial DPO checkpoint; inspect artifacts before restarting: {ckpt_dir}")
    return None


def validate_resume_position(state: dict, *, batches: int, epochs: int, grad_accum: int) -> dict:
    """Require an optimizer boundary with a step count matching this exact loader."""
    if not isinstance(state, dict) or any(type(state.get(k)) is not int for k in ("epoch", "it", "step")):
        raise ValueError("Invalid DPO checkpoint position: missing or non-integer epoch/it/step")
    epoch, iteration, step = (state[k] for k in ("epoch", "it", "step"))
    if not (0 <= epoch < epochs and 0 <= iteration < batches):
        raise ValueError("Invalid DPO checkpoint position: epoch or iteration outside this run")
    boundary = (iteration + 1) % grad_accum == 0 or iteration + 1 == batches
    expected = epoch * math.ceil(batches / grad_accum) + math.ceil((iteration + 1) / grad_accum)
    if not boundary or step != expected:
        raise ValueError("Invalid DPO checkpoint position: step does not match an optimizer boundary")
    return dict(state)


def write_dpo_completion(directory, run_root, admitted_pairs, loader_batches, world_size, epochs, step, total_steps):
    """Bind a fully trained adapter to actual steps and the enclosing input guard."""
    from fastfill.v1_1.evaluation_protocol import file_identity, write_json_atomic

    if step != total_steps or step < 1 or epochs < 1 or loader_batches < 1 or admitted_pairs < 1:
        raise ValueError("Cannot publish incomplete DPO training as a final adapter")
    target, guard = Path(directory), Path(run_root) / "training_guard.json"
    names = ("adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json", "tokenizer.json")
    write_json_atomic(target / "completion.json", {
        "schema_version": 1, "kind": "fastfill_v1.1_dpo_completion",
        "training_guard": file_identity(guard) if guard.exists() else None,
        "admitted_pairs": admitted_pairs, "loader_batches": loader_batches,
        "world_size": world_size, "epochs": epochs,
        "position": {"epoch": epochs - 1, "it": loader_batches - 1, "step": step},
        "expected_total_steps": total_steps,
        "adapter_files": {name: file_identity(target / name) for name in names},
    })


def _save_dpo_ckpt(ckpt_dir: Path, raw_peft: Any, opt: torch.optim.Optimizer, sched: Any, state: dict[str, int]) -> None:
    import shutil

    tmp_dir = ckpt_dir.with_name(f"{ckpt_dir.name}.tmp.{os.getpid()}")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    raw_peft.save_pretrained(tmp_dir, selected_adapters=[POLICY])
    torch.save(
        {
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "state": state,
        },
        tmp_dir / "trainer_state.pt",
    )
    _validate_dpo_checkpoint(tmp_dir)
    previous = ckpt_dir.with_name(f"{ckpt_dir.name}.previous")
    if ckpt_dir.exists():
        _validate_dpo_checkpoint(ckpt_dir)
        if previous.exists():
            shutil.rmtree(previous)
        os.replace(ckpt_dir, previous)
    try:
        os.replace(tmp_dir, ckpt_dir)
    except OSError:
        if not ckpt_dir.exists() and previous.exists():
            os.replace(previous, ckpt_dir)
        raise


def main() -> None:
    import contextlib
    import shutil
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="Base/merged SFT model path")
    ap.add_argument("--pairs", type=Path, required=True, help="DPO pairs JSONL from build_dpo.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=8192)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--save_steps", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rpo_alpha", type=float, default=0.0, help="RPO: + alpha x chosen mean token NLL (0 = plain DPO)")
    ap.add_argument("--save_every_epoch", action="store_true", help="also save the adapter to OUT/epoch-NN at each epoch end")
    ap.add_argument("--init_adapter", type=Path, help="continue this trained adapter of --model; it is also the frozen reference")
    ap.add_argument("--sft_only", action="store_true", help="loss = chosen mean token NLL only (no DPO term, no reference pass)")
    ap.add_argument("--sft_token_mean", action="store_true", help="with --sft_only: window answer NLL sum / its token count, all ranks")
    ap.add_argument("--lora_regex", help="LoRA targets: modules whose full name matches this regex; all trainable params need a grad")
    ap.add_argument("--answer_logits", action="store_true", help="LM head only on answer-predicting positions (logits_to_keep)")
    ap.add_argument("--check_render", action="store_true", help="EOS = <|im_end|>; prompt render = infer.prompt_ids and a prefix")
    ap.add_argument("--base_receipt", action="store_true", help="rank 0 rehashes --model against its pinned base receipt")
    ap.add_argument("--image_text_model", action="store_true", help="load --model with AutoModelForImageTextToText (text inputs)")
    ap.add_argument("--perf_out", type=Path, help="per-rank step wall seconds, input tokens and peak CUDA memory")
    ap.add_argument("--warmup_ratio", type=float, help="cosine warmup share of the total steps (default 0.05)")
    args = ap.parse_args()
    if not (math.isfinite(args.rpo_alpha) and args.rpo_alpha >= 0):
        raise ValueError("--rpo_alpha must be finite and >= 0")
    if args.sft_only and args.rpo_alpha:
        raise ValueError("--sft_only has no DPO term; --rpo_alpha must be 0")
    if args.sft_token_mean and not args.sft_only:
        raise ValueError("--sft_token_mean is the --sft_only loss normalisation")

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    is_ddp = world_size > 1
    if is_ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    if rank == 0:
        receipt = {}
        if args.base_receipt:  # Amendment 021: the pinned 27B (or generic 8B) base, weights rehashed.
            from fastfill.comparison import a021_train
            from fastfill.comparison.infer import _hash
            base = a021_train.base_receipt(args.model)
            receipt = {"base_receipt": {"path": base["path"], "files_sha256": _hash(base["files"])}}
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "run_manifest.json").write_text(
            json.dumps({"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                                 if not (k == "init_adapter" and v is None) and not (k == "sft_only" and not v)
                                 and not (k in OPTIONAL and v in (None, False))}, **receipt},
                       indent=1) + "\n",
            encoding="utf-8",
        )

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = DPOPairDataset(tok, args.pairs, max_len=args.max_len, render_check=args.check_render)
    if len(ds) == 0:
        raise ValueError("No valid DPO pairs remain after tokenization; refusing an empty training run")
    # Also for one process: the epoch order is a function of (seed, epoch), so a resumed epoch replays it.
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
    loader = DataLoader(
        ds,
        batch_size=args.bs,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=lambda b: collate_dpo(b, tok.pad_token_id),
    )

    if args.image_text_model:
        from transformers import AutoModelForImageTextToText as model_class
    else:
        model_class = AutoModelForCausalLM
    base = model_class.from_pretrained(
        args.model,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="sdpa",
    )
    base.config.use_cache = False
    base.enable_input_require_grads()
    config = lora_config(args.lora_r, args.lora_alpha, args.lora_regex)
    peft_model = get_peft_model(base, config) if args.init_adapter is None else init_policy(base, args.init_adapter, config)
    peft_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    peft_model.to(device)
    if rank == 0:
        peft_model.print_trainable_parameters()

    opt = torch.optim.AdamW([p for p in peft_model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    steps_per_epoch = max(1, math.ceil(len(loader) / args.grad_accum))
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup = 0.05 if args.warmup_ratio is None else args.warmup_ratio
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=max(1, int(warmup * total_steps)), num_training_steps=total_steps)

    ckpt_dir = args.out / "ckpt_latest"
    if rank == 0:
        recover_dpo_checkpoint(ckpt_dir)
    if is_ddp:
        dist.barrier()
    resume_epoch = 0
    resume_it = -1
    step = 0
    if ckpt_dir.exists():
        _validate_dpo_checkpoint(ckpt_dir)
        adapter_sd = load_file(str(ckpt_dir / "adapter_model.safetensors"), device=str(device))
        set_peft_model_state_dict(peft_model, adapter_sd)
        t_state = torch.load(ckpt_dir / "trainer_state.pt", map_location=device, weights_only=False)
        st = validate_resume_position(t_state["state"], batches=len(loader), epochs=args.epochs, grad_accum=args.grad_accum)
        opt.load_state_dict(t_state["optimizer"])
        sched.load_state_dict(t_state["scheduler"])
        resume_epoch, resume_it, step = (st[k] for k in ("epoch", "it", "step"))
        if rank == 0:
            print(f"[train_dpo] Resumed from {ckpt_dir} at epoch={resume_epoch + 1}, it={resume_it + 1}, step={step}/{total_steps}", flush=True)

    model = DDP(peft_model, device_ids=[local_rank], find_unused_parameters=False) if is_ddp else peft_model
    raw_peft = model.module if is_ddp else model

    def logps(*tensors):  # --answer_logits; the module-level sequence_logps is looked up per call.
        return sequence_logps(*tensors, answer_only=True) if args.answer_logits else sequence_logps(*tensors)

    if args.sft_token_mean:
        supervised = [sum(v != -100 for v in r["c_labels"][1:]) for r in ds.rows]
    grads_checked = not args.lora_regex
    perf, last = [], None
    model.train()
    opt.zero_grad(set_to_none=True)
    for epoch in range(resume_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        if args.sft_token_mean:  # This rank's epoch order is the sampler's (the loader iterates it once per epoch).
            order = list(sampler)
            counts = [sum(supervised[i] for i in order[b * args.bs:(b + 1) * args.bs]) for b in range(len(loader))]
            windows = window_tokens(counts, args.grad_accum)
        for it, batch in enumerate(loader):
            if epoch == resume_epoch and it <= resume_it:
                continue
            batch = {k: v.to(device) for k, v in batch.items()}
            sync_step = ((it + 1) % args.grad_accum == 0) or ((it + 1) == len(loader))
            window_size = accumulation_window_size(it, len(loader), args.grad_accum)
            sync_ctx = contextlib.nullcontext if (not is_ddp or sync_step) else model.no_sync
            if args.perf_out and last is None:
                last, window_input = time.monotonic(), 0
            if args.perf_out:
                window_input += int(batch["c_attention_mask"].sum())

            if args.sft_only:  # One forward: the chosen response under the policy; no reference, no rejected.
                with sync_ctx():
                    pol_c = logps(model, batch["c_input_ids"], batch["c_attention_mask"], batch["c_labels"])
                    if args.sft_token_mean:
                        if int((batch["c_labels"][:, 1:] != -100).sum()) != counts[it]:
                            raise ValueError("loader batch differs from the sampler order used for the token counts")
                        tokens = torch.tensor(windows[it], dtype=torch.long, device=device)
                        if is_ddp:
                            dist.all_reduce(tokens)
                        total = token_mean_loss(pol_c, tokens, world_size)
                        shown = (-pol_c.sum() / counts[it]).detach()
                        backward_finite_loss(total)
                    else:
                        total = sft_loss(pol_c, (batch["c_labels"][:, 1:] != -100).sum(dim=-1))
                        shown = total
                        backward_finite_loss(total / window_size)
            else:
                with torch.no_grad(), reference_policy(raw_peft, args.init_adapter):
                    ref_c = logps(raw_peft, batch["c_input_ids"], batch["c_attention_mask"], batch["c_labels"])
                    ref_r = logps(raw_peft, batch["r_input_ids"], batch["r_attention_mask"], batch["r_labels"])

                # Forward on raw_peft under no_sync first for chosen, then sync on rejected at sync_step
                with (model.no_sync() if is_ddp else contextlib.nullcontext()):
                    pol_c = logps(model, batch["c_input_ids"], batch["c_attention_mask"], batch["c_labels"])
                with sync_ctx():
                    pol_r = logps(model, batch["r_input_ids"], batch["r_attention_mask"], batch["r_labels"])
                    chosen_tokens = (batch["c_labels"][:, 1:] != -100).sum(dim=-1)
                    total, dpo_term, nll, logits_diff = dpo_loss(pol_c, pol_r, ref_c, ref_r, args.beta, args.rpo_alpha, chosen_tokens)
                    loss = total / window_size
                    backward_finite_loss(loss)

            if sync_step:
                if not grads_checked:  # --lora_regex: every trainable parameter is on the loss path.
                    missing = missing_grads(model)
                    if missing:
                        raise ValueError(f"{len(missing)} trainable parameters have no gradient after step 1: {missing[:3]}")
                    grads_checked = True
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if args.perf_out:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize(device)
                    now = time.monotonic()
                    perf.append({"step": step, "seconds": now - last, "input_tokens": window_input})
                    last, window_input = now, 0
                if rank == 0 and args.sft_only and (step == 1 or step % 10 == 0 or step == total_steps):
                    print(json.dumps({"epoch": epoch + 1, "step": step, "total_steps": total_steps,
                                      "chosen_nll": round(float(shown.item()), 4), "loss": round(float(shown.item()), 4)}),
                          flush=True)
                elif rank == 0 and (step == 1 or step % 10 == 0 or step == total_steps):
                    acc = float((logits_diff > 0).float().mean().item())
                    margin = float((args.beta * logits_diff).mean().item())
                    print(
                        json.dumps(
                            {
                                "epoch": epoch + 1,
                                "step": step,
                                "total_steps": total_steps,
                                "dpo_loss": round(float(dpo_term.item()), 4),
                                "chosen_nll": round(float(nll.item()), 4),
                                "loss": round(float(total.item()), 4),
                                "reward_acc": round(acc, 4),
                                "reward_margin": round(margin, 4),
                            }
                        ),
                        flush=True,
                    )
                epoch_end = (it + 1) == len(loader)
                save_ckpt = args.save_steps > 0 and (step % args.save_steps == 0 or epoch_end)
                save_epoch = args.save_every_epoch and epoch_end
                if save_ckpt or save_epoch:
                    if is_ddp:
                        dist.barrier()
                    if rank == 0:
                        if save_epoch:  # Before the checkpoint: a resume after it never skips this epoch's adapter.
                            save_epoch_adapter(args.out / f"epoch-{epoch + 1:02d}", raw_peft, tok,
                                               {"epoch": epoch + 1, "step": step})
                        if save_ckpt:
                            _save_dpo_ckpt(ckpt_dir, raw_peft, opt, sched, {"epoch": epoch, "it": it, "step": step})
                    if is_ddp:
                        dist.barrier()

    if args.perf_out:  # Every rank: its steps and its peak CUDA memory.
        cuda = torch.cuda.is_available()
        args.perf_out.mkdir(parents=True, exist_ok=True)
        (args.perf_out / f"perf-rank{rank}.json").write_text(json.dumps({
            "rank": rank, "world_size": world_size, "steps": perf,
            "max_memory_allocated": torch.cuda.max_memory_allocated(device) if cuda else None,
            "max_memory_reserved": torch.cuda.max_memory_reserved(device) if cuda else None,
            "device_total_memory": torch.cuda.get_device_properties(device).total_memory if cuda else None,
            "device_name": torch.cuda.get_device_name(device) if cuda else None}) + "\n", encoding="utf-8")

    if rank == 0:
        final_dir = args.out / "final"
        tmp_final = args.out / f"final.tmp.{os.getpid()}"
        if tmp_final.exists():
            shutil.rmtree(tmp_final)
        tmp_final.mkdir(parents=True, exist_ok=True)
        raw_peft.save_pretrained(tmp_final, selected_adapters=[POLICY])
        tok.save_pretrained(tmp_final)
        write_dpo_completion(tmp_final, args.out, len(ds), len(loader), world_size, args.epochs, step, total_steps)
        if final_dir.exists():
            shutil.rmtree(final_dir)
        os.replace(tmp_final, final_dir)
        print(f"[train_dpo] Saved DPO adapter to {final_dir}", flush=True)

    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
