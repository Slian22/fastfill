"""Pure-PyTorch + Transformers + PEFT Distributed DPO Trainer for FastFill v1.1.

Why this trainer instead of `trl.DPOTrainer` from old OptiScene (`scripts/dpo_train.py`):
1. Old `scripts/dpo_train.py` had a random 80/20 row split leak, 3200-token silent truncation, and required a duplicate
   8B `ref_model` or `trl` version-specific APIs (`transformers 5.14` vs `trl`).
2. Here we load the merged SFT (or Stage-1 DPO) model, attach LoRA (`r=32, alpha=64`), and compute exact reference
   sequence log-probabilities `log pi_ref(y|x)` inside `with model.disable_adapter(), torch.no_grad():` — zero extra
   VRAM for a second 8B model and zero `trl` dependency!
3. Prompt tokens are strictly masked (`labels = -100` for all `system` + `user` tokens), so DPO log-ratios are summed
   exclusively over the `assistant` JSON tokens.
"""
from __future__ import annotations

import argparse
import json
import math
import os
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


class DPOPairDataset(Dataset):
    def __init__(self, tok: Any, path: Path, max_len: int = 8192) -> None:
        self.rows: list[dict[str, list[int]]] = []
        skipped = 0
        lines = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        prompts = [prompt_text(tok, r["prompt"]) for r in lines]
        chosen_full = [p + r["chosen"] + tok.eos_token for p, r in zip(prompts, lines)]
        rejected_full = [p + r["rejected"] + tok.eos_token for p, r in zip(prompts, lines)]

        p_ids_list = tok(prompts, add_special_tokens=False)["input_ids"]
        c_ids_list = tok(chosen_full, add_special_tokens=False)["input_ids"]
        r_ids_list = tok(rejected_full, add_special_tokens=False)["input_ids"]

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


def sequence_logps(model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute summed token log-probabilities over non-masked (`labels != -100`) completion tokens in FP32."""
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
    raw_peft.save_pretrained(tmp_dir)
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
    args = ap.parse_args()

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
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "run_manifest.json").write_text(
            json.dumps({"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}, indent=1) + "\n",
            encoding="utf-8",
        )

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = DPOPairDataset(tok, args.pairs, max_len=args.max_len)
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

    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="sdpa",
    )
    base.config.use_cache = False
    base.enable_input_require_grads()
    peft_model = get_peft_model(
        base,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    peft_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    peft_model.to(device)
    if rank == 0:
        peft_model.print_trainable_parameters()

    opt = torch.optim.AdamW([p for p in peft_model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    steps_per_epoch = max(1, math.ceil(len(loader) / args.grad_accum))
    total_steps = max(1, steps_per_epoch * args.epochs)
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=max(1, int(0.05 * total_steps)), num_training_steps=total_steps)

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

    model.train()
    opt.zero_grad(set_to_none=True)
    for epoch in range(resume_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for it, batch in enumerate(loader):
            if epoch == resume_epoch and it <= resume_it:
                continue
            batch = {k: v.to(device) for k, v in batch.items()}
            sync_step = ((it + 1) % args.grad_accum == 0) or ((it + 1) == len(loader))
            window_size = accumulation_window_size(it, len(loader), args.grad_accum)
            sync_ctx = contextlib.nullcontext if (not is_ddp or sync_step) else model.no_sync

            with torch.no_grad(), raw_peft.disable_adapter():
                ref_c = sequence_logps(raw_peft, batch["c_input_ids"], batch["c_attention_mask"], batch["c_labels"])
                ref_r = sequence_logps(raw_peft, batch["r_input_ids"], batch["r_attention_mask"], batch["r_labels"])

            # Forward on raw_peft under no_sync first for chosen, then sync on rejected at sync_step
            with (model.no_sync() if is_ddp else contextlib.nullcontext()):
                pol_c = sequence_logps(model, batch["c_input_ids"], batch["c_attention_mask"], batch["c_labels"])
            with sync_ctx():
                pol_r = sequence_logps(model, batch["r_input_ids"], batch["r_attention_mask"], batch["r_labels"])
                logits_diff = (pol_c - ref_c) - (pol_r - ref_r)
                loss = -F.logsigmoid(args.beta * logits_diff).mean() / window_size
                backward_finite_loss(loss)

            if sync_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if rank == 0 and (step == 1 or step % 10 == 0 or step == total_steps):
                    acc = float((logits_diff > 0).float().mean().item())
                    margin = float((args.beta * logits_diff).mean().item())
                    print(
                        json.dumps(
                            {
                                "epoch": epoch + 1,
                                "step": step,
                                "total_steps": total_steps,
                                "dpo_loss": round(float(loss.item() * window_size), 4),
                                "reward_acc": round(acc, 4),
                                "reward_margin": round(margin, 4),
                            }
                        ),
                        flush=True,
                    )
                if args.save_steps > 0 and (step % args.save_steps == 0 or (it + 1) == len(loader)):
                    if is_ddp:
                        dist.barrier()
                    if rank == 0:
                        _save_dpo_ckpt(ckpt_dir, raw_peft, opt, sched, {"epoch": epoch, "it": it, "step": step})
                    if is_ddp:
                        dist.barrier()

    if rank == 0:
        final_dir = args.out / "final"
        tmp_final = args.out / f"final.tmp.{os.getpid()}"
        if tmp_final.exists():
            shutil.rmtree(tmp_final)
        tmp_final.mkdir(parents=True, exist_ok=True)
        raw_peft.save_pretrained(tmp_final)
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
