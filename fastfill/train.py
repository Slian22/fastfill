"""LoRA SFT for FastFill v1 (adapted from scripts/sft_train.py).

Differences from the OptiScene script: pre-split files instead of a random split, loss on the
assistant answer only, over-length samples dropped (never truncated mid-JSON), EOS = <|im_end|>
supervised, Qwen3 thinking disabled in the chat template, final adapter saved.

Effective batch = bs x grad_accum x n_gpus; keep it at 32 when changing the GPU count:
  1 GPU:   CUDA_VISIBLE_DEVICES=0 python -m fastfill.train --model Qwen/Qwen3-8B --data DATA --out outputs/ff-v1 --bs 4 --grad_accum 8
  8 GPUs:  torchrun --nproc_per_node 8 -m fastfill.train --model Qwen/Qwen3-8B --data DATA --out outputs/ff-v1 --bs 4 --grad_accum 1
Resume after an interruption: add --resume (continues from the last checkpoint in --out).
"""
import argparse
import json
import math
import os

import torch
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer,
                          TrainingArguments, set_seed)

from fastfill.scene import prompt_text


def end_token(tok):
    return "<|im_end|>" if "<|im_end|>" in tok.get_vocab() else tok.eos_token


def encode(tok, rows, max_len, chunk=2000):
    """Tokenize (prompt, answer) pairs in batches (per-row tokenizer calls are ~300x slower); labels mask the
    prompt; rows longer than max_len are dropped, never truncated (a cut JSON answer teaches broken output)."""
    out, dropped, end = [], 0, end_token(tok)
    for i in range(0, len(rows), chunk):
        part = rows[i:i + chunk]
        ps = tok([prompt_text(tok, m[:-1]) for m in part], add_special_tokens=False)["input_ids"]
        ans = tok([m[-1]["content"] + end for m in part], add_special_tokens=False)["input_ids"]
        for p, a in zip(ps, ans):
            if len(p) + len(a) > max_len:
                dropped += 1
            else:
                out.append({"input_ids": p + a, "attention_mask": [1] * (len(p) + len(a)), "labels": [-100] * len(p) + a})
    return out, dropped


def load(tok, path, max_len):
    rows, dropped = encode(tok, [json.loads(line)["messages"] for line in open(path)], max_len)
    print(f"{path}: {len(rows)} samples, dropped {dropped} longer than {max_len} tokens", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or local path, e.g. Qwen/Qwen3-8B")
    ap.add_argument("--data", required=True, help="dir with train.jsonl / dev.jsonl from fastfill.build")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--bs", type=int, default=4, help="per-device batch size")
    ap.add_argument("--grad_accum", type=int, default=8, help="per GPU; divide by the GPU count to keep batch 32")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--full_ft", action="store_true", help="full fine-tuning instead of LoRA (use with --deepspeed)")
    ap.add_argument("--deepspeed", default=None)
    ap.add_argument("--attn", default="sdpa", help="sdpa | flash_attention_2")
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--report_to", default="none")
    ap.add_argument("--dry_run", action="store_true", help="tokenize and print length stats only (no GPU)")
    ap.add_argument("--resume", action="store_true", help="continue from the last checkpoint in --out")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    train = load(tok, f"{a.data}/train.jsonl", a.max_len)
    dev = load(tok, f"{a.data}/dev.jsonl", a.max_len)
    if a.dry_run:
        lens = sorted(len(r["input_ids"]) for r in train)
        ans = sorted(sum(l != -100 for l in r["labels"]) for r in train)
        q = lambda v, f: v[min(int(f * len(v)), len(v) - 1)]
        print(f"train tokens: total {sum(lens)}  p50 {q(lens, .5)}  p99 {q(lens, .99)}  max {lens[-1]}  "
              f"answer p50 {q(ans, .5)}")
        print(tok.decode(train[0]["input_ids"]))
        print("supervised part:", repr(tok.decode([t for t in train[0]["labels"] if t != -100])))
        return

    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, attn_implementation=a.attn)
    model.config.use_cache = False
    set_seed(42)                            # before get_peft_model: same LoRA init every run (v1.0 vs v1.1 pairing)
    if not a.full_ft:
        model.enable_input_require_grads()  # needed for gradient checkpointing with frozen embeddings
        model = get_peft_model(model, LoraConfig(
            task_type="CAUSAL_LM", r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        model.print_trainable_parameters()

    # 3% warmup as an integer step count: transformers 4.x rejects fractional warmup_steps, >=5.15 has no warmup_ratio
    # one process that sees several GPUs runs DataParallel: its batch is bs x n_gpu, like DDP's
    world = int(os.environ.get("WORLD_SIZE", 0)) or max(1, torch.cuda.device_count())
    total_steps = math.ceil(math.ceil(len(train) / (a.bs * world)) / a.grad_accum) * a.epochs
    # batch similar lengths together (~1/3 less padding); the argument was renamed in transformers 5.14
    by_length = ({"group_by_length": True} if "group_by_length" in TrainingArguments.__dataclass_fields__
                 else {"train_sampling_strategy": "group_by_length"})
    args = TrainingArguments(
        output_dir=a.out, per_device_train_batch_size=a.bs, per_device_eval_batch_size=a.bs,
        gradient_accumulation_steps=a.grad_accum, learning_rate=a.lr, num_train_epochs=a.epochs, **by_length,
        lr_scheduler_type="cosine", warmup_steps=max(1, int(0.03 * total_steps)), weight_decay=0.0,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10, eval_strategy="steps", eval_steps=a.save_steps, save_strategy="steps",
        save_steps=a.save_steps, save_total_limit=3, remove_unused_columns=False,
        report_to=a.report_to, deepspeed=a.deepspeed, ddp_find_unused_parameters=False)
    trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=dev,
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
    trainer.train(resume_from_checkpoint=True if a.resume else None)
    trainer.save_model(f"{a.out}/final")
    tok.save_pretrained(f"{a.out}/final")


if __name__ == "__main__":
    main()
