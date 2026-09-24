import json

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
 
 
def apply_lora(model_name_or_path, output_path, lora_path):
    print(f"Loading the base model from {model_name_or_path}")
    base = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
 
    print(f"Loading the LoRA adapter from {lora_path}")
 
    lora_model = PeftModel.from_pretrained(
        base,
        lora_path,
        torch_dtype=torch.bfloat16,
    )
 
    print("Applying the LoRA")
    model = lora_model.merge_and_unload()
 
    print(f"Saving the target model to {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    # transformers>=5 writes rope_theta only inside rope_parameters and extra_special_tokens as a list; transformers
    # 4.x (vLLM<=0.19) then silently uses rope_theta=10000 and fails to load the tokenizer. Write both forms.
    cfg_p, tok_p = f"{output_path}/config.json", f"{output_path}/tokenizer_config.json"
    cfg, tc = json.load(open(cfg_p)), json.load(open(tok_p))
    if "rope_theta" in (cfg.get("rope_parameters") or {}):
        cfg.setdefault("rope_theta", cfg["rope_parameters"]["rope_theta"])
    if isinstance(tc.get("extra_special_tokens"), list):
        tc["additional_special_tokens"] = tc.pop("extra_special_tokens")
    json.dump(cfg, open(cfg_p, "w"), indent=2)
    json.dump(tc, open(tok_p, "w"), indent=2, ensure_ascii=False)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    apply_lora(args.base_model_path, args.output_path, args.lora_path)
