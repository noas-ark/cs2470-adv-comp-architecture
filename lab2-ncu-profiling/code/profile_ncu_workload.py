"""Llama-3.2-1B inference workload for Nsight Compute (lab 2, Part I).

Run under ncu, not directly:
    ncu -o profile_ncu_basic -f python profile_ncu_workload.py
"""
import argparse

import torch
from transformers import AutoTokenizer, LlamaForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="meta-llama/Llama-3.2-1B")
ap.add_argument("--max-new-tokens", type=int, default=50)
args = ap.parse_args()

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(args.model)
tokenizer.pad_token = tokenizer.eos_token
model = LlamaForCausalLM.from_pretrained(args.model).to("cuda")

# Prepare input text
input_text = "Hello, how are you?"
inputs = tokenizer(input_text, return_tensors="pt", padding=True).to("cuda")

# Warmup
output_sequences = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=args.max_new_tokens,
)

# Output
output_sequences = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=args.max_new_tokens,
)
torch.cuda.synchronize()
print(tokenizer.decode(output_sequences[0], skip_special_tokens=True))
