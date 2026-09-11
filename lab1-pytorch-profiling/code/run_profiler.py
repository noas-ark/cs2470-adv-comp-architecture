"""Profile one generate() call with PyTorch Profiler and export a Chrome trace.

Usage: python run_profiler.py                                   # Llama-3.2-1B -> profile.json
       python run_profiler.py --model google/gemma-2-2b --out profile_gemma.json
"""
import argparse

import torch
from torch.profiler import ProfilerActivity, profile
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="meta-llama/Llama-3.2-1B")
ap.add_argument("--out", default="profile.json")
ap.add_argument("--max-new-tokens", type=int, default=50)
args = ap.parse_args()

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(args.model)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(args.model).to("cuda")
print(f"model={args.model} dtype={model.dtype} attn_implementation={model.config._attn_implementation}")

# Prepare input text
input_text = "Hello, how are you?"
inputs = tokenizer(input_text, return_tensors="pt", padding=True).to("cuda")

# Warmup
output_sequences = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=args.max_new_tokens,
)

# Profile
activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
with profile(activities=activities, record_shapes=True, profile_memory=True,
             with_flops=True, with_stack=False) as prof:
    with torch.profiler.record_function("CS2470Profile_MyCode"):
        with torch.inference_mode():
            output_sequences = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=args.max_new_tokens,
            )
            torch.cuda.synchronize()
    prof.step()

print("==================================")
print(args.out)
print("==================================")
prof.export_chrome_trace(args.out)
