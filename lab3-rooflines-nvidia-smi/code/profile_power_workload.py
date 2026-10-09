"""Part IV + Exercises III/IV: log nvidia-smi power while running an LLM.

Usage:
    python profile_power_workload.py --model meta-llama/Llama-3.2-1B --batch 1   # Part IV
    python profile_power_workload.py --model meta-llama/Llama-3.1-8B --batch 1   # Ex III
    python profile_power_workload.py --model google/gemma-2-2b --batch 100       # Ex IV
Writes results/power_<tag>.csv and results/power_<tag>.json (workload window, TDP).
"""
import argparse
import json
import os
import subprocess
import time
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="meta-llama/Llama-3.2-1B")
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--iters", type=int, default=5)
ap.add_argument("--new_tokens", type=int, default=50)
args = ap.parse_args()

tag = f"{args.model.split('/')[-1]}_bs{args.batch}"
os.makedirs("results", exist_ok=True)
log_file = f"results/power_{tag}.csv"

################ Model Definition ################
tokenizer = AutoTokenizer.from_pretrained(args.model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to("cuda").eval()

input_text = ["Hello, how are you?"] * args.batch
inputs = tokenizer(input_text, return_tensors="pt", padding=True).to("cuda")


def run():
    with torch.inference_mode():
        model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                       max_new_tokens=args.new_tokens, min_new_tokens=args.new_tokens, do_sample=False)


run()  # warmup outside the logged window
torch.cuda.synchronize()

tdp = float(subprocess.check_output(
    "nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits", shell=True).decode().split()[0])

################ Power Logging ################
nvidia_smi_cmd = (f"nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu,utilization.memory,"
                  f"memory.used,memory.total --format=csv,nounits -lms 1 > {log_file}")
nvidia_smi_proc = subprocess.Popen(nvidia_smi_cmd, shell=True, executable='/bin/bash')
print("Started power logging with nvidia-smi...")
time.sleep(5)  # idle baseline

try:
    start = datetime.now()
    for _ in range(args.iters):
        run()
    torch.cuda.synchronize()
    end = datetime.now()
finally:
    print("Stopping power logging...")
    time.sleep(5)
    nvidia_smi_proc.terminate()
    nvidia_smi_proc.wait()
    subprocess.run("pkill -f 'nvidia-smi --query-gpu=timestamp'", shell=True)
    print("Power logging stopped.")

meta = {"model": args.model, "batch": args.batch, "iters": args.iters, "new_tokens": args.new_tokens,
        "start": start.strftime("%Y/%m/%d %H:%M:%S.%f"), "end": end.strftime("%Y/%m/%d %H:%M:%S.%f"),
        "duration_s": (end - start).total_seconds(), "tdp_w": tdp, "csv": log_file}
with open(f"results/power_{tag}.json", "w") as f:
    json.dump(meta, f, indent=2)
print(json.dumps(meta, indent=2))
