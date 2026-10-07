"""Lab 2 Parts III/IV: same Llama workload, with an NVTX "Annotation" range around layer 0's MLP.

Equivalent to adding range_push/range_pop inside LlamaMLP.forward in modeling_llama.py,
but done as a patch at runtime so the installed transformers source stays stock.
"""
import torch
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers.models.llama import modeling_llama

_orig_mlp_forward = modeling_llama.LlamaMLP.forward


def annotated_mlp_forward(self, x):
    if not getattr(self, "nvtx_annotate", False):
        return _orig_mlp_forward(self, x)
    torch.cuda.nvtx.range_push("Annotation")
    out = _orig_mlp_forward(self, x)
    torch.cuda.nvtx.range_pop()
    return out


modeling_llama.LlamaMLP.forward = annotated_mlp_forward

model_id = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
model = LlamaForCausalLM.from_pretrained(model_id).to("cuda")
model.model.layers[0].mlp.nvtx_annotate = True  # profile only layer 0's MLP

input_text = "Hello, how are you?"
inputs = tokenizer(input_text, return_tensors="pt", padding=True).to("cuda")

# Warmup
output_sequences = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=50)
# Output
output_sequences = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=50)
torch.cuda.synchronize()
