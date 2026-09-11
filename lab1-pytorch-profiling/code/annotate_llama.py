"""Insert CS2470Profile_* record_function annotations into HF Llama (transformers 4.57).

Always re-patches from a pristine backup, so it is safe to run repeatedly.
Usage: python annotate_llama.py            # apply annotations
       python annotate_llama.py --restore  # restore original file
"""
import os
import re
import shutil
import sys

import transformers.models.llama.modeling_llama as m

path = m.__file__
bak = path + ".bak"
if not os.path.exists(bak):
    shutil.copy(path, bak)
if "--restore" in sys.argv:
    shutil.copy(bak, path)
    print("restored", path)
    sys.exit(0)

src = open(bak).read()


def with_line(indent, label):
    return f'{indent}with torch.profiler.record_function("CS2470Profile_{label}"):'


def wrap(snippet, label):
    """Wrap an exact code snippet in a `with record_function(...)` block."""
    global src
    n = src.count(snippet)
    assert n == 1, f"{label}: expected 1 match, found {n}"
    lines = snippet.split("\n")
    ind = " " * (len(lines[0]) - len(lines[0].lstrip()))
    body = "\n".join(("    " + l) if l.strip() else l for l in lines)
    src = src.replace(snippet, with_line(ind, label) + "\n" + body)


def wrap_forward(cls, label):
    """Wrap the entire body of cls.forward() in a `with record_function(...)` block."""
    global src
    lines = src.split("\n")
    ci = next(i for i, l in enumerate(lines) if l.startswith(f"class {cls}("))
    di = next(i for i in range(ci, len(lines)) if lines[i].startswith("    def forward("))
    depth = 0
    for si in range(di, len(lines)):  # find the end of the (possibly multi-line) signature
        depth += lines[si].count("(") - lines[si].count(")")
        if depth == 0 and lines[si].rstrip().endswith(":"):
            break
    ei = si + 1
    while ei < len(lines) and (not lines[ei].strip() or lines[ei].startswith("        ")):
        ei += 1
    while not lines[ei - 1].strip():  # leave trailing blank lines outside the block
        ei -= 1
    body = [("    " + l) if l.strip() else l for l in lines[si + 1:ei]]
    lines[si + 1:ei] = [with_line("        ", label)] + body
    src = "\n".join(lines)


# 1) Fine-grained annotations inside attention (applied first, on the original indentation)
wrap("        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)", "QProj")
wrap("        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)", "KProj")
wrap("        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)", "VProj")
wrap("        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)", "ApplyRotary")
wrap("            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)", "KVCacheUpdate")
sdpa = re.search(r"        attn_output, attn_weights = attention_interface\(\n.*?\n        \)", src, re.S).group(0)
wrap(sdpa, "SDPA")
wrap("        attn_output = self.o_proj(attn_output)", "OProj")

# 2) Model-level annotations
wrap("            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)", "Embedding")
wrap("        logits = self.lm_head(hidden_states[:, slice_indices, :])", "LMHead")

# 3) Whole-module annotations (wrap each forward body)
for cls, label in [
    ("LlamaRMSNorm", "RMSNorm"),
    ("LlamaRotaryEmbedding", "RotaryEmbedding"),
    ("LlamaMLP", "MLP"),
    ("LlamaAttention", "LlamaAttention"),
    ("LlamaDecoderLayer", "DecoderLayer"),
]:
    wrap_forward(cls, label)

compile(src, path, "exec")
open(path, "w").write(src)
print("annotated", path, "-", src.count("CS2470Profile_"), "annotations")
