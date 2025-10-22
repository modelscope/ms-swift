import os
import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ------------------------------
# Configuration
# ------------------------------
MERGED_MODEL = '/path/to/merged/model'
BASE_MODEL = '/path/to/base/model'
ADAPTER_DIR = '/path/to/adapter/directory'
TOKENIZER = BASE_MODEL

# GPU assignment (0-based)
GPU_MERGED = 0
GPU_PEFT = 1

# Tolerance
ATOL = 1e-5
RTOL = 1e-4

DTYPE = torch.float16
MAX_NEW_TOKENS = 128
PAD_AS_EOS = True

PROMPTS = [
    'User: Please explain the definition of CPU.\nAssistant: ',
    "Translate the following sentence to English: 'The wind is so refreshing today.'",
]


# -----------------------------
# Utilities
# -----------------------------
def pin_to_single_gpu(model_name: str, gpu_index: int, dtype=torch.bfloat16):
    device_map = {'': gpu_index}
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map=device_map).eval()
    return model


def attach_peft_on_gpu(base_model_name: str, adapter_dir: str, gpu_index: int, dtype=torch.bfloat16):
    device_map = {'': gpu_index}
    base = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=dtype, device_map=device_map).eval()
    model = PeftModel.from_pretrained(base, adapter_dir).eval()
    return model


def make_inputs(tokenizer: AutoTokenizer, seq_len: int = 32, batch_size: int = 2):
    texts = [f"Verification sample #{i}." for i in range(batch_size)]
    enc = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=seq_len)
    return enc


@dataclass
class DiffStat:
    max_abs: float
    mean_abs: float
    cos_sim: float


def tensor_diff(a: torch.Tensor, b: torch.Tensor) -> DiffStat:
    a = a.detach().float().cpu().reshape(-1)
    b = b.detach().float().cpu().reshape(-1)
    max_abs = (a - b).abs().max().item()
    mean_abs = (a - b).abs().mean().item()
    cos = float('nan') if a.norm() == 0 or b.norm() == 0 else torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    return DiffStat(max_abs, mean_abs, cos)


def report(stat: DiffStat, tag: str):
    ok = stat.max_abs <= ATOL + RTOL * max(1.0, stat.mean_abs)
    print(
        f"[{tag}] max|Δ|={stat.max_abs:.3e}  mean|Δ|={stat.mean_abs:.3e}  cos={stat.cos_sim:.6f}  -> {'OK' if ok else 'MISMATCH'}"
    )
    return ok


# -----------------------------
# 1) End-to-end comparison
# -----------------------------
@torch.inference_mode()
def compare_e2e(merged, peft_model, tokenizer):
    batch_cpu = make_inputs(tokenizer)
    batch_m = {k: v.to(f"cuda:{GPU_MERGED}") for k, v in batch_cpu.items()}
    batch_p = {k: v.to(f"cuda:{GPU_PEFT}") for k, v in batch_cpu.items()}

    out_m = merged(**batch_m, output_hidden_states=True)
    out_p = peft_model(**batch_p, output_hidden_states=True)

    ok1 = report(tensor_diff(out_m.logits, out_p.logits), 'logits')
    ok2 = report(tensor_diff(out_m.hidden_states[-1], out_p.hidden_states[-1]), 'hidden_states[-1]')
    return ok1 and ok2


# -----------------------------
# 2) Module-wise effective weight comparison
#    (W_eff = W0 + Σ(B@A)*scale)
# -----------------------------
def find_linear_modules(model: nn.Module, suffixes=('q_proj', 'k_proj', 'v_proj', 'o_proj')) -> Dict[str, nn.Linear]:
    out = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and any(name.endswith(f".self_attn.{suf}") for suf in suffixes):
            out[name] = mod
    return out


def peft_has_lora(mod: nn.Module) -> bool:
    return hasattr(mod, 'lora_A') and hasattr(mod, 'lora_B')


def peft_effective_weight(linear_mod: nn.Module) -> torch.Tensor:
    W0 = linear_mod.weight.detach().float().cpu()
    if not peft_has_lora(linear_mod):
        return W0

    delta = torch.zeros_like(W0)
    fan_in_fan_out = getattr(linear_mod, 'fan_in_fan_out', False)

    for name, A in linear_mod.lora_A.items():
        if name not in linear_mod.lora_B:
            continue
        B = linear_mod.lora_B[name]

        A_w = A.weight.detach().float().cpu()
        B_w = B.weight.detach().float().cpu()

        BA = (A_w.t() @ B_w.t()).t() if fan_in_fan_out else (B_w @ A_w)

        # scaling
        if hasattr(linear_mod, 'scaling'):
            scale = float(linear_mod.scaling[name])
        else:
            r = A_w.shape[0]
            alpha = getattr(linear_mod, 'lora_alpha', r)
            scale = float(alpha) / float(r)

        delta += BA * scale

    return W0 + delta


def _resolve_in_peft(peft_model: nn.Module, merged_name: str) -> Optional[nn.Module]:
    """
    Based on the merged module name, sequentially try possible prefixes in the PEFT wrapper.
    """
    candidates = [
        merged_name,
        f"base_model.{merged_name}",
        f"base_model.model.{merged_name}",
        f"base_model.model.model.{merged_name}",
    ]
    peft_named = dict(peft_model.named_modules())
    for cand in candidates:
        if cand in peft_named:
            return peft_named[cand]
    return None


@torch.inference_mode()
def compare_weights(merged, peft_model):
    ok_all = True
    merged_lin = find_linear_modules(merged)

    for name, m_lin in merged_lin.items():
        p_lin = _resolve_in_peft(peft_model, name)
        if p_lin is None:
            print(f"[SKIP] Cannot resolve in PEFT: {name}")
            ok_all = False
            continue

        W_merged = m_lin.weight.detach().float().cpu()
        W_peft_eff = peft_effective_weight(p_lin)

        ok = report(tensor_diff(W_merged, W_peft_eff), f"Weights::{name}")
        ok_all = ok_all and ok

    return ok_all


# -----------------------------
# Generation comparison
# -----------------------------
def load_models():
    tok = AutoTokenizer.from_pretrained(TOKENIZER, use_fast=True)
    if PAD_AS_EOS and tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'  # Improved batch padding stability in causal LM

    merged = AutoModelForCausalLM.from_pretrained(MERGED_MODEL, torch_dtype=DTYPE, device_map={'': GPU_MERGED}).eval()

    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=DTYPE, device_map={'': GPU_PEFT}).eval()

    peft_model = PeftModel.from_pretrained(base, ADAPTER_DIR).eval()
    model = peft_model.merge_and_unload().eval()

    return tok, merged, model


@torch.inference_mode()
def run_generate(model, tok, prompts, device, **gen_kwargs):
    enc = tok(prompts, return_tensors='pt', padding=True)
    enc = {k: v.to(f"cuda:{device}") for k, v in enc.items()}
    out = model.generate(**enc, **gen_kwargs, return_dict_in_generate=True, output_scores=True)
    texts = tok.batch_decode(out.sequences, skip_special_tokens=True)
    return out, texts


def compare_texts(a_list, b_list):
    ok = True
    for i, (a, b) in enumerate(zip(a_list, b_list)):
        same = (a == b)
        ok &= same
        tag = 'SAME ' if same else 'DIFF*'
        print(f"[{tag}] sample#{i}\n--- merged ---\n{a}\n--- base+peft ---\n{b}\n")
    return ok


def main():
    torch.manual_seed(0)

    tok, merged, peft_model = load_models()

    # ===== (Optional) End-to-end logits/hidden comparison =====
    print('\n=== (0) End-to-end tensors (sanity) ===')
    _ = compare_e2e(merged, peft_model, tok)

    # ===== 1) Deterministic verification (greedy) =====
    greedy_args = dict(
        do_sample=False,
        max_new_tokens=MAX_NEW_TOKENS,
        num_beams=1,  # ✅ Beam search disabled
        repetition_penalty=1.0,  # ✅ Keep default value
        temperature=None,
        top_p=None,
        top_k=None,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        use_cache=True)
    print('\n=== GREEDY (deterministic) ===')
    out_m_g, texts_m_g = run_generate(merged, tok, PROMPTS, GPU_MERGED, **greedy_args)
    out_p_g, texts_p_g = run_generate(peft_model, tok, PROMPTS, GPU_PEFT, **greedy_args)
    ok_greedy = compare_texts(texts_m_g, texts_p_g)

    # ===== 2) Module-wise effective weight comparison =====
    print('\n=== (2) Module-wise effective weights ===')
    ok_w = compare_weights(merged, peft_model)

    # Summary
    print('\n=== SUMMARY ===')
    print('GREEDY MATCH ✅' if ok_greedy else 'GREEDY MISMATCH ❌')
    if not ok_greedy:
        print('※ Please recheck adapter/key mapping to match from greedy.')


if __name__ == '__main__':
    main()
