# Copyright (c) Kakao Corp. (AI Alignment Team).
# Contact: kevin.us@kakaocorp.com

import json
import os
from collections import OrderedDict
from dataclasses import asdict

from safetensors.torch import save_file
from swift.utils import get_logger

logger = get_logger()


def convert_mcore_lora_to_hf_peft(peft_model, mg_model, dst_dir: str, num_groups: int) -> None:
    """
    Megatron Core LoRA 어댑터를 HuggingFace PEFT 형식으로 변환합니다.
    
    Args:
        peft_model:  Megatron Core PEFTModel
        mg_model: loaded Megatron Core Model (for shape)
        dst_dir: Dir path to saving HuggingFace PEFT
        num_groups: number of Attention group 
    """
    os.makedirs(dst_dir, exist_ok=True)
    dst_model = os.path.join(dst_dir, "adapter_model.safetensors")
    dst_cfg = os.path.join(dst_dir, "adapter_config.json")
    
    logger.info(f"Converting Megatron Core LoRA to HF PEFT format at {dst_dir}")
    
    # Megatron Core 모델에서 shape 추론
    logger.info("Extracting shape information from Megatron Core model...")
    mg_language_model = mg_model.language_model if hasattr(mg_model, 'language_model') else mg_model
    
    # Megatron Core의 attention 모듈에서 shape 추론
    # 첫 번째 레이어의 attention 모듈 사용
    first_layer = mg_language_model.decoder.layers[0]
    if hasattr(first_layer, 'self_attention'):
        attn_module = first_layer.self_attention
    else:
        attn_module = first_layer.attention
    
    # Megatron Core의 attention shape 추론
    if hasattr(attn_module, 'linear_qkv'):
        # fused qkv의 경우
        qkv_weight = attn_module.linear_qkv.weight
        out_features, in_features = qkv_weight.shape
        q_dim = out_features // (num_groups * 3)  # q, k, v 각각
        kv_dim = q_dim
    else:
        # 분리된 q, k, v의 경우
        q_weight = attn_module.linear_q.weight
        k_weight = attn_module.linear_k.weight
        v_weight = attn_module.linear_v.weight
        q_out, in_features = q_weight.shape
        k_out, _ = k_weight.shape
        v_out, _ = v_weight.shape
        
        q_dim = q_out // num_groups
        kv_dim = k_out // num_groups
        assert v_out // num_groups == kv_dim, "k/v group out dim mismatch"
    
    logger.info(f"Shape inference: num_groups={num_groups}, q_dim={q_dim}, kv_dim={kv_dim}, in_features={in_features}")
    
    # peft_model의 state_dict에서 모듈 단위 버킷화
    logger.info("Extracting LoRA weights from loaded PEFTModel...")
    bucket = {}  # prefix -> {local_name: tensor}
    state_dict = peft_model.state_dict()
    
    for fullkey, tensor in state_dict.items():
        # adapter 관련 키만 처리
        if "lora_A" not in fullkey and "lora_B" not in fullkey:
            continue
        parts = fullkey.split(".")
        local = ".".join(parts[-2:])     # e.g., lora_A.weight
        prefix = ".".join(parts[:-2])    # e.g., ...linear_qkv
        bucket.setdefault(prefix, {})[local] = tensor.cpu()
    
    dst_tensors = OrderedDict()
    
    def push(dst, key, tensor):
        """텐서를 저장용으로 독립 복사 + 연속 메모리 보장"""
        t = tensor.detach().clone().contiguous()
        if key in dst:
            raise ValueError(f"Duplicate key: {key}")
        if "weight" not in key:
            logger.debug(f"Skipping non-weight key: {key}")
            return
        key = remap_key_for_peft(key)
        dst[key] = t
    
    def remap_key_for_peft(key: str) -> str:
        """키를 HuggingFace PEFT 형식으로 변환"""
        # 1) decoder → model
        key = key.replace(".decoder.layers.", ".model.layers.")
        # 2) self_attention → self_attn
        key = key.replace(".self_attention.", ".self_attn.")
        # 3) check prefix
        if key.startswith("model.layers."):
            key = "base_model.model." + key
        return key
    
    def convert_linear_proj(prefix, tensors):
        """mcore: ...self_attention.linear_proj -> HF: ...self_attn.o_proj"""
        new_prefix = prefix.replace(".self_attention.linear_proj", ".self_attn.o_proj")
        for local, T in tensors.items():
            push(dst_tensors, f"{new_prefix}.{local}", T)
    
    def convert_linear_qkv(prefix, tensors):
        """
        Megatron Core fused qkv LoRA를 HF q_proj, k_proj, v_proj로 분리
        
        mcore:
          A: [r, in_features]  (공유)
          B: [num_groups*(q_dim+kv_dim+kv_dim), r]
        -> HF:
          q_proj: A=[r,in], B=[num_groups*q_dim, r]
          k_proj: A=[r,in], B=[num_groups*kv_dim, r]
          v_proj: A=[r,in], B=[num_groups*kv_dim, r]
        """
        A = tensors.get("lora_A.weight", None)
        B = tensors.get("lora_B.weight", None)
        if A is None or B is None:
            # 핵심 가중치 없으면 원키로 pass
            for local, T in tensors.items():
                push(dst_tensors, f"{prefix}.{local}", T)
            return
        
        r, in_A = A.shape
        out_B, rB = B.shape
        assert rB == r, f"LoRA rank mismatch: A={r}, B={rB}"
        assert in_A == in_features, f"in_features mismatch: A={in_A}, base={in_features}"
        
        expected_out = num_groups * (q_dim + kv_dim + kv_dim)
        assert out_B == expected_out, f"Fused B out({out_B}) != expected({expected_out})"
        
        # [num_groups, (q_dim+kv_dim+kv_dim), r]로 reshape한 뒤 슬라이스
        Bg = B.reshape(num_groups, q_dim + kv_dim + kv_dim, r)
        Bq = Bg[:, :q_dim, :].reshape(num_groups * q_dim, r)
        Bk = Bg[:, q_dim:q_dim+kv_dim, :].reshape(num_groups * kv_dim, r)
        Bv = Bg[:, q_dim+kv_dim:, :].reshape(num_groups * kv_dim, r)
        
        misc = {k: v for k, v in tensors.items() if k not in ("lora_A.weight", "lora_B.weight")}
        
        # q_proj
        q_prefix = prefix.replace(".self_attention.linear_qkv", ".self_attn.q_proj")
        push(dst_tensors, f"{q_prefix}.lora_A.weight", A)
        push(dst_tensors, f"{q_prefix}.lora_B.weight", Bq)
        
        # k_proj
        k_prefix = prefix.replace(".self_attention.linear_qkv", ".self_attn.k_proj")
        push(dst_tensors, f"{k_prefix}.lora_A.weight", A)
        push(dst_tensors, f"{k_prefix}.lora_B.weight", Bk)
        for k, v in misc.items():
            push(dst_tensors, f"{k_prefix}.{k}", v)
        
        # v_proj
        v_prefix = prefix.replace(".self_attention.linear_qkv", ".self_attn.v_proj")
        push(dst_tensors, f"{v_prefix}.lora_A.weight", A)
        push(dst_tensors, f"{v_prefix}.lora_B.weight", Bv)
        for k, v in misc.items():
            push(dst_tensors, f"{v_prefix}.{k}", v)
    
    def convert_mla_attention(prefix, tensors):
        """
        Multi-Latent Attention (MLA) LoRA 변환
        
        mcore -> HF:
          linear_q_down_proj -> q_a_proj
          linear_q_up_proj -> q_b_proj
          linear_kv_down_proj -> kv_a_proj_with_mqa
          linear_kv_up_proj -> kv_b_proj
        """
        # q_proj (down -> a, up -> b)
        if ".linear_q_down_proj" in prefix:
            new_prefix = prefix.replace(".linear_q_down_proj", ".q_a_proj")
            for local, T in tensors.items():
                push(dst_tensors, f"{new_prefix}.{local}", T)
        elif ".linear_q_up_proj" in prefix:
            new_prefix = prefix.replace(".linear_q_up_proj", ".q_b_proj")
            for local, T in tensors.items():
                push(dst_tensors, f"{new_prefix}.{local}", T)
        elif ".linear_kv_down_proj" in prefix:
            new_prefix = prefix.replace(".linear_kv_down_proj", ".kv_a_proj_with_mqa")
            for local, T in tensors.items():
                push(dst_tensors, f"{new_prefix}.{local}", T)
        elif ".linear_kv_up_proj" in prefix:
            new_prefix = prefix.replace(".linear_kv_up_proj", ".kv_b_proj")
            for local, T in tensors.items():
                push(dst_tensors, f"{new_prefix}.{local}", T)
    
    def convert_mlp_linear_fc1(prefix, tensors):
        """
        MLP linear_fc1 LoRA를 HF gate_proj, up_proj로 분리
        
        mcore: linear_fc1 [gate_up_dim, in_features]
        -> HF: gate_proj [gate_dim, in_features], up_proj [up_dim, in_features]
        """
        A = tensors.get("lora_A.weight", None)
        B = tensors.get("lora_B.weight", None)
        if A is None or B is None:
            for local, T in tensors.items():
                push(dst_tensors, f"{prefix}.{local}", T)
            return
        
        # gate_up_dim을 gate_dim과 up_dim으로 분리 (보통 1:1 비율)
        gate_up_dim = B.shape[0]
        gate_dim = gate_up_dim // 2
        up_dim = gate_up_dim - gate_dim
        
        # B를 gate와 up으로 분리
        B_gate = B[:gate_dim, :]
        B_up = B[gate_dim:, :]
        
        misc = {k: v for k, v in tensors.items() if k not in ("lora_A.weight", "lora_B.weight")}
        
        # gate_proj
        gate_prefix = prefix.replace(".mlp.linear_fc1", ".mlp.gate_proj")
        push(dst_tensors, f"{gate_prefix}.lora_A.weight", A)
        push(dst_tensors, f"{gate_prefix}.lora_B.weight", B_gate)
        for k, v in misc.items():
            push(dst_tensors, f"{gate_prefix}.{k}", v)
        
        # up_proj
        up_prefix = prefix.replace(".mlp.linear_fc1", ".mlp.up_proj")
        push(dst_tensors, f"{up_prefix}.lora_A.weight", A)
        push(dst_tensors, f"{up_prefix}.lora_B.weight", B_up)
        for k, v in misc.items():
            push(dst_tensors, f"{up_prefix}.{k}", v)
    
    def convert_mlp_linear_fc2(prefix, tensors):
        """MLP linear_fc2 LoRA를 HF down_proj로 변환"""
        new_prefix = prefix.replace(".mlp.linear_fc2", ".mlp.down_proj")
        for local, T in tensors.items():
            push(dst_tensors, f"{new_prefix}.{local}", T)
    
    def convert_moe_experts(prefix, tensors):
        """MoE experts LoRA 변환"""
        # experts[expert_idx].linear_fc1 -> experts[expert_idx].gate_proj, up_proj
        if ".linear_fc1" in prefix:
            convert_mlp_linear_fc1(prefix, tensors)
        # experts[expert_idx].linear_fc2 -> experts[expert_idx].down_proj
        elif ".linear_fc2" in prefix:
            convert_mlp_linear_fc2(prefix, tensors)
    
    # 모듈별 변환 실행
    for prefix, tensors in bucket.items():
        # Attention 변환
        if ".self_attention.linear_proj" in prefix:
            convert_linear_proj(prefix, tensors)
        elif ".self_attention.linear_qkv" in prefix:
            convert_linear_qkv(prefix, tensors)
        # Multi-Latent Attention 변환
        elif any(x in prefix for x in [".linear_q_down_proj", ".linear_q_up_proj", 
                                      ".linear_kv_down_proj", ".linear_kv_up_proj"]):
            convert_mla_attention(prefix, tensors)
        # MLP 변환
        elif ".mlp.linear_fc1" in prefix:
            convert_mlp_linear_fc1(prefix, tensors)
        elif ".mlp.linear_fc2" in prefix:
            convert_mlp_linear_fc2(prefix, tensors)
        # MoE experts 변환 (router는 제외)
        elif ".experts" in prefix and (".linear_fc1" in prefix or ".linear_fc2" in prefix):
            convert_moe_experts(prefix, tensors)
        else:
            # 알 수 없는 모듈은 그대로 복사
            logger.warning(f"Unknown module pattern: {prefix}")
            for local, T in tensors.items():
                push(dst_tensors, f"{prefix}.{local}", T)
    
    # 변환된 텐서 저장
    save_file(dst_tensors, dst_model, metadata={"format": "pt"})
    logger.info(f"Saved converted LoRA tensors to {dst_model}")
    
    # adapter_config.json 갱신
    logger.info("Converting adapter config...")
    cfg = peft_model.peft_config['default'] if isinstance(peft_model.peft_config['default'], dict) else asdict(peft_model.peft_config['default'])
    
    tm = cfg.get("target_modules", None)
    if tm is not None:
        if isinstance(tm, str):
            tm = [tm]
        new_tm = []
        for t in tm:
            if t == "linear_proj":
                new_tm.append("o_proj")
            elif t in ("linear_qkv", "query_key_value"):
                new_tm.extend(["q_proj", "k_proj", "v_proj"])
            elif t == "linear_fc1":
                new_tm.extend(["gate_proj", "up_proj"])
            elif t == "linear_fc2":
                new_tm.append("down_proj")
            elif t == "linear_q_down_proj":
                new_tm.append("q_a_proj")
            elif t == "linear_q_up_proj":
                new_tm.append("q_b_proj")
            elif t == "linear_kv_down_proj":
                new_tm.append("kv_a_proj_with_mqa")
            elif t == "linear_kv_up_proj":
                new_tm.append("kv_b_proj")
            else:
                new_tm.append(t)
        cfg["target_modules"] = sorted(set(new_tm))
    
    with open(dst_cfg, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2, default=str)

    logger.info(f"cfg: {cfg}")
    logger.info(f"Saved converted adapter config to {dst_cfg}")
