#!/usr/bin/env python3
"""
验证 npu_fusion_attention 是否原生支持 GQA (Grouped Query Attention)

运行方式:
    cd ~/Documents/project/ms-swift
    python test_gqa_native.py

预期输出:
    - 如果原生支持 GQA: 显示 "✅ 原生 GQA 支持正常" 和 backward 成功
    - 如果不支持: 显示具体错误信息
"""

import torch
import sys


def check_npu_available():
    """检查 NPU 是否可用"""
    if not hasattr(torch, "npu"):
        print("❌ torch.npu 不存在，请确认 torch-npu 已安装")
        return False
    if not torch.npu.is_available():
        print("❌ NPU 设备不可用")
        return False
    print(f"✅ NPU 可用，设备数量: {torch.npu.device_count()}")
    return True


def test_gqa_forward():
    """测试 npu_fusion_attention 是否支持不同 head 数的 Q/K/V (GQA)"""
    from torch_npu import npu_fusion_attention

    batch_size = 2
    seq_len = 16
    num_q_heads = 32   # Query heads
    num_kv_heads = 8   # Key/Value heads (GQA: 4:1)
    head_dim = 64

    device = "npu:0"

    # 创建测试张量
    q = torch.randn(batch_size, seq_len, num_q_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, device=device)

    print(f"\n📐 测试配置:")
    print(f"   Q shape: {q.shape}  (bs={batch_size}, seq={seq_len}, heads={num_q_heads}, dim={head_dim})")
    print(f"   K shape: {k.shape}  (bs={batch_size}, seq={seq_len}, heads={num_kv_heads}, dim={head_dim})")
    print(f"   V shape: {v.shape}  (bs={batch_size}, seq={seq_len}, heads={num_kv_heads}, dim={head_dim})")
    print(f"   GQA 比例: {num_q_heads // num_kv_heads}:1")

    # 转换为 BSND 格式
    q_bsnd = q.permute(0, 2, 1, 3).contiguous()
    k_bsnd = k.permute(0, 2, 1, 3).contiguous()
    v_bsnd = v.permute(0, 2, 1, 3).contiguous()

    print(f"\n📐 BSND 格式:")
    print(f"   Q: {q_bsnd.shape}")
    print(f"   K: {k_bsnd.shape}")
    print(f"   V: {v_bsnd.shape}")

    # 测试 non-causal forward
    print("\n🔹 测试 1: Non-causal forward (不同 KV heads)...")
    try:
        out = npu_fusion_attention(
            q_bsnd, k_bsnd, v_bsnd,
            head_num=num_q_heads,
            input_layout="BSND",
            keep_prob=1.0,
            scale=head_dim ** -0.5,
        )[0]
        print(f"   ✅ Forward 成功! Output shape: {out.shape}")
        return out
    except Exception as e:
        print(f"   ❌ Forward 失败: {e}")
        return None


def test_gqa_backward(out, q_bsnd, k_bsnd, v_bsnd):
    """测试 backward 是否正常"""
    print("\n🔹 测试 2: Backward (不同 KV heads)...")
    try:
        # 重新创建带 requires_grad 的张量
        q = q_bsnd.detach().clone().requires_grad_(True)
        k = k_bsnd.detach().clone().requires_grad_(True)
        v = v_bsnd.detach().clone().requires_grad_(True)

        from torch_npu import npu_fusion_attention
        out = npu_fusion_attention(
            q, k, v,
            head_num=q.shape[1],
            input_layout="BSND",
            keep_prob=1.0,
            scale=q.shape[-1] ** -0.5,
        )[0]

        loss = out.sum()
        loss.backward()

        print(f"   ✅ Backward 成功!")
        print(f"   Q grad shape: {q.grad.shape}, sum: {q.grad.sum().item():.4f}")
        print(f"   K grad shape: {k.grad.shape}, sum: {k.grad.sum().item():.4f}")
        print(f"   V grad shape: {v.grad.shape}, sum: {v.grad.sum().item():.4f}")
        return True
    except Exception as e:
        print(f"   ❌ Backward 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gqa_with_expand():
    """对比测试：用 expand 方式做 GQA 的 forward + backward"""
    from torch_npu import npu_fusion_attention

    batch_size = 2
    seq_len = 16
    num_q_heads = 32
    num_kv_heads = 8
    head_dim = 64
    device = "npu:0"

    q = torch.randn(batch_size, seq_len, num_q_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, device=device)

    # expand 到相同 head 数
    n_rep = num_q_heads // num_kv_heads
    k_expanded = k.unsqueeze(3).expand(batch_size, seq_len, num_kv_heads, n_rep, head_dim)
    k_expanded = k_expanded.reshape(batch_size, seq_len, num_q_heads, head_dim)
    v_expanded = v.unsqueeze(3).expand(batch_size, seq_len, num_kv_heads, n_rep, head_dim)
    v_expanded = v_expanded.reshape(batch_size, seq_len, num_q_heads, head_dim)

    q_bsnd = q.permute(0, 2, 1, 3).contiguous().requires_grad_(True)
    k_bsnd = k_expanded.permute(0, 2, 1, 3).contiguous().requires_grad_(True)
    v_bsnd = v_expanded.permute(0, 2, 1, 3).contiguous().requires_grad_(True)

    print("\n🔹 测试 3: Expand 方式的 GQA forward + backward...")
    try:
        out = npu_fusion_attention(
            q_bsnd, k_bsnd, v_bsnd,
            head_num=num_q_heads,
            input_layout="BSND",
            keep_prob=1.0,
            scale=head_dim ** -0.5,
        )[0]
        loss = out.sum()
        loss.backward()
        print(f"   ✅ Expand 方式成功!")
        print(f"   Q grad sum: {q_bsnd.grad.sum().item():.4f}")
        print(f"   K grad sum: {k_bsnd.grad.sum().item():.4f}")
        print(f"   V grad sum: {v_bsnd.grad.sum().item():.4f}")
        return True
    except Exception as e:
        print(f"   ❌ Expand 方式失败: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("NPU Fusion Attention GQA 原生支持验证")
    print("=" * 60)

    if not check_npu_available():
        print("\n⚠️  当前环境没有 NPU，请在 NPU 服务器上运行此脚本")
        sys.exit(1)

    # 测试 1 & 2: 原生 GQA (不同 head 数)
    out = test_gqa_forward()
    if out is not None:
        # 重新准备带 grad 的张量做 backward 测试
        batch_size, seq_len, num_q_heads, num_kv_heads, head_dim = 2, 16, 32, 8, 64
        device = "npu:0"
        q = torch.randn(batch_size, seq_len, num_q_heads, head_dim, device=device)
        k = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, device=device)
        v = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, device=device)
        q_bsnd = q.permute(0, 2, 1, 3).contiguous()
        k_bsnd = k.permute(0, 2, 1, 3).contiguous()
        v_bsnd = v.permute(0, 2, 1, 3).contiguous()
        test_gqa_backward(out, q_bsnd, k_bsnd, v_bsnd)

    # 测试 3: Expand 方式对比
    test_gqa_with_expand()

    print("\n" + "=" * 60)
    print("验证完成。请将结果截图或复制给我。")
    print("=" * 60)
