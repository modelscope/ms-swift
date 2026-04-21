# feat: Add native NPU Flash Attention support

## Summary

This PR adds native NPU Flash Attention support through transformers' built-in `npu_flash_attention` integration, providing a lightweight alternative to MindSpeed-based optimizations for Ascend NPU users.

## Motivation

Currently, ms-swift supports attention implementations including `eager`, `sdpa`, `flash_attention_2`, and `flash_attention_3`, but lacks native support for NPU Flash Attention. While there are ongoing efforts to integrate MindSpeed for comprehensive NPU optimization, this PR offers a simpler, zero-additional-dependency approach for users who:

- Have Ascend NPU hardware available
- Want quick NPU acceleration without installing MindSpeed
- Need compatibility with standard transformers workflows

## Changes

### New Files
- `swift/model/npu_flash_attention.py`: Registration module for NPU Flash Attention
  - Auto-detects NPU availability
  - Registers `npu_flash_attention_forward` to transformers' `ALL_ATTENTION_FUNCTIONS`
  - Handles GQA (Grouped Query Attention) automatically
  - Supports causal attention detection from model config

### Modified Files
- `swift/model/__init__.py`: Auto-register NPU FA when NPU is detected on import
- `swift/model/utils.py`: 
  - Update `AttnImpl.to_use_flash_attn()` to recognize `npu_flash_attention`
  - Update `AttnImpl.update_attn_impl()` to preserve `npu_flash_attention` name (not convert to `flash_attention_2`)
- `swift/ui/llm_train/hyper.py`: Add `npu_flash_attention` to UI dropdown choices

## Usage

### Command Line
```bash
swift sft \
    --model_id qwen3.5-0.8B \
    --attn_impl npu_flash_attention \
    --dataset your_dataset
```

### Python API
```python
from swift import sft

sft(
    model_id='qwen3.5-0.8B',
    attn_impl='npu_flash_attention',
    dataset='your_dataset',
)
```

### Direct Transformers Usage
```python
import swift  # Auto-registers npu_flash_attention
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen3.5-0.8B',
    attn_implementation='npu_flash_attention',
)
```

## Technical Details

### Registration Flow
```python
# When import swift on NPU machine:
if is_torch_npu_available():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS["npu_flash_attention"] = npu_flash_attention_forward
```

### Attention Function Implementation
```python
def npu_flash_attention_forward(module, query, key, value, ...):
    # 1. Handle GQA (expand KV heads)
    # 2. Convert bshd -> bsnd format
    # 3. Call native npu_flash_attn_func()
    # 4. Convert bsnd -> bshd format
    return attn_output, None
```

### Environment Variable
Users can disable auto-registration if needed:
```bash
export SWIFT_DISABLE_NPU_FA=1
```

## Testing

### New Tests Added
- `tests/models/test_npu_flash_attention.py` (5 test cases)

### Test Scenarios
1. **NPU available**: Auto-register `npu_flash_attention` ✅
2. **Non-NPU environment**: Gracefully skip, no error ✅
3. **Module import**: Works on any machine ✅
4. **End-to-end workflow**: Registration → Model loading ✅
5. **Missing torch_npu**: No crash ✅

### Backward Compatibility Verified
- `attn_impl='flash_attn'` → converts to `flash_attention_2` ✅ (unchanged)
- `attn_impl='eager'` → stays `eager` ✅ (unchanged)
- `attn_impl='flash_attention_2'` → stays `flash_attention_2` ✅ (unchanged)

## Requirements

- `torch-npu` installed (standard for NPU users)
- CANN toolkit (standard for NPU users)
- No additional dependencies required

## Relation to Other PRs

This PR complements ongoing MindSpeed integration efforts by providing a lightweight, quick-enable option alongside comprehensive MindSpeed optimizations.

| Feature | This PR | MindSpeed PR |
|---------|---------|--------------|
| Dependency | None (transformers native) | MindSpeed |
| Scope | Attention only | Full model optimization |
| Setup | Zero-config | Requires MindSpeed install |
| Use case | Quick enable, standard workflows | Maximum performance |

## Checklist

- [x] Code follows project style guidelines
- [x] Added comprehensive tests
- [x] Verified backward compatibility
- [x] Added UI support
- [x] No breaking changes
- [x] Works on both NPU and non-NPU environments

## Related Issues

N/A - New feature

---

**Author**: Your Name
**Date**: 2026-04-20
