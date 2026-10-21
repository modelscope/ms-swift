# Twinkle Kernel Integration

## 1. Implementation

This change integrates Twinkle kernel optimization into ms-swift.

### Configuration

The `enable_kernel` option controls whether Twinkle kernelization is enabled.

### Integration Point

The integration is implemented in:

`swift/dev/builders/model.py`

When `enable_kernel=True`, Swift calls `kernelize(model.model)` after model construction to apply the Twinkle kernel mappings.

When the option is disabled, the existing model-building path is retained.

## 2. Unit Test Results

Test command:

python -m pytest swift/dev/tests/component/config/test_kernel_config.py swift/dev/tests/component/model/test_kernel.py

Result: **3 passed**

The tests cover the kernel configuration and integration call path.

## 3. NPU Validation

Environment:
- Hardware: Ascend 910B
- PyTorch: 2.9.0
- torch_npu: 2.9.0
- Model: Qwen2.5-0.5B-Instruct

Configuration: `enable_kernel=True`

The following kernel mappings were observed in the Twinkle logs:
- `Qwen2RMSNorm` → `rms_norm` → NPU backend
- `apply_rotary_pos_emb` → `rotary` → NPU backend
- `Qwen2MLP.forward` → `swiglu` → NPU backend

The model was moved to `npu:0`, and the forward test completed successfully:

`forward PASS`

This validates kernel activation and successful forward execution for the tested model. Performance improvements have not yet been benchmarked.
