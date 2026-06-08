NPU_TORCH_VERSION=${NPU_TORCH_VERSION:-2.7.1}
NPU_TORCH_NPU_VERSION=${NPU_TORCH_NPU_VERSION:-2.7.1.post2}
NPU_MODELSCOPE_VERSION=${NPU_MODELSCOPE_VERSION:-1.37.0}
NPU_PIP_INDEX=${NPU_PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple/}
NPU_CONSTRAINT_FILE=${NPU_CONSTRAINT_FILE:-/tmp/ms_swift_npu_constraints.txt}
NPU_PIP_BLOCK_CUDA_DEPS=${NPU_PIP_BLOCK_CUDA_DEPS:-True}

print_npu_warning() {
    echo "======================================================================"
    echo "WARNING: NPU runtime is unavailable, tests will continue on CPU path"
    echo "======================================================================"
}

setup_npu_pip_constraints() {
    cat >"$NPU_CONSTRAINT_FILE" <<EOF
torch==$NPU_TORCH_VERSION
torch_npu==$NPU_TORCH_NPU_VERSION
modelscope==$NPU_MODELSCOPE_VERSION
EOF
    if [ "$NPU_PIP_BLOCK_CUDA_DEPS" == "True" ]; then
        cat >>"$NPU_CONSTRAINT_FILE" <<'EOF'
# NPU CI should not resolve CUDA runtime wheels. If a dependency starts requiring
# these packages, fail in pip's resolver instead of downloading hundreds of MB.
cuda-toolkit<0
nvidia-cublas<0
nvidia-cuda-runtime<0
nvidia-cuda-nvrtc<0
nvidia-cuda-cupti<0
nvidia-cudnn<0
nvidia-cufft<0
nvidia-curand<0
nvidia-cusolver<0
nvidia-cusparse<0
nvidia-nccl<0
nvidia-nvjitlink<0
nvidia-nvtx<0
nvidia-cublas-cu12<0
nvidia-cuda-runtime-cu12<0
nvidia-cuda-nvrtc-cu12<0
nvidia-cuda-cupti-cu12<0
nvidia-cudnn-cu12<0
nvidia-cufft-cu12<0
nvidia-curand-cu12<0
nvidia-cusolver-cu12<0
nvidia-cusparse-cu12<0
nvidia-cusparselt-cu12<0
nvidia-nccl-cu12<0
nvidia-nvjitlink-cu12<0
nvidia-nvtx-cu12<0
EOF
    fi
    export PIP_CONSTRAINT="$NPU_CONSTRAINT_FILE"
    echo "Using NPU pip constraints: $PIP_CONSTRAINT"
    cat "$PIP_CONSTRAINT"
}

is_npu_runtime_matched() {
    python - <<PY
import importlib.util

expected_torch = '$NPU_TORCH_VERSION'
expected_torch_npu = '$NPU_TORCH_NPU_VERSION'

try:
    import torch
except Exception:
    raise SystemExit(1)

if importlib.util.find_spec('torch_npu') is None:
    raise SystemExit(1)

try:
    import torch_npu
except Exception:
    raise SystemExit(1)

torch_version = torch.__version__.split('+', 1)[0]
torch_npu_version = getattr(torch_npu, '__version__', '')
if torch_version == expected_torch and torch_npu_version == expected_torch_npu:
    raise SystemExit(0)

print(f'WARNING: NPU runtime version mismatch: torch={torch.__version__}, torch_npu={torch_npu_version}; '
      f'expected torch=={expected_torch}, torch_npu=={expected_torch_npu}')
raise SystemExit(1)
PY
}

ensure_npu_runtime() {
    if is_npu_runtime_matched; then
        echo "NPU runtime already matched: torch==$NPU_TORCH_VERSION torch_npu==$NPU_TORCH_NPU_VERSION"
        return
    fi

    echo "Installing NPU runtime: torch==$NPU_TORCH_VERSION torch_npu==$NPU_TORCH_NPU_VERSION"
    if ! python -m pip install --force-reinstall "torch==$NPU_TORCH_VERSION" "torch_npu==$NPU_TORCH_NPU_VERSION" -i "$NPU_PIP_INDEX"; then
        echo "WARNING: Failed to install torch/torch_npu NPU runtime packages."
        print_npu_warning
    fi
}

report_npu_runtime() {
    echo "==================== NPU runtime report ===================="
    echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
    echo "Installed torch/CUDA related pip packages:"
    python -m pip freeze | grep -Ei '^(torch|torch-npu|torch_npu|cuda-|nvidia-)' || true
    if command -v npu-smi >/dev/null 2>&1; then
        npu-smi info || echo "WARNING: npu-smi info failed."
    else
        echo "WARNING: npu-smi not found."
    fi
    python - <<'PY'
import importlib.util
import os

warning = 'WARNING: NPU runtime is unavailable, tests will continue on CPU path'
print(f"ASCEND_RT_VISIBLE_DEVICES={os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '')}")
print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")

try:
    import torch
    print(f"torch={torch.__version__}")
except Exception as e:
    print(f"WARNING: failed to import torch: {e!r}")
    print('=' * 70)
    print(warning)
    print('=' * 70)
    raise SystemExit(0)

try:
    import transformers
    print(f"transformers={transformers.__version__}")
except Exception as e:
    print(f"WARNING: failed to import transformers: {e!r}")

if importlib.util.find_spec('torch_npu') is None:
    print('WARNING: torch_npu is not installed.')
    print('=' * 70)
    print(warning)
    print('=' * 70)
    raise SystemExit(0)

try:
    import torch_npu
    print(f"torch_npu={getattr(torch_npu, '__version__', 'unknown')}")
except Exception as e:
    print(f"WARNING: failed to import torch_npu: {e!r}")
    print('=' * 70)
    print(warning)
    print('=' * 70)
    raise SystemExit(0)

try:
    npu = getattr(torch, 'npu', None)
    available = bool(npu is not None and npu.is_available())
    count = npu.device_count() if npu is not None else 0
    print(f"torch.npu.is_available={available}")
    print(f"torch.npu.device_count={count}")
    if not available:
        print('=' * 70)
        print(warning)
        print('=' * 70)
except Exception as e:
    print(f"WARNING: failed to query torch.npu status: {e!r}")
    print('=' * 70)
    print(warning)
    print('=' * 70)
PY
    echo "============================================================"
}

if [ "$MODELSCOPE_SDK_DEBUG" == "True" ]; then
    if [ "$SWIFT_CI_USE_NPU" == "True" ]; then
        setup_npu_pip_constraints
    fi

    # pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
    pip install -r requirements/tests.txt -i https://mirrors.aliyun.com/pypi/simple/
    git config --global --add safe.directory /ms-swift
    git config --global user.email tmp
    git config --global user.name tmp.com

    # linter test
    # use internal project for pre-commit due to the network problem
    if [ `git remote -v | grep alibaba  | wc -l` -gt 1 ]; then
        pre-commit run -c .pre-commit-config_local.yaml --all-files
        if [ $? -ne 0 ]; then
            echo "linter test failed, please run 'pre-commit run --all-files' to check"
            echo "From the repository folder"
            echo "Run 'pip install -r requirements/tests.txt' install test dependencies."
            echo "Run 'pre-commit install' install pre-commit hooks."
            echo "Finally run linter with command: 'pre-commit run --all-files' to check."
            echo "Ensure there is no failure!!!!!!!!"
            exit -1
        fi
    fi

    if [ "$SWIFT_CI_USE_NPU" == "True" ]; then
        ensure_npu_runtime
    fi
    pip install -r requirements/framework.txt -U -i https://mirrors.aliyun.com/pypi/simple/
    if [ "$SWIFT_CI_USE_NPU" == "True" ]; then
        ensure_npu_runtime
    fi
    pip install decord einops -U -i https://mirrors.aliyun.com/pypi/simple/
    pip uninstall autoawq -y
    pip install optimum
    pip install diffusers
    if [ "$SWIFT_CI_USE_NPU" == "True" ]; then
        pip install math-verify -i "$NPU_PIP_INDEX"
    fi
    pip install "transformers<5.0" "peft<0.19"
    # pip install autoawq -U --no-deps

    # test with install
    pip install .
    if [ "$SWIFT_CI_USE_NPU" == "True" ]; then
        echo "NPU CI skips auto_gptq because it is a CUDA/GPTQ optional dependency."
        pip install bitsandbytes deepspeed -U -i https://mirrors.aliyun.com/pypi/simple/
    else
        pip install auto_gptq bitsandbytes deepspeed -U -i https://mirrors.aliyun.com/pypi/simple/
    fi
    if [ "$SWIFT_CI_USE_NPU" == "True" ]; then
        ensure_npu_runtime
        report_npu_runtime
    fi
else
    echo "Running case in release image, run case directly!"
fi
# remove torch_extensions folder to avoid ci hang.
rm -rf ~/.cache/torch_extensions
if [ $# -eq 0 ]; then
    ci_command="python tests/run.py --subprocess"
else
    ci_command="$@"
fi
echo "Running case with command: $ci_command"
$ci_command
