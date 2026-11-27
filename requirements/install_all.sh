# please use python=3.11/3.12/3.13, cuda12.*
# run from the repository root: sh requirements/install_all.sh
# pip install sglang -U
pip install "vllm>=0.5.1" -U
pip install optimum bitsandbytes "gradio<5.33" -U
# base + eval + megatron extras; transformers/trl/peft/datasets/ray bounds live in pyproject.toml
pip install -e ".[all,megatron]"
# released build instead of the checkout: pip install "ms-swift[all,megatron]@git+https://github.com/modelscope/ms-swift.git"
pip install timm "deepspeed>=0.19.3,<0.20" -U
pip install qwen_vl_utils qwen_omni_utils keye_vl_utils -U
pip install torchvision decord librosa icecream soundfile -U
pip install pybind11 liger_kernel nvitop pre-commit math_verify py-spy wandb swanlab -U
pip install "flash-attn==2.8.3" --no-build-isolation

# megatron (not covered by any extra: every item below compiles from source)
pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable --no-build-isolation
pip install git+https://github.com/deepseek-ai/DeepGEMM.git@v2.1.1.post3 --no-build-isolation
pip install -U flash-linear-attention --no-build-isolation
pip install -U git+https://github.com/Dao-AILab/causal-conv1d.git --no-build-isolation
pip install git+https://github.com/Dao-AILab/fast-hadamard-transform --no-build-isolation
pip install git+https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git@v0.3.0
