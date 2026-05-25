# please use python=3.10/3.11, cuda12.*
# sh requirements/install_all.sh
# pip install sglang -U
pip install "vllm>=0.5.1" -U
pip install "transformers<5.9" "trl<1.0" "peft<0.20" "datasets<4.8.5" -U
pip install optimum bitsandbytes "gradio<5.33" mcore-bridge -U
pip install "ms-swift[all]@git+https://github.com/modelscope/ms-swift.git"
pip install timm "deepspeed<0.19" ray -U
pip install qwen_vl_utils qwen_omni_utils keye_vl_utils -U
pip install decord librosa icecream soundfile -U
pip install liger_kernel nvitop pre-commit math_verify py-spy wandb swanlab -U
pip install "flash-attn==2.8.3" --no-build-isolation
# megatron
pip install pybind11 git+https://github.com/NVIDIA/TransformerEngine.git@stable --no-build-isolation
pip install git+https://github.com/deepseek-ai/DeepGEMM.git@v2.1.1.post3 --no-build-isolation
pip install -U flash-linear-attention --no-build-isolation
pip install -U git+https://github.com/Dao-AILab/causal-conv1d --no-build-isolation
pip install git+https://github.com/Dao-AILab/fast-hadamard-transform --no-build-isolation
pip install git+https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git@v0.2.0
# apex
