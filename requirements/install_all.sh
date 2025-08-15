# please use python=3.10, cuda12.*
# sh requirements/install_all.sh
pip install "sglang[all]" -U
pip install "vllm>=0.5.1" "transformers<4.55" "trl<0.21" -U
pip install "lmdeploy>=0.5" -U
pip install autoawq -U --no-deps
pip install auto_gptq optimum bitsandbytes "gradio<5.33" -U
pip install git+https://github.com/modelscope/ms-swift.git
pip install timm -U
pip install "deepspeed" -U
pip install qwen_vl_utils qwen_omni_utils decord librosa icecream soundfile -U
pip install liger_kernel nvitop pre-commit math_verify py-spy -U
# flash-attn: https://github.com/Dao-AILab/flash-attention/releases
