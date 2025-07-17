# please use python=3.10, cuda12.*
# sh requirements/install_all.sh
pip install "sglang[all]<0.4.7" -U
pip install "vllm>=0.5.1,<0.9" "transformers<4.52" -U
pip install "lmdeploy>=0.5,<0.9" -U --no-deps
pip install autoawq -U --no-deps
pip install auto_gptq optimum bitsandbytes -U
pip install git+https://github.com/modelscope/ms-swift.git
pip install timm -U
pip install "deepspeed<0.17" -U
pip install qwen_vl_utils qwen_omni_utils decord librosa icecream soundfile -U
pip install liger_kernel nvitop pre-commit math_verify -U
# flash-attn: https://github.com/Dao-AILab/flash-attention/releases
