# please use python=3.10/3.11, cuda12.*
# sh requirements/install_all.sh
pip install "sglang[all]" -U
pip install "vllm>=0.5.1" -U
pip install "lmdeploy>=0.5" -U
pip install "transformers<4.58" "trl<0.24" peft -U
pip install auto_gptq optimum bitsandbytes "gradio<5.33" -U
pip install git+https://github.com/modelscope/ms-swift.git#egg=ms-swift[all]
pip install timm "deepspeed<0.18" -U
pip install qwen_vl_utils qwen_omni_utils keye_vl_utils -U
pip install decord librosa icecream soundfile -U
pip install liger_kernel nvitop pre-commit math_verify==0.5.2 py-spy wandb swanlab -U
# flash-attn: https://github.com/Dao-AILab/flash-attention/releases
