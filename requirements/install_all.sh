# please use python=3.10, cuda12.*
# sh requirements/install_all.sh
pip install "vllm>=0.5.1,<0.8" -U
pip install "lmdeploy>=0.5" -U --no-deps
pip install autoawq -U --no-deps
pip install auto_gptq optimum bitsandbytes -U
pip install git+https://github.com/modelscope/ms-swift.git#egg=ms-swift[all]
pip install timm -U
pip install deepspeed -U
pip install qwen_vl_utils qwen_omni_utils decord librosa pyav icecream soundfile -U
# flash-attn: https://github.com/Dao-AILab/flash-attention/releases
