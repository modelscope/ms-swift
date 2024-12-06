# please use python=3.10, cuda12.*
pip install vllm -U
pip install lmdeploy -U --no-deps
pip install autoawq -U --no-deps
pip install auto_gptq optimum bitsandbytes -U
pip install -e .[all]
pip install deepspeed==0.14.*
# flash-attn: https://github.com/Dao-AILab/flash-attention/releases
