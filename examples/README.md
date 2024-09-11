# Instructions

Examples offer a brief usage of the technics provided by SWIFT, by default the model will be downloaded from the ModelScope community, 
if you want to use the Huggingface community you can change the command line like this:
```shell
USE_HF=1 \
...
  swift sft \
  --model_id_or_path hf-group/hf-model \
  ...
```