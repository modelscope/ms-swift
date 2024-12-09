# Instructions

The example provides instructions for using SWIFT for training, inference, deployment, evaluation, and quantization. By default, the model will be downloaded from the ModelScope community.

If you want to use the Huggingface community, you can change the command line like this:

```shell
...
swift sft \
    --model <model_id_or_path> \
    --use_hf 1 \
    ...
```
