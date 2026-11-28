# Instructions

The example provides instructions for using SWIFT for training, inference, deployment, evaluation, and quantization. By default, the model will be downloaded from the ModelScope community.

If you want to use the Huggingface community, you can change the command line like this:

```shell
...
USE_SWIFT_V5=0 swift sft \
    --model <model_id_or_path> \
    --use_hf 1 \
    ...
```

## Routing to the v4 CLI

The `swift` CLI defaults to v5 (`swift/dev`). Everything under `examples/v4` and `docs/v4` targets the v4 stack, so every `swift <cmd>` / `megatron <cmd>` command line in those directories is written as:

```shell
USE_SWIFT_V5=0 swift sft ...
USE_SWIFT_V5=0 megatron sft ...
```

Put the prefix in front of the other environment assignments (`USE_SWIFT_V5=0 CUDA_VISIBLE_DEVICES=0 swift sft ...`), or `export USE_SWIFT_V5=0` once in the shell. Without it the v4 arguments are rejected by the v5 CLI, and `megatron`/`sample`/`merge-lora` fail with a redirect error.

The `.py` scripts here import `swift.*` directly, which is the v4 package regardless of the variable, so they do not set it. The exception is a script that shells out to the `swift` CLI (see `train/rft/rft.py`): the child inherits the environment, so it has to set `USE_SWIFT_V5=0` itself.
