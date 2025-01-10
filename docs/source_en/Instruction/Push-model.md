# Pushing Model

When using SWIFT, users can choose to push their trained model to the community.

```shell
swift sft/export/pt/rlhf \
    --use_hf 0/1 \
    ...
```

The above parameter controls the community to which the model is pushed (this parameter also affects downloads). When set to 0, it pushes to ModelScope; when set to 1, it pushes to Hugging Face. The default value is 0. You can also use an environment variable to control it:
```shell
USE_HF=0/1 \
swift sft/export/pt/rlhf \
    ...
```

## Pushing to ModelScope
To use the model pushing feature, first ensure you have registered for a ModelScope official account and obtained your sdk token from the [page](https://www.modelscope.cn/my/myaccesstoken).

### Pushing Model During Training

To push the model during training, you need to add the following parameters in the command line:
```shell
--push_to_hub true \
--hub_model_id my-group/my-model \
--hub_token <token-from-modelscope-page>
```

Once these parameters are added, the trained checkpoint and parameters will be pushed to the ModelScope community for future use. Note that ModelScope allows you to upload private models. To make the model private, add the following parameter:

```shell
--hub_private_repo true
```

This way, only individuals with permissions from the organization can see the model.

### Pushing Model During Export

The parameters for pushing the model during export are the same as those for training:
```shell
# If it is full parameter training, it is `--model`
CUDA_VISIBLE_DEVICES=0 swift export \
    --adapters output/vx-xxx/checkpoint-xxx \
    --push_to_hub true \
    --hub_model_id '<your-model-id>' \
    --hub_token '<your-sdk-token>'
```

This allows you to push the LoRA merged or quantized model directly to ModelScope.

## Pushing to Hugging Face

You can register for a Hugging Face token on this [page](https://huggingface.co/settings/tokens). After registration, execute the following locally:
```shell
huggingface-cli login
# Enter the token in the popup input
```

After logging in locally, you only need to specify in the command line:
```shell
--push_to_hub true \
--hub_model_id my-group/my-model
```
to push the model.
