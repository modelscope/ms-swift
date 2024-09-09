# Push Model

When using SWIFT, users can choose to push their trained models to the ModelScope community.

To utilize the model pushing feature, first ensure that you have registered for an official ModelScope account and obtained the corresponding SDK token on the [page](https://www.modelscope.cn/my/myaccesstoken).

## Pushing Model During Training

To push a model during training, you need to add the following parameters in the command line:
```shell
--push_to_hub true \
--hub_model_id my-group/my-model \
--hub_token <token-from-modelscope-page> \
```

Once these parameters are added, the trained checkpoint and training parameters will be pushed to the ModelScope community, making it easier to use later. It is important to note that ModelScope community allows you to upload private models. If you want the model to be private, add the following parameter:

```shell
--hub_private_repo true
```

This way, only those with the organization's permission can see the model.

## Pushing Model During Export

The parameters for pushing a model during export are the same as those during training:
```shell
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir output/qwen1half-4b-chat/vx-xxx/checkpoint-xxx \
    --push_to_hub true \
    --hub_model_id qwen1half-4b-chat-lora \
    --hub_token '<your-sdk-token>'
```

This allows you to directly push the merged or quantized LoRA model to ModelScope.
