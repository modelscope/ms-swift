Please refer to the examples in [examples/infer](../../infer/) and change `swift infer` to `swift deploy` to start the service.

e.g.
```shell
CUDA_VISIBLE_DEVICES=0 \
swift deploy \
    --model Qwen/Qwen2.5-7B-Instruct \
    --infer_backend vllm
```
