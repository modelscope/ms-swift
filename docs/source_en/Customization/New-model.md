# New Model

It is generally recommended to specify the model ID directly using `--model`, in conjunction with `--model_type` and `--template`. For example:

```shell
swift sft --model my-model --model_type llama --template chatml --dataset xxx
```

If you need to add a new `model_type` or `template`, please submit an issue to us. If you have read our source code, you can also add new types in `llm/template` and `llm/model`.
