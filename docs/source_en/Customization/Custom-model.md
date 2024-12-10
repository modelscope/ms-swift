# Custom Model

The models built into ms-swift can be used directly by specifying either `model_id` or `model_path`: `--model <model_id_or_path>`. ms-swift determines the `model_type` based on the suffix of `model_id/model_path` and the `config.json` file. Each `model_type` has a unique model structure, template, and loading method. Of course, you can also manually override these by passing `--model_type` and `--template`. You can check the supported `model_type` and templates in the [Supported Models and Datasets](../Instruction/Supported-models-and-datasets.md).

> [!TIP]
> When using `swift sft` to fine-tune a base model into a chat model using LoRA technology, for instance, fine-tuning Llama3.2-1B into a chat model, you may need to manually set the template. Adding the `--template default` parameter can help avoid issues where the base model fails to stop properly due to encountering special characters in the conversation template that it hasn't seen before.
## Model Registration

Please refer to the example code in [examples](https://github.com/modelscope/swift/blob/main/examples/custom). You can parse the registered content by specifying `--custom_register_path xxx.py`.
