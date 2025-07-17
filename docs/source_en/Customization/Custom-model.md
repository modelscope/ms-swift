# Custom Model

The models built into ms-swift can be used directly by specifying either `model_id` or `model_path`: `--model <model_id_or_path>`. ms-swift determines the `model_type` based on the suffix of `model_id/model_path` and the `config.json` file.

Each `model_type` has a unique model structure, template, and loading method. Of course, you can also manually override these by passing `--model_type` and `--template`. You can check the supported `model_type` and templates in the [Supported Models and Datasets](../Instruction/Supported-models-and-datasets.md).

The following introduces how to register a new model and its corresponding template.

## Model Registration

Custom models are typically implemented using model registration. You can refer to the [built-in model](https://github.com/modelscope/ms-swift/blob/main/swift/llm/model/model/qwen.py), the [built-in dialogue template](https://github.com/modelscope/ms-swift/blob/main/swift/llm/template/template/qwen.py), or the example code in the [examples](https://github.com/modelscope/ms-swift/blob/main/examples/custom). You can specify the `--custom_register_path xxx.py` to parse the externally registered content, which is convenient for users installing via pip instead of git clone.

The `register_model` function registers a model in the `MODEL_MAPPING`. You can complete the model registration by calling the function `register_model(model_meta)`, where `model_meta` will store the model's metadata. The parameter list for ModelMeta is as follows:

- model_type: Required. The model type, which is also the unique ID.
- model_groups: Required. Lists the ModelScope/HuggingFace model IDs and local paths. Running the [run_model_info.py](https://github.com/modelscope/ms-swift/blob/main/scripts/utils/run_model_info.py) file will automatically generate the [supported models documentation](https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html) and automatically match the model_type based on the `--model` suffix.
- template: Required. The default template type when `--template` is not specified in the command line.
- get_function: Required. The loading function for the model and tokenizer/processor (for multi-modal models). LLM is typically set to `get_model_tokenizer_with_flash_attn`.
- model_arch: The model architecture. Defaults to None. Multi-modal model training requires setting this parameter to determine the prefix for llm/vit/aligner.
- architectures: The architectures item in config.json, used to automatically match the model with its model_type. Defaults to `[]`.
- additional_saved_files: Files that need to be additionally saved during full parameter training and merge-lora. Defaults to `[]`.
- torch_dtype: The default dtype when `torch_dtype` is not passed during model loading. Defaults to None, read from config.json.
- is_multimodal: Indicates whether the model is multi-modal. Defaults to False.
- ignore_patterns: File patterns to be ignored when downloading from the hub. Defaults to `[]`.

The `register_template` function registers a dialogue template in `TEMPLATE_MAPPING`. To complete the registration of the dialogue template, simply call the function `register_template(template_meta)`, where `template_meta` will store the metadata of the template. The parameter list for TemplateMeta is as follows:

- template_type: Required. The type of dialogue template, which also serves as a unique ID.
- prefix: Required. The prefix of the dialogue template, usually encompassing parts like system, bos_token, and is generated independently of multi-turn dialogue loops. For example, the prefix for qwen is `[]`.
- prompt: Required. Represents the dialogue portion before `{{RESPONSE}}`. We use `{{QUERY}}` as a placeholder for the user's inquiry part. For example, the prompt for qwen is `['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n']`.
- chat_sep: Required. The separator for each turn in multi-turn dialogues. If set to None, the template does not support multi-turn dialogue. For example, the chat_sep for qwen is `['<|im_end|>\n']`.
- suffix: Defaults to `[['eos_token_id']]`. The suffix part of the dialogue template, generated independently of multi-turn dialogue loops, usually the eos_token. For example, the suffix for qwen is `['<|im_end|>']`.
- template_cls: Defaults to `Template`. Customization is generally required when defining templates for multimodal models, particularly in customizing the `_encode`, `_post_encode`, and `_data_collator` functions.
- system_prefix: Defaults to None. The prefix for dialogue templates with a system. We use`{{SYSTEM}}`as a placeholder for the system. For example, the system_prefix for qwen is`['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n']`.
  - Note: If the system is empty and `prefix` can be replaced by `system_prefix`, you can write `prefix` as a prefix including the system without setting `system_prefix`.
  - If the prefix does not include `{{SYSTEM}}` and system_prefix is not set, the template does not support the system.
- default_system: Defaults to None. The default system used when `--system` is not provided. For example, the default_system for qwen is `'You are a helpful assistant.'`.
- stop_words: Defaults to`[]`. Additional stop words besides eos_token and`suffix[-1]`. For example, the stop_words for qwen is`['<|endoftext|>']`
  - Note: During inference, the output response will be filtered by eos_token and `suffix[-1]`, but additional stop_words will be retained.
