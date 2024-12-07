# Using Tuners

Tuners refer to additional structural components attached to a model, aimed at reducing the number of training parameters or enhancing training accuracy. The tuners currently supported by SWIFT include:

- LoRA: [LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/abs/2106.09685)
- LoRA+: [LoRA+: Efficient Low Rank Adaptation of Large Models](https://arxiv.org/pdf/2402.12354.pdf)
- LLaMA PRO: [LLAMA PRO: Progressive LLaMA with Block Expansion](https://arxiv.org/pdf/2401.02415.pdf)
- GaLore/Q-GaLore: [GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection](https://arxiv.org/abs/2403.03507)
- Liger Kernel: [Liger Kernel: Efficient Triton Kernels for LLM Training](https://arxiv.org/abs/2410.10989)
- LISA: [LISA: Layerwise Importance Sampling for Memory-Efficient Large Language Model Fine-Tuning](https://arxiv.org/abs/2403.17919)
- UnSloth: https://github.com/unslothai/unsloth
- SCEdit: [SCEdit: Efficient and Controllable Image Diffusion Generation via Skip Connection Editing](https://arxiv.org/abs/2312.11392)  < [arXiv](https://arxiv.org/abs/2312.11392)  |  [Project Page](https://scedit.github.io/) >
- NEFTune: [Noisy Embeddings Improve Instruction Finetuning](https://arxiv.org/abs/2310.05914)
- LongLoRA: [Efficient Fine-tuning of Long-Context Large Language Models](https://arxiv.org/abs/2309.12307)
- Adapter: [Parameter-Efficient Transfer Learning for NLP](http://arxiv.org/abs/1902.00751)
- Vision Prompt Tuning: [Visual Prompt Tuning](https://arxiv.org/abs/2203.12119)
- Side: [Side-Tuning: A Baseline for Network Adaptation via Additive Side Networks](https://arxiv.org/abs/1912.13503)
- Res-Tuning: [Res-Tuning: A Flexible and Efficient Tuning Paradigm via Unbinding Tuner from Backbone](https://arxiv.org/abs/2310.19859)  < [arXiv](https://arxiv.org/abs/2310.19859)  |  [Project Page](https://res-tuning.github.io/)  |  [Usage](ResTuning.md) >
- Tuners provided by [PEFT](https://github.com/huggingface/peft), such as AdaLoRA, DoRA, Fourierft, etc.

## Interface List

### Swift Class Static Interfaces

- `Swift.prepare_model(model, config, **kwargs)`
  - Function: Loads a tuner into a model. If it is a subclass of `PeftConfig`, it uses the corresponding interface from the Peft library to load the tuner. When using `SwiftConfig`, this interface can accept `SwiftModel` instances and can be called repeatedly, functioning similarly to passing a dictionary of configs.
    - This interface supports the parallel loading of multiple tuners of different types for concurrent use.
  - Parameters:
    - `model`: An instance of `torch.nn.Module` or `SwiftModel`, the model to be loaded.
    - `config`: An instance of `SwiftConfig` or `PeftConfig`, or a dictionary of custom tuner names paired with their respective configs.
  - Return Value: An instance of `SwiftModel` or `PeftModel`.

- `Swift.merge_and_unload(model)`
  - Function: Merges LoRA weights back into the original model and completely unloads the LoRA component.
  - Parameters:
    - `model`: An instance of `SwiftModel` or `PeftModel` that has had LoRA loaded.
  - Return Value: None.

- `Swift.merge(model)`
  - Function: Merges LoRA weights back into the original model without unloading the LoRA component.
  - Parameters:
    - `model`: An instance of `SwiftModel` or `PeftModel` that has had LoRA loaded.
  - Return Value: None.

- `Swift.unmerge(model)`
  - Function: Splits LoRA weights back from the original model weights into the LoRA structure.
  - Parameters:
    - `model`: An instance of `SwiftModel` or `PeftModel` that has had LoRA loaded.
  - Return Value: None.

- `Swift.save_to_peft_format(ckpt_dir, output_dir)`
  - Function: Converts stored LoRA checkpoints to a PEFT-compatible format. Key changes include:
    - The `default` will be split from the corresponding `default` folder into the root directory of `output_dir`.
    - The `{tuner_name}.` field will be removed from weight keys, e.g., `model.layer.0.self.in_proj.lora_A.default.weight` becomes `model.layer.0.self.in_proj.lora_A.weight`.
    - Weight keys will have a `basemodel.model` prefix added.
    - Note: Only LoRA can be converted; other types of tuners will raise conversion errors due to PEFT not supporting them. Additionally, due to the presence of extra parameters in LoRAConfig, such as `dtype`, conversion to Peft format is not supported when these parameters are set. In such cases, you can manually delete the corresponding fields in adapter_config.json.
  - Parameters:
    - `ckpt_dir`: Original weights directory.
    - `output_dir`: The target directory for the weights.
  - Return Value: None.

- `Swift.from_pretrained(model, model_id, adapter_name, revision, **kwargs)`
  - Function: Load the tuner onto the model from the stored weights directory. If `adapter_name` is not provided, all tuners from the `model_id` directory will be loaded. This interface can also be called repeatedly, similar to `prepare_model`.
  - Parameters:
    - `model`: An instance of `torch.nn.Module` or `SwiftModel` to which the tuner will be loaded.
    - `model_id`: A string indicating the tuner checkpoint to be loaded, which can be an ID from the model hub or a local directory.
    - `adapter_name`: Can be of type `str`, `List[str]`, `Dict[str, str]`, or `None`. If `None`, all tuners in the specified directory will be loaded. If it is a `str` or `List[str]`, only specific tuners will be loaded. If it is a `Dict`, the key represents the tuner to load, which will be renamed to the corresponding value.
    - `revision`: If `model_id` is an ID from the model hub, `revision` can specify the corresponding version number.

### SwiftModel Interfaces

Below is a list of interfaces that users may call. Other internal or less recommended interfaces can be viewed by running the `make docs` command to access the API Doc.

- `SwiftModel.create_optimizer_param_groups(self, **defaults)`
  - Function: Creates parameter groups based on the loaded tuners; currently, this only applies to the `LoRA+` algorithm.
  - Parameters:
    - `defaults`: Default parameters for the `optimizer_groups`, such as `lr` and `weight_decay`.
  - Return Value:
    - The created `optimizer_groups`.

- `SwiftModel.add_weighted_adapter(self, ...)`
  - Function: Merges existing LoRA tuners into one.
  - Parameters:
    - This interface is a passthrough to `PeftModel.add_weighted_adapter`, and parameters can be referenced in the [add_weighted_adapter documentation](https://huggingface.co/docs/peft/main/en/package_reference/lora#peft.LoraModel.add_weighted_adapter).

- `SwiftModel.save_pretrained(self, save_directory, safe_serialization, adapter_name)`
  - Function: Saves tuner weights.
  - Parameters:
    - `save_directory`: The directory for saving.
    - `safe_serialization`: Whether to use safe tensors, default is `False`.
    - `adapter_name`: Stored adapter tuner, if not provided, defaults to storing all tuners.

- `SwiftModel.set_active_adapters(self, adapter_names, offload=None)`
  - Function: Sets the currently active adapters; adapters not in the list will be deactivated.
    - In inference, the environment variable `USE_UNIQUE_THREAD=0/1`, default is `1`. If set to `0`, then `set_active_adapters` only takes effect in the current thread, at which point it defaults to using the tuners activated in this thread, with tuners in different threads not interfering with each other.
  - Parameters:
    - `adapter_names`: The names of the active tuners.
    - `offload`: How to handle deactivated adapters; default is `None`, meaning they remain in GPU memory. Can also use `cpu` or `meta` to offload to CPU or meta device to reduce memory consumption. In `USE_UNIQUE_THREAD=0`, do not pass the `offload` value to avoid affecting other threads.
  - Return Value: None.

- `SwiftModel.activate_adapter(self, adapter_name)`
  - Function: Activates a tuner.
    - In inference, the environment variable `USE_UNIQUE_THREAD=0/1`, default is `1`. If set to `0`, `activate_adapter` will only be effective for the current thread, at which point it defaults to using the tuners activated in this thread, with tuners in different threads not interfering with each other.
  - Parameters:
    - `adapter_name`: The name of the tuner to be activated.
  - Return Value: None.

- `SwiftModel.deactivate_adapter(self, adapter_name, offload)`
  - Function: Deactivates a tuner.
    - During `inference`, do not call this interface when the `USE_UNIQUE_THREAD=0`.
  - Parameters:
    - `adapter_name`: The name of the tuner to be deactivated.
    - `offload`: How to handle deactivated adapters; defaults to `None`, meaning they remain in GPU memory. Can also use `cpu` or `meta` to offload to CPU or meta device to reduce memory consumption.
  - Return Value: None.

- `SwiftModel.get_trainable_parameters(self)`
  - Function: Returns information about the trainable parameters.
  - Parameters: None.
  - Return Value: Information about trainable parameters in the following format:
    ```text
    trainable params: 100M || all params: 1000M || trainable%: 10.00% || cuda memory: 10GiB.
    ```
