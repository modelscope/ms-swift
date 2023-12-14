def _create_and_replace(
        self,
        lora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optional_kwargs,
):
    if current_key is None:
        raise ValueError("Current Key shouldn't be `None`")
    # Regexp matching - Find key which matches current target_name in patterns provided
    pattern_keys = list(chain(lora_config.rank_pattern.keys(), lora_config.alpha_pattern.keys()))
    target_name_key = next(filter(lambda key: re.match(f".*\.{key}$", current_key), pattern_keys), current_key)

    r = lora_config.rank_pattern.get(target_name_key, lora_config.r)
    alpha = lora_config.alpha_pattern.get(target_name_key, lora_config.lora_alpha)
    bias = hasattr(target, "bias") and target.bias is not None
    kwargs = {
        "r": r,
        "lora_alpha": alpha,
        "lora_dropout": lora_config.lora_dropout,
        "fan_in_fan_out": lora_config.fan_in_fan_out,
        "init_lora_weights": lora_config.init_lora_weights,
    }
    kwargs["loaded_in_8bit"] = optional_kwargs.pop("loaded_in_8bit", False)
    kwargs["loaded_in_4bit"] = optional_kwargs.pop("loaded_in_4bit", False)
    kwargs["bias"] = bias

    quantization_config = get_quantization_config(self.model, method="gptq")
    if quantization_config is not None:
        kwargs["gptq_quantization_config"] = quantization_config

    linear_types = (Linear,)
    if is_bnb_available():
        from .bnb import Linear8bitLt

        linear_types += (Linear8bitLt,)
    if is_bnb_4bit_available():
        from .bnb import Linear4bit

        linear_types += (Linear4bit,)

    # TODO: better deal with that
    if isinstance(target, Conv2d):
        target.update_layer_conv2d(
            adapter_name,
            r,
            alpha,
            lora_config.lora_dropout,
            lora_config.init_lora_weights,
        )
    elif isinstance(target, Embedding):
        target.update_layer_embedding(
            adapter_name,
            r,
            alpha,
            lora_config.lora_dropout,
            lora_config.init_lora_weights,
        )
    elif isinstance(target, linear_types):
        target.update_layer(
            adapter_name,
            r,
            alpha,
            lora_config.lora_dropout,
            lora_config.init_lora_weights,
        )
    else:
        new_module = self._create_new_module(lora_config, adapter_name, target, **kwargs)
        if adapter_name != self.active_adapter:
            # adding an additional adapter: it is not automatically trainable
            new_module.requires_grad_(False)
        self._replace_module(parent, target_name, new_module, target)


def _replace_module(self, parent, child_name, new_module, child):
    setattr(parent, child_name, new_module)
    # It's not necessary to set requires_grad here, as that is handled by
    # _mark_only_adapters_as_trainable

    # child layer wraps the original module, unpack it
    if hasattr(child, "base_layer"):
        child = child.base_layer

    if not hasattr(new_module, "base_layer"):
        new_module.weight = child.weight
        if hasattr(child, "bias"):
            new_module.bias = child.bias

    if getattr(child, "state", None) is not None:
        if hasattr(new_module, "base_layer"):
            new_module.base_layer.state = child.state
        else:
            new_module.state = child.state
        new_module.to(child.weight.device)

    # dispatch to correct device
    for name, module in new_module.named_modules():
        if (self.prefix in name) or ("ranknum" in name):
            weight = child.qweight if hasattr(child, "qweight") else child.weight
            module.to(weight.device)


def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
    for n, p in model.named_parameters():
        if self.prefix not in n:
            p.requires_grad = False

    for active_adapter in self.active_adapters:
        bias = self.peft_config[active_adapter].bias
        if bias == "none":
            continue

        if bias == "all":
            for n, p in model.named_parameters():
                if "bias" in n:
                    p.requires_grad = True
        elif bias == "lora_only":
            for m in model.modules():
                if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                    m.bias.requires_grad = True
        else:
            raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")


@staticmethod
def _create_new_module(lora_config, adapter_name, target, **kwargs):
    # avoid eager bnb import
    if is_bnb_available():
        import bitsandbytes as bnb

        from .bnb import Linear8bitLt

    if is_bnb_4bit_available():
        from .bnb import Linear4bit

    gptq_quantization_config = kwargs.get("gptq_quantization_config", None)
    AutoGPTQQuantLinear = get_auto_gptq_quant_linear(gptq_quantization_config)

    loaded_in_8bit = kwargs.pop("loaded_in_8bit", False)
    loaded_in_4bit = kwargs.pop("loaded_in_4bit", False)

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    megatron_core = None
    if lora_config.megatron_config:
        megatron_core = importlib.import_module(lora_config.megatron_core)

    if loaded_in_8bit and isinstance(target_base_layer, bnb.nn.Linear8bitLt):
        eightbit_kwargs = kwargs.copy()
        eightbit_kwargs.update(
            {
                "has_fp16_weights": target.state.has_fp16_weights,
                "memory_efficient_backward": target.state.memory_efficient_backward,
                "threshold": target.state.threshold,
                "index": target.index,
            }
        )
        new_module = Linear8bitLt(target, adapter_name, **eightbit_kwargs)
    elif loaded_in_4bit and is_bnb_4bit_available() and isinstance(target_base_layer, bnb.nn.Linear4bit):
        fourbit_kwargs = kwargs.copy()
        fourbit_kwargs.update(
            {
                "compute_dtype": target_base_layer.compute_dtype,
                "compress_statistics": target_base_layer.weight.compress_statistics,
                "quant_type": target_base_layer.weight.quant_type,
            }
        )
        new_module = Linear4bit(target, adapter_name, **fourbit_kwargs)
    elif AutoGPTQQuantLinear is not None and isinstance(target_base_layer, AutoGPTQQuantLinear):
        new_module = QuantLinear(target, adapter_name, **kwargs)
        target.qweight = target_base_layer.qweight
    elif isinstance(target_base_layer, torch.nn.Embedding):
        embedding_kwargs = kwargs.copy()
        embedding_kwargs.pop("fan_in_fan_out", None)
        embedding_kwargs.update(lora_config.loftq_config)
        new_module = Embedding(target, adapter_name, **embedding_kwargs)
    elif isinstance(target_base_layer, torch.nn.Conv2d):
        kwargs.update(lora_config.loftq_config)
        new_module = Conv2d(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, torch.nn.Linear):
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        kwargs.update(lora_config.loftq_config)
        new_module = Linear(target, adapter_name, **kwargs)
    elif megatron_core and isinstance(
            target_base_layer,
            (megatron_core.tensor_parallel.ColumnParallelLinear, megatron_core.tensor_parallel.RowParallelLinear),
    ):
        from .tp_layer import LoraParallelLinear

        megatron_kwargs = kwargs.copy()
        megatron_config = lora_config.megatron_config
        if isinstance(megatron_config, dict):
            transformer_config_class = megatron_core.transformer.transformer_config.TransformerConfig
            megatron_config = transformer_config_class(**lora_config.megatron_config)
        megatron_kwargs["megatron_config"] = megatron_config
        if megatron_kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `ColumnParallelLinear` "
                "or `RowParallelLinear`. "
                "Setting fan_in_fan_out to False."
            )
            megatron_kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        new_module = LoraParallelLinear(
            base_layer=target, adapter_name=adapter_name, backend=megatron_core.tensor_parallel, **megatron_kwargs
        )
    elif isinstance(target_base_layer, Conv1D):
        if not kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                "Setting fan_in_fan_out to True."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
        kwargs.update(lora_config.loftq_config)
        new_module = Linear(target, adapter_name, is_target_conv_1d_layer=True, **kwargs)
    else:
        raise ValueError(
            f"Target module {target} is not supported. Currently, only the following modules are supported: "
            "`torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.Conv2d`, `transformers.pytorch_utils.Conv1D`."
        )

    return new_module