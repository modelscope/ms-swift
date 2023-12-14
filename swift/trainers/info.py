def inject_adapter(self, model: nn.Module, adapter_name: str):
    r"""
    Creates adapter layers and replaces the target modules with the adapter layers. This method is called under the
    hood by `peft.mapping.get_peft_model` if a non-prompt tuning adapter class is passed.

    The corresponding PEFT config is directly retrieved from the `peft_config` attribute of the BaseTuner class.

    Args:
        model (`nn.Module`):
            The model to be tuned.
        adapter_name (`str`):
            The adapter name.
    """
    peft_config = self.peft_config[adapter_name]
    # Note: If possible, all checks should be performed *at the start of this method*.
    # This way, we can raise early if something goes wrong, without leaving the model
    # in a bad (half-initialized) state.
    self._check_new_adapter_config(peft_config)

    is_target_modules_in_base_model = False
    key_list = [key for key, _ in model.named_modules()]

    _check_for_modules_to_save = getattr(peft_config, "modules_to_save", None) is not None
    _has_modules_to_save = False

    model_config = getattr(model, "config", {"model_type": "custom"})
    if hasattr(model_config, "to_dict"):
        model_config = model_config.to_dict()

    peft_config = self._prepare_adapter_config(peft_config, model_config)

    for key in key_list:
        # Check for modules_to_save in case
        if _check_for_modules_to_save and any(
                key.endswith(f"{module_to_save}") for module_to_save in peft_config.modules_to_save
        ):
            # Optionally set the modules to save
            parent, target, target_name = _get_submodules(model, key)

            if not isinstance(target, ModulesToSaveWrapper):
                new_module = ModulesToSaveWrapper(target, adapter_name)
                setattr(parent, target_name, new_module)
            else:
                target.update(adapter_name)

            _has_modules_to_save = True
            continue

        if not self._check_target_module_exists(peft_config, key):
            continue

        is_target_modules_in_base_model = True
        parent, target, target_name = _get_submodules(model, key)

        optional_kwargs = {
            "loaded_in_8bit": getattr(model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(model, "is_loaded_in_4bit", False),
            "current_key": key,
        }
        self._create_and_replace(peft_config, adapter_name, target, target_name, parent, **optional_kwargs)

    if not is_target_modules_in_base_model:
        raise ValueError(
            f"Target modules {peft_config.target_modules} not found in the base model. "
            f"Please check the target modules and try again."
        )

    self._mark_only_adapters_as_trainable(model)

    if self.peft_config[adapter_name].inference_mode:
        for n, p in model.named_parameters():
            if adapter_name in n:
                p.requires_grad = False

    if _has_modules_to_save:
        if not hasattr(model, "modules_to_save"):
            model.modules_to_save = set(peft_config.modules_to_save)
        else:
            model.modules_to_save.update(set(peft_config.modules_to_save))
