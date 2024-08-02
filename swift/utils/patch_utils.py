from accelerate import DistributedType, Accelerator


def get_state_dict(self, model, unwrap=True):
    if self.distributed_type == DistributedType.DEEPSPEED:
        if self.deepspeed_config["zero_optimization"]["stage"] == 3:
            if model.zero_gather_16bit_weights_on_model_save():
                # Patch here to reduce memory usage
                state_dict = model._zero3_consolidated_16bit_state_dict(exclude_frozen_parameters=True)
            else:
                raise ValueError(
                    "Cannot get 16bit model weights because `stage3_gather_16bit_weights_on_model_save` in DeepSpeed config is False. "
                    "To save the model weights in 16bit, set `stage3_gather_16bit_weights_on_model_save` to True in DeepSpeed config file or "
                    "set `zero3_save_16bit_model` to True when using `accelerate config`. "
                    "To save the full checkpoint, run `model.save_checkpoint(save_dir)` and use `zero_to_fp32.py` to recover weights."
                )
        else:
            from deepspeed.checkpoint.utils import clone_tensors_for_torch_save

            state_dict = clone_tensors_for_torch_save(self.unwrap_model(model).state_dict())
    elif self.distributed_type == DistributedType.FSDP:
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
            state_dict = model.state_dict()
    else:
        if unwrap:
            model = self.unwrap_model(model)
        state_dict = model.state_dict()

    return state_dict


def patch_accelerate():
    Accelerator.get_state_dict = get_state_dict
