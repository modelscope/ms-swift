# Copyright (c) Alibaba, Inc. and its affiliates.


def ta_accelerate(model,
                  fsdp_num,
                  layer_cls_name,
                  bf16=True,
                  fp16=False,
                  gradient_checkpointing=True,
                  fsdp_flatten_parameters=False):
    """ accelerate LLM training using TorchAcc(only available internally).
    """
    import torchacc as ta
    assert layer_cls_name is not None

    def get_ta_config():
        config = ta.Config()
        config.compute.fp16 = fp16
        config.compute.bf16 = bf16

        config.memory.gc = gradient_checkpointing
        if config.memory.gc:
            config.memory.gc_cls = {layer_cls_name}

        config.dist.fsdp.size = fsdp_num
        config.dist.fsdp.wrap_layer_cls = {layer_cls_name}
        config.dist.fsdp.flatten_parameters = fsdp_flatten_parameters
        config.dist.dp.size = 1

        return config

    ta_config = get_ta_config()
    model = ta.accelerate(model, ta_config)
    return model
