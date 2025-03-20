# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Optional

import torch
from peft import IA3Config, PeftModel, get_peft_model

from swift.llm import MODEL_ARCH_MAPPING, ModelKeys
from swift.utils import find_all_linears


class Tuner:

    @staticmethod
    def prepare_model(args: 'TrainArguments', model: torch.nn.Module) -> torch.nn.Module:
        """Prepare a new model with a tuner

        Args:
            args: The training arguments
            model: The model instance

        Returns:
            The wrapped model
        """
        raise NotImplementedError

    @staticmethod
    def save_pretrained(
        model: torch.nn.Module,
        save_directory: str,
        state_dict: Optional[dict] = None,
        safe_serialization: bool = True,
        **kwargs,
    ) -> None:
        """Save when save_steps reaches

        Args:
            model: The wrapped model by `prepare_model`
            save_directory: The directory to save
            safe_serialization: Use safetensors or not
        """
        raise NotImplementedError

    @staticmethod
    def from_pretrained(model: torch.nn.Module, model_id: str, **kwargs) -> torch.nn.Module:
        """Load the ckpt_dir

        Args:
            model: The original model instance.
            model_id: The model id or ckpt_dir to load
        Returns:
            The wrapped model instance
        """
        raise NotImplementedError


class PeftTuner(Tuner):

    @staticmethod
    def save_pretrained(
        model: torch.nn.Module,
        save_directory: str,
        state_dict: Optional[dict] = None,
        safe_serialization: bool = True,
        **kwargs,
    ) -> None:
        model.save_pretrained(save_directory, safe_serialization=safe_serialization, **kwargs)

    @staticmethod
    def from_pretrained(model: torch.nn.Module, model_id: str, **kwargs) -> torch.nn.Module:
        return PeftModel.from_pretrained(model, model_id, **kwargs)


# Here gives a simple example of IA3
class IA3(PeftTuner):

    @staticmethod
    def prepare_model(args: 'TrainArguments', model: torch.nn.Module) -> torch.nn.Module:
        model_arch: ModelKeys = MODEL_ARCH_MAPPING[model.model_meta.model_arch]
        ia3_config = IA3Config(
            target_modules=find_all_linears(model), feedforward_modules='.*' + model_arch.mlp.split('{}.')[1] + '.*')
        return get_peft_model(model, ia3_config)


class DummyTuner(PeftTuner):

    @staticmethod
    def prepare_model(args: 'TrainArguments', model: torch.nn.Module) -> torch.nn.Module:
        return model


# Add your own tuner here, use --train_type xxx to begin
extra_tuners = {'ia3': IA3, 'dummy': DummyTuner}
