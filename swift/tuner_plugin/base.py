# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import TYPE_CHECKING, Optional

import torch
from peft import PeftModel

if TYPE_CHECKING:
    from swift.arguments import SftArguments


class Tuner:
    """Base class for model tuners that adapt pre-trained models for specific tasks."""

    @staticmethod
    def prepare_model(args: 'SftArguments', model: torch.nn.Module) -> torch.nn.Module:
        """Prepare a new model with a tuner.

        Args:
            args: The training arguments containing tuner configuration.
            model: The model instance to be wrapped.

        Returns:
            The wrapped model with tuner applied.
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
        """Save the model checkpoint.

        Args:
            model: The wrapped model by `prepare_model`.
            save_directory: The directory path where the model will be saved.
            state_dict: The model's state_dict, used during DeepSpeed training.
                Only contains trainable parameters
            safe_serialization: Whether to use safetensors format for serialization. Defaults to True.
            **kwargs: Additional keyword arguments for saving.
        """
        raise NotImplementedError

    @staticmethod
    def from_pretrained(model: torch.nn.Module, model_id: str, **kwargs) -> torch.nn.Module:
        """Load a model from a checkpoint directory.

        Args:
            model: The original model instance.
            model_id: The model identifier or checkpoint directory path to load from.
            **kwargs: Additional keyword arguments for loading.

        Returns:
            The wrapped model instance with loaded weights.
        """
        raise NotImplementedError


class PeftTuner(Tuner):
    """Tuner implementation using the PEFT library."""

    @staticmethod
    def save_pretrained(
        model: torch.nn.Module,
        save_directory: str,
        state_dict: Optional[dict] = None,
        safe_serialization: bool = True,
        **kwargs,
    ) -> None:
        """Save the PEFT model checkpoint."""
        if isinstance(model, PeftModel):
            if 'selected_adapters' not in kwargs:
                kwargs['selected_adapters'] = ['default']
        model.save_pretrained(save_directory, safe_serialization=safe_serialization, **kwargs)

    @staticmethod
    def from_pretrained(model: torch.nn.Module, model_id: str, **kwargs) -> torch.nn.Module:
        """Load a PEFT model from a checkpoint."""
        return PeftModel.from_pretrained(model, model_id, **kwargs)
