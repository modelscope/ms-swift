import torch
from peft import IA3Config, PeftModel, get_peft_model

from swift.llm import MODEL_ARCH_MAPPING, ModelKeys
from swift.utils import find_all_linears


class Tuner:

    @staticmethod
    def prepare_model(args: 'TrainArguments', model):
        raise NotImplementedError

    @staticmethod
    def save_pretrained(
        model: torch.nn.Module,
        save_directory: str,
        safe_serialization: bool = True,
        **kwargs,
    ):
        raise NotImplementedError

    @staticmethod
    def from_pretrained(model: torch.nn.Module, model_id: str, **kwargs):
        raise NotImplementedError


class IA3(Tuner):

    @staticmethod
    def prepare_model(args: 'TrainArguments', model: torch.nn.Module):
        model_arch: ModelKeys = MODEL_ARCH_MAPPING[model.model_meta.model_arch]
        ia3_config = IA3Config(
            target_modules=find_all_linears(model), feedforward_modules='.*' + model_arch.mlp.split('{}.')[1] + '.*')
        return get_peft_model(model, ia3_config)

    @staticmethod
    def save_pretrained(
        model: torch.nn.Module,
        save_directory: str,
        safe_serialization: bool = True,
        **kwargs,
    ):
        model: PeftModel
        model.save_pretrained(save_directory, safe_serialization=safe_serialization, **kwargs)

    @staticmethod
    def from_pretrained(model: torch.nn.Module, model_id: str, **kwargs):
        return PeftModel.from_pretrained(model, model_id, **kwargs)


extra_tuners = {'ia3': IA3}
