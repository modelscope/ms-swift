from peft import IA3Config, PeftModel, get_peft_model

from swift.utils import find_all_linears


class Tuner:

    @staticmethod
    def prepare_model(args, model):
        raise NotImplementedError

    @staticmethod
    def save_pretrained(model, output_dir):
        raise NotImplementedError

    @staticmethod
    def from_pretrained(model, output_dir):
        raise NotImplementedError


class IA3(Tuner):

    @staticmethod
    def prepare_model(args: 'TrainArguments', model):
        model_group = args.get_model_group()
        mapping: ModelKeys = MODEL_KEYS_MAPPING.get(model_group)

        if not mapping:
            raise ValueError('Module not supported')
        ia3_config = IA3Config(
            target_modules=find_all_linears(model, 0, args.model_type, None),
            feedforward_modules=mapping.mlp.split('{}')[1])
        return get_peft_model(model, ia3_config)

    @staticmethod
    def save_pretrained(model, output_dir):
        model: PeftModel
        model.save_pretrained(output_dir)

    @staticmethod
    def from_pretrained(model, output_dir):
        return PeftModel.from_pretrained(model, output_dir)


extra_tuners = {'ia3': IA3}
