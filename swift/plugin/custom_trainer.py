import torch

from swift.trainers import Trainer, TrainerFactory


class SequenceClassificationTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=None, **kwargs):
        inputs['labels'] = torch.tensor(inputs.pop('label')).unsqueeze(1)
        return super().compute_loss(model, inputs, return_outputs=return_outputs)


def custom_trainer_class(trainer_mapping, training_args_mapping):
    trainer_mapping['train'] = 'swift.plugin.custom_trainer.SequenceClassificationTrainer'
