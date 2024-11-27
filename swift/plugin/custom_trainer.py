from swift.trainers import Trainer
from swift.utils import compute_acc, use_torchacc
import torch

from swift.utils.torchacc_utils import ta_trim_graph


class SequenceClassificationTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=None, **kwargs):
        inputs['labels'] = torch.tensor(inputs.pop('label')).unsqueeze(1)
        return super().compute_loss(model, inputs, return_outputs=return_outputs)


def custom_trainer_class(trainer_mapping, trainer_args_mapping):
    TrainerFactory.trainer_mapping['train'] = 'swift.plugin.custom_trainer.SequenceClassificationTrainer'


custom_trainer_class()
