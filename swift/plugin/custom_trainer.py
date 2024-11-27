from swift.trainers import Seq2SeqTrainer


class SequenceClassificationTrainer(Seq2SeqTrainer):

    def compute_loss(self, model, inputs, return_outputs=None, num_items_in_batch=None):
        inputs['labels'] = inputs.pop('label')
        return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)


def custom_trainer_class(trainer_mapping, trainer_args_mapping):
    trainer_mapping['train'] = 'swift.plugin.custom_trainer.SequenceClassificationTrainer'
