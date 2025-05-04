import abc
from abc import abstractmethod


class SequenceParallel(abc.ABC):

    @abstractmethod
    def init_sequence_parallel(self, size):
        pass

    @abstractmethod
    def prepare_model(self, model, tokenizer, split_in_forward):
        pass

    @abstractmethod
    def pad_and_split_inputs(self, tokenizer, input_ids, labels, position_ids, attention_mask, loss_scale):
        pass

    @abstractmethod
    def reduce_outputs(self, loss, labels):
        pass

    @abstractmethod
    def world_size(self):
        pass

    @abstractmethod
    def get_dataloader(self, trainer):
        pass
