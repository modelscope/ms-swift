# Copyright (c) ModelScope Contributors. All rights reserved.
import torch.nn.functional as F

from swift.utils import get_logger
from .trainer import Trainer
from .utils import gather_for_unpadded_tensors

logger = get_logger()


class EmbeddingTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gather_function = gather_for_unpadded_tensors
        mrl_dims = self.args.mrl_dims
        if mrl_dims and self.compute_loss_func is not None:
            origin_loss_func = self.compute_loss_func

            def mrl_loss_func(outputs, labels, **kwargs):
                # Matryoshka Representation Learning: compute loss on each truncated dimension
                # and aggregate with the corresponding weights.
                last_hidden_state = outputs['last_hidden_state']
                loss = None
                for dim, weight in mrl_dims.items():
                    if dim > last_hidden_state.shape[-1]:
                        logger.warning_once(f'MRL: skipping dimension {dim} because it exceeds the model hidden size '
                                            f'({last_hidden_state.shape[-1]}).')
                        continue
                    sliced = F.normalize(last_hidden_state[..., :dim], p=2, dim=-1)
                    cur_loss = weight * origin_loss_func({'last_hidden_state': sliced}, labels, **kwargs)
                    loss = cur_loss if loss is None else loss + cur_loss
                return loss

            self.compute_loss_func = mrl_loss_func

    def evaluation_loop(self, *args, **kwargs):
        output = super().evaluation_loop(*args, **kwargs)
        self.gather_function = gather_for_unpadded_tensors
        return output
