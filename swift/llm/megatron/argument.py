import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from transformers.modeling_utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME, shard_checkpoint, load_sharded_checkpoint
from megatron.training.initialize import initialize_megatron
from megatron.training import get_args
# from megatron.training.checkpointing import get_checkpoint_name, get_checkpoint_tracker_filename, read_metadata
# from megatron.training.utils import get_ltor_masks_and_position_ids
import sys
from argparse import Namespace

@dataclass
class MegatronArguments:
    pass
    # hidden_size: Optional[int] = None
    # ffn_hidden_size: Optional[int] = None
    # num_attention_heads: Optional[int] = None
    # max_position_embeddings: Optional[int] = None
    # num_layers: Optional[int] = None
    # num_query_groups: Optional[int] = None
    # target_tensor_model_parallel_size: int = 1
    # target_pipeline_model_parallel_size: int = 1
    micro_batch_size: int = 1



    def get_megatron_args(self) -> Namespace:
        new_args = []

        sys._old_argv = sys.argv
        sys.argv = sys._old_argv[:1] + new_args

        initialize_megatron()
        return get_args()

if __name__ == '__main__':
    args = MegatronArguments()
    megatron_args = args.get_megatron_args()
    print()

