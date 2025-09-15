# Copyright (c) Alibaba, Inc. and its affiliates.
from contextlib import contextmanager

import torch
from megatron.core import InferenceParams
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel import VocabParallelEmbedding, scatter_to_sequence_parallel_region
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training import get_args

from .gpt_model import GPTModel


class MultimodalGPTModel(MegatronModule):

    def __init__(self,
                 config: TransformerConfig,
                 transformer_layer_spec: ModuleSpec,
                 vocab_size: int,
                 max_sequence_length: int,
                 pre_process: bool = True,
                 post_process: bool = True,
                 *args,
                 **kwargs):
        super().__init__(config)
        self.pre_process = pre_process
        self.post_process = post_process
        self.language_model = GPTModel(config, transformer_layer_spec, vocab_size, max_sequence_length, pre_process,
                                       post_process, *args, **kwargs)

        self.share_embeddings_and_output_weights = self.language_model.share_embeddings_and_output_weights
        args = get_args()
        self.visual = None
        if pre_process and args.megatron_model_meta.visual_cls is not None:
            self.visual = args.megatron_model_meta.visual_cls(config)

    @contextmanager
    def _patch_word_embeddings(self, kwargs):
        origin_forward = VocabParallelEmbedding.forward

        def forward(_self, input_):
            reduce_scatter_embeddings = _self.reduce_scatter_embeddings
            _self.reduce_scatter_embeddings = False
            input_ = torch.masked_fill(input_, input_ < 0, 0)
            res = origin_forward(_self, input_)
            _self.reduce_scatter_embeddings = reduce_scatter_embeddings
            if self.visual is not None:
                res = self.visual.get_inputs_embeds(res, **kwargs)
            if reduce_scatter_embeddings:
                res = res.transpose(0, 1).contiguous()
                res = scatter_to_sequence_parallel_region(res, group=_self.tp_group)
            return res

        VocabParallelEmbedding.forward = forward
        try:
            yield
        finally:
            VocabParallelEmbedding.forward = origin_forward

    # Code borrowed from NVIDIA/Megatron-LM
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        decoder_input: torch.Tensor = None,
        labels: torch.Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        **kwargs,
    ) -> torch.Tensor:
        if decoder_input is not None:
            pass
        elif self.pre_process:
            from ..trainers.utils import get_batch_on_this_cp_rank
            kwargs.update({'input_ids': input_ids})
            with self._patch_word_embeddings(kwargs):
                decoder_input = self.language_model.embedding(input_ids=input_ids, position_ids=position_ids)
                decoder_input = get_batch_on_this_cp_rank({
                    'decoder_input': decoder_input,
                    'packed_seq_params': packed_seq_params
                })['decoder_input']
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = None
        return self.language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
        )

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        return self.language_model.set_input_tensor(input_tensor)
