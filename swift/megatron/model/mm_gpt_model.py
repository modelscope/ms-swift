# Copyright (c) ModelScope Contributors. All rights reserved.
from contextlib import contextmanager

import megatron.core
import torch
from megatron.core import InferenceParams
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel import VocabParallelEmbedding, reduce_scatter_to_sequence_parallel_region
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training import get_args
from packaging import version

from .gpt_model import GPTModel

mcore_013 = version.parse(megatron.core.__version__) >= version.parse('0.13.0rc0')


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
        from .register import get_megatron_model_meta
        super().__init__(config)
        self.pre_process = pre_process
        self.post_process = post_process
        self.language_model = GPTModel(config, transformer_layer_spec, vocab_size, max_sequence_length, pre_process,
                                       post_process, *args, **kwargs)
        self.vp_stage = self.language_model.vp_stage
        self.share_embeddings_and_output_weights = self.language_model.share_embeddings_and_output_weights
        args = get_args()
        self.megatron_model_meta = get_megatron_model_meta(args.hf_model_type)
        self.visual = None
        if args.mtp_num_layers:
            raise ValueError('MTP currently does not support multimodal models.')
        if pre_process and self.megatron_model_meta.visual_cls is not None:
            self.visual = self.megatron_model_meta.visual_cls(config)

    @contextmanager
    def _patch_word_embeddings(self, kwargs):
        origin_forward = VocabParallelEmbedding.forward

        def forward(_self, input_):
            from ..trainers.utils import split_cp_inputs
            args = get_args()
            reduce_scatter_embeddings = _self.reduce_scatter_embeddings
            _self.reduce_scatter_embeddings = False
            input_ = torch.masked_fill(input_, input_ < 0, 0)
            res = origin_forward(_self, input_)
            _self.reduce_scatter_embeddings = reduce_scatter_embeddings
            packed_seq_params = kwargs.get('packed_seq_params')
            if self.visual is not None:
                res = self.visual.get_inputs_embeds(res, **kwargs)
                kwargs.clear()
                if isinstance(res, dict):
                    # compat dict
                    inputs_embeds = res.pop('inputs_embeds')
                    kwargs.update(res)
                    res = inputs_embeds
            if args.context_parallel_size > 1:
                res = split_cp_inputs(res, getattr(packed_seq_params, 'cu_seqlens_q', None), 1)
            if reduce_scatter_embeddings:
                res = res.transpose(0, 1).contiguous()
                group_kwargs = {'group': _self.tp_group} if mcore_013 else {}
                res = reduce_scatter_to_sequence_parallel_region(res, **group_kwargs) / args.tensor_model_parallel_size
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
            kwargs.update({'input_ids': input_ids, 'packed_seq_params': packed_seq_params})
            with self._patch_word_embeddings(kwargs):
                decoder_input = self.language_model.embedding(input_ids=input_ids, position_ids=position_ids)
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = None
            kwargs = {}
        return self.language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            **kwargs,
        )

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        return self.language_model.set_input_tensor(input_tensor)

    def get_input_tensor(self):
        return self.language_model.get_input_tensor()

    def shared_embedding_or_output_weight(self) -> torch.Tensor:
        return self.language_model.shared_embedding_or_output_weight()
