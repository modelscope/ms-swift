import torch
from megatron.core import InferenceParams, mpu
from megatron.core.packed_seq_params import PackedSeqParams
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

        args = get_args()
        self.visual = None
        if args.megatron_model_meta.visual_cls is not None:
            self.visual = args.megatron_model_meta.visual_cls(config)

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
        args = get_args()
        if decoder_input is not None:
            pass
        elif self.pre_process:
            decoder_input = self.language_model.embedding(input_ids=input_ids, position_ids=position_ids)
            if self.visual is not None:
                if args.tensor_model_parallel_size > 1 and args.sequence_parallel:
                    input_ids = input_ids.chunk(
                        args.tensor_model_parallel_size, dim=-1)[mpu.get_tensor_model_parallel_rank()]
                kwargs.update({'input_ids': input_ids})
                decoder_input = decoder_input.transpose(0, 1)
                decoder_input = self.visual.get_inputs_embeds(decoder_input, **kwargs)
                decoder_input = decoder_input.transpose(0, 1)
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
