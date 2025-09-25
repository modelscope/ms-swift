# Copyright (c) Alibaba, Inc. and its affiliates.
from contextlib import nullcontext
from typing import List, Optional, Union

import torch
from megatron.core import parallel_state, tensor_parallel
from megatron.core.enums import Fp8Recipe
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.gpt import gpt_model
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import WrappedTensor, deprecate_inference_params, make_viewless_tensor
from megatron.training import get_args
from PIL import Image

from swift.llm import ModelType, to_device
from ..constant import MegatronModelType
from ..gpt.hf2mcore import set_layer_state as set_layer_state_hf2mcore
from ..gpt.mcore2hf import set_layer_state as set_layer_state_mcore2hf
from ..mm_gpt_model import MultimodalGPTModel
from ..register import register_megatron_model
from .utils import HuggingFaceModule, MMGPTMegatronModelMeta

te_checkpoint = None

try:
    import transformer_engine.pytorch as te  # pylint: disable=unused-import

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

if HAVE_TE:
    from megatron.core.extensions.transformer_engine import te_checkpoint


def convert_hf2mcore_qwen3_omni(hf_model, mg_model):
    language_model = hf_model.thinker.model
    mg_language_model = mg_model.language_model
    args = get_args()
    mg_language_model.embedding.word_embeddings.weight.data.copy_(language_model.embed_tokens.weight)
    if args.untie_embeddings_and_output_weights:
        mg_language_model.output_layer.weight.data.copy_(hf_model.thinker.lm_head.weight)
    mg_language_model.decoder.final_layernorm.weight.data.copy_(language_model.norm.weight)
    for layer_idx in range(args.num_layers):
        set_layer_state_hf2mcore(args, mg_language_model, language_model, layer_idx)
    mg_model.visual.thinker.visual.load_state_dict(hf_model.thinker.visual.state_dict())
    mg_model.visual.thinker.audio_tower.load_state_dict(hf_model.thinker.audio_tower.state_dict())


def convert_mcore2hf_qwen3_omni(hf_model, mg_model):
    language_model = hf_model.thinker.model
    mg_language_model = mg_model.language_model
    args = get_args()
    language_model.embed_tokens.weight.data.copy_(mg_language_model.embedding.word_embeddings.weight)
    if args.untie_embeddings_and_output_weights:
        lm_head_weight = (
            hf_model.thinker.score.weight if args.task_type == 'seq_cls' else hf_model.thinker.lm_head.weight)
        lm_head_weight.data.copy_(mg_language_model.output_layer.weight)
    language_model.norm.weight.data.copy_(mg_language_model.decoder.final_layernorm.weight)
    for layer_idx in range(args.num_layers):
        set_layer_state_mcore2hf(args, mg_language_model, language_model, layer_idx)
    hf_model.thinker.visual.load_state_dict(mg_model.visual.thinker.visual.state_dict())
    hf_model.thinker.audio_tower.load_state_dict(mg_model.visual.thinker.audio_tower.state_dict())


class Qwen3Omni_Vit(HuggingFaceModule):
    module_mapping = {
        'thinker': 'thinker',
    }
    _vision_tower = ['thinker.audio_tower', 'thinker.visual']
    _aligner = [
        'thinker.audio_tower.proj1', 'thinker.audio_tower.proj2', 'thinker.visual.merger', 'thinker.visual.merger_list'
    ]

    def __init__(self, config):
        from transformers.models.qwen3_omni_moe import (Qwen3OmniMoeThinkerTextModel,
                                                        Qwen3OmniMoeTalkerForConditionalGeneration,
                                                        Qwen3OmniMoeCode2Wav)
        super().__init__(
            config, [Qwen3OmniMoeThinkerTextModel, Qwen3OmniMoeTalkerForConditionalGeneration, Qwen3OmniMoeCode2Wav])

    def prepare_model(self, hf_model):
        del self.thinker.model
        del self.thinker.lm_head

    @staticmethod
    def _get_inputs_embeds(inputs_embeds, inputs, visual, processor, config):
        from ...trainers.utils import split_cp_inputs
        input_ids = inputs['input_ids']
        packed_seq_params = inputs.get('packed_seq_params')
        pixel_values = inputs.get('pixel_values')
        pixel_values_videos = inputs.get('pixel_values_videos')
        image_grid_thw = inputs.get('image_grid_thw')
        video_grid_thw = inputs.get('video_grid_thw')
        dtype = visual.dtype
        if pixel_values is None and pixel_values_videos is None:  # plain-text
            images = [Image.new('RGB', (32, 32), (0, 0, 0))]
            media_inputs = processor.image_processor(images=images, return_tensors='pt')
            media_inputs = to_device(media_inputs, input_ids.device)
            pixel_values = media_inputs['pixel_values'].type(dtype)
            image_embeds = visual(pixel_values, grid_thw=media_inputs['image_grid_thw'])[0]
            inputs_embeds = inputs_embeds + image_embeds.mean().to(device=inputs_embeds.device) * 0.
            deepstack_visual_embeds = None
            visual_pos_masks = None
        else:
            if pixel_values is None:
                pixel_values_mixed = pixel_values_videos
                grid_thw = video_grid_thw
            elif pixel_values_videos is None:
                pixel_values_mixed = pixel_values
                grid_thw = image_grid_thw
            else:
                pixel_values_mixed = torch.concat([pixel_values, pixel_values_videos], dim=0)
                grid_thw = torch.concat([image_grid_thw, video_grid_thw], dim=0)
            pixel_values_mixed = pixel_values_mixed.type(dtype)
            mixed_embeds, deepstack_visual_embeds = visual(pixel_values_mixed, grid_thw=grid_thw)
            deepstack_visual_embeds = torch.stack(deepstack_visual_embeds, dim=0)
            if pixel_values is None:
                image_embeds = None
                video_embeds = mixed_embeds
            elif pixel_values_videos is None:
                image_embeds = mixed_embeds
                video_embeds = None
            else:
                merge_length = processor.image_processor.merge_size**2
                image_tokens = (image_grid_thw.prod(dim=-1) // merge_length).sum()
                image_embeds = mixed_embeds[:image_tokens]
                video_embeds = mixed_embeds[image_tokens:]

            image_mask = (input_ids == config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            video_mask = (input_ids == config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            if image_embeds is not None:
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                image_mask = image_mask.to(inputs_embeds.device)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if video_embeds is not None:
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                video_mask = video_mask.to(inputs_embeds.device)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
            visual_pos_masks = image_mask[..., 0] | video_mask[..., 0]
            visual_pos_masks = visual_pos_masks.transpose(0, 1)
            # compat cp
            args = get_args()
            if args.context_parallel_size > 1:
                assert packed_seq_params is not None
                device = visual_pos_masks.device
                cp_mask = torch.full(visual_pos_masks.shape[:1], -1, dtype=torch.long, device=device)
                cp_mask[visual_pos_masks[:, 0]] = torch.arange(visual_pos_masks.sum(), device=device)
                cp_mask = split_cp_inputs(cp_mask, packed_seq_params.cu_seqlens_q, 0)
                visual_pos_masks = split_cp_inputs(visual_pos_masks, packed_seq_params.cu_seqlens_q, 0)
                deepstack_visual_embeds = deepstack_visual_embeds[:, cp_mask[(cp_mask != -1)]]
            # compat sp
            tp_world_size = parallel_state.get_tensor_model_parallel_world_size()
            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            if args.sequence_parallel and tp_world_size > 1:
                visual_pos_masks = visual_pos_masks.view(tp_world_size, -1, *visual_pos_masks.shape[1:])
                mask_tokens = visual_pos_masks.sum(dim=(1, 2)).tolist()
                visual_start = 0 if tp_rank == 0 else sum(mask_tokens[:tp_rank])
                visual_end = visual_start + mask_tokens[tp_rank]
                visual_pos_masks = visual_pos_masks[tp_rank]
                deepstack_visual_embeds = deepstack_visual_embeds[:, visual_start:visual_end]
        return {
            'inputs_embeds': inputs_embeds,
            'visual_pos_masks': visual_pos_masks,
            'deepstack_visual_embeds': deepstack_visual_embeds
        }

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        input_ids = kwargs['input_ids']
        visual = self.thinker.visual
        config = self.model_config.thinker_config
        res = self._get_inputs_embeds(inputs_embeds, kwargs, visual, self.processor, config)
        inputs_embeds = res['inputs_embeds']
        input_features = kwargs.get('input_features')
        feature_attention_mask = kwargs.get('feature_attention_mask')

        if input_features is None:
            input_features = input_ids.new_zeros([1, 128, 128], dtype=self.thinker.audio_tower.dtype)
            feature_attention_mask = input_ids.new_ones([1, 128], dtype=torch.bool)
            audio_embeds = self.thinker.get_audio_features(input_features, feature_attention_mask)
            inputs_embeds = inputs_embeds + audio_embeds.mean() * 0.
        else:
            audio_embeds = self.thinker.get_audio_features(input_features, feature_attention_mask)
            audio_mask = (input_ids == config.audio_token_index).unsqueeze(-1).expand_as(inputs_embeds)
            audio_embeds = audio_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_embeds)
        res['inputs_embeds'] = inputs_embeds
        return res


class Qwen3VLTransformerBlock(gpt_model.TransformerBlock):
    # Code borrowed from NVIDIA/Megatron-LM

    def _checkpointed_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        attention_bias: torch.Tensor,
        packed_seq_params: PackedSeqParams,
        use_inner_fp8_context: bool,
        # args for deepstack
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[List[torch.Tensor]] = None,
    ):
        """Forward method with activation checkpointing."""

        def custom(start: int, end: int):

            def custom_forward(hidden_states, attention_mask, context, context_mask, rotary_pos_emb, visual_pos_masks,
                               deepstack_visual_embeds):
                for index in range(start, end):
                    layer = self._get_layer(index)
                    inner_fp8_context = (
                        get_fp8_context(self.config, layer.layer_number
                                        - 1) if use_inner_fp8_context else nullcontext())
                    with inner_fp8_context:
                        hidden_states, context = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            context=context,
                            context_mask=context_mask,
                            rotary_pos_emb=rotary_pos_emb,
                            attention_bias=attention_bias,
                            inference_context=None,
                            packed_seq_params=packed_seq_params,
                        )
                    # add visual features to the hidden states of first several layers
                    layer_number = layer.layer_number - 1
                    if deepstack_visual_embeds is not None and layer_number in range(len(deepstack_visual_embeds)):
                        hidden_states = self._deepstack_process(
                            hidden_states,
                            visual_pos_masks,
                            deepstack_visual_embeds[layer_number],
                        )
                return hidden_states, context

            return custom_forward

        def checkpoint_handler(forward_func):
            """Determines whether to use the `te_checkpoint` or `tensor_parallel.checkpoint`"""
            if self.config.fp8:
                return te_checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    parallel_state.get_tensor_model_parallel_group(),
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                    visual_pos_masks,
                    deepstack_visual_embeds,
                )
            else:
                return tensor_parallel.checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                    visual_pos_masks,
                    deepstack_visual_embeds,
                )

        if self.config.recompute_method == 'uniform':
            # Uniformly divide the total number of Transformer layers and checkpoint
            # the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            layer_idx = 0
            while layer_idx < self.num_layers_per_pipeline_rank:
                hidden_states, context = checkpoint_handler(
                    custom(layer_idx, layer_idx + self.config.recompute_num_layers))

                layer_idx += self.config.recompute_num_layers

        elif self.config.recompute_method == 'block':
            # Checkpoint the input activation of only a set number of individual
            # Transformer layers and skip the rest.
            # A method fully use the device memory removing redundant re-computation.
            recompute_skip_num_layers = 0
            for layer_idx in range(self.num_layers_per_pipeline_rank):
                # Skip recomputation when input grad computation is not needed.
                # Need to have at least one input tensor with gradient computation
                # for re-enterant autograd engine.
                if self.config.fp8 and not hidden_states.requires_grad:
                    recompute_skip_num_layers += 1
                if (layer_idx >= recompute_skip_num_layers
                        and layer_idx < self.config.recompute_num_layers + recompute_skip_num_layers):
                    hidden_states, context = checkpoint_handler(custom(layer_idx, layer_idx + 1))
                else:
                    hidden_states, context = custom(layer_idx, layer_idx + 1)(hidden_states, attention_mask, context,
                                                                              context_mask, rotary_pos_emb)
        else:
            raise ValueError('Invalid activation recompute method.')

        return hidden_states

    def forward(
        self,
        hidden_states: Union[torch.Tensor, WrappedTensor],
        attention_mask: Optional[torch.Tensor],
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        rotary_pos_cos: Optional[torch.Tensor] = None,
        rotary_pos_sin: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[torch.Tensor] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        # args for deepstack
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[List[torch.Tensor]] = None,
    ):
        """
        Perform the forward pass through the transformer block.
        This method handles the core computation of the transformer, including
        self-attention, optional cross-attention, and feed-forward operations.
        Args:
            hidden_states (Union[Tensor, WrappedTensor]): Input tensor of shape [s, b, h]
                where s is the sequence length, b is the batch size, and h is the hidden size.
                Can be passed as a WrappedTensor during inference to avoid an obsolete
                reference in the calling function.
            attention_mask (Tensor): Boolean tensor of shape [1, 1, s, s] for masking
                self-attention.
            context (Tensor, optional): Context tensor for cross-attention.
            context_mask (Tensor, optional): Mask for cross-attention context
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            attention_bias (Tensor): Bias tensor for Q * K.T of shape in shape broadcastable
                to [b, num_head, sq, skv], e.g. [1, 1, sq, skv].
                Used as an alternative to apply attention mask for TE cuDNN attention.
            inference_context (BaseInferenceContext, optional): Parameters for inference-time
                optimizations.
            packed_seq_params (PackedSeqParams, optional): Parameters for packed sequence
                processing.
        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: The output hidden states tensor of shape
            [s, b, h], and optionally the updated context tensor if cross-attention is used.
        """
        if deepstack_visual_embeds is not None:
            assert len(deepstack_visual_embeds) <= len(
                self.layers), (f'len(deepstack_visual_embeds): {len(deepstack_visual_embeds)}, '
                               f'len(self.layers): {len(self.layers)}.')
        inference_context = deprecate_inference_params(inference_context, inference_params)

        # Delete the obsolete reference to the initial input tensor if necessary
        if isinstance(hidden_states, WrappedTensor):
            hidden_states = hidden_states.unwrap()

        if not self.pre_process:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        # Viewless tensor.
        # - We only need to create a viewless tensor in the case of micro batch
        #   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
        #   above creates a view tensor, and '.contiguous()' is a pass-through.
        #   For mbs >= 2, '.contiguous()' creates a new tensor, eliminating
        #   the need to make it viewless.
        #
        #   However, we don't explicitly check mbs == 1 here because
        #   make_viewless_tensor() has negligible overhead when its input
        #   is already viewless.
        #
        # - For the 'else' case above, calling make_viewless_tensor() here is
        #   likely redundant, since p2p_communication.py (likely originator)
        #   already creates viewless tensors. That said, make_viewless_tensor()
        #   is called here to be future-proof and corner-case-proof.
        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        # If fp8_recipe is delayed, wrap the entire pass with get_fp8_context(),
        # otherwise do nothing extra at the outer level
        # if we are using other fp8 recipes, then the context manager enter&exit are free
        # we can wrap fp8_context within the for loop over layers, so that we can fine-grained
        # control which layer will be fp8 or bf16
        use_outer_fp8_context = self.config.fp8 and self.config.fp8_recipe == Fp8Recipe.delayed
        use_inner_fp8_context = self.config.fp8 and self.config.fp8_recipe != Fp8Recipe.delayed
        outer_fp8_context = get_fp8_context(self.config) if use_outer_fp8_context else nullcontext()

        with rng_context, outer_fp8_context:
            # Forward pass.
            if self.config.recompute_granularity == 'full' and self.training:
                hidden_states = self._checkpointed_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                    use_inner_fp8_context=use_inner_fp8_context,
                    visual_pos_masks=visual_pos_masks,
                    deepstack_visual_embeds=deepstack_visual_embeds,
                )
            else:
                for l_no, layer in enumerate(self.layers):
                    inner_fp8_context = (
                        get_fp8_context(self.config, layer.layer_number
                                        - 1) if use_inner_fp8_context else nullcontext())
                    with self.offload_context, inner_fp8_context:
                        hidden_states, context = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            context=context,
                            context_mask=context_mask,
                            rotary_pos_emb=rotary_pos_emb,
                            rotary_pos_cos=rotary_pos_cos,
                            rotary_pos_sin=rotary_pos_sin,
                            attention_bias=attention_bias,
                            inference_context=inference_context,
                            packed_seq_params=packed_seq_params,
                            sequence_len_offset=sequence_len_offset,
                        )
                    # add visual features to the hidden states of first several layers
                    layer_number = layer.layer_number - 1
                    if deepstack_visual_embeds is not None and layer_number in range(len(deepstack_visual_embeds)):
                        hidden_states = self._deepstack_process(
                            hidden_states,
                            visual_pos_masks,
                            deepstack_visual_embeds[layer_number],
                        )

                    if (torch.is_grad_enabled() and self.config.cpu_offloading
                            and self.group_prefetch_offload_commit_async is not None):
                        hidden_states = self.group_prefetch_offload_commit_async(hidden_states)

        # Final layer norm.
        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)
            # TENorm produces a "viewed" tensor. This will result in schedule.py's
            # deallocate_output_tensor() throwing an error, so a viewless tensor is
            # created to prevent this.
            hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        # If this TransformerBlock is empty, input and output hidden states will be the same node
        # on the computational graph and will lead to unexpected errors in pipeline schedules.
        if not self.pre_process and len(self.layers) == 0 and not self.final_layernorm:
            hidden_states = hidden_states.clone()

        return hidden_states

    def _deepstack_process(self, hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor,
                           visual_embeds: torch.Tensor):
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        local_this = hidden_states[visual_pos_masks, :].clone() + visual_embeds
        hidden_states[visual_pos_masks, :] = local_this
        return hidden_states


class Qwen3VLGPTModel(MultimodalGPTModel):

    def _patch_transformer_block(self):
        if hasattr(gpt_model, 'OriginTransformerBlock'):
            return
        gpt_model.OriginTransformerBlock = gpt_model.TransformerBlock
        gpt_model.TransformerBlock = Qwen3VLTransformerBlock

    def __init__(self, *args, **kwargs):
        self._patch_transformer_block()
        super().__init__(*args, **kwargs)


register_megatron_model(
    MMGPTMegatronModelMeta(
        MegatronModelType.qwen3_omni, [
            ModelType.qwen3_omni,
        ],
        convert_hf2mcore=convert_hf2mcore_qwen3_omni,
        convert_mcore2hf=convert_mcore2hf_qwen3_omni,
        model_cls=Qwen3VLGPTModel,
        visual_cls=Qwen3Omni_Vit))


def convert_hf2mcore_qwen3_vl(hf_model, mg_model):
    language_model = hf_model.model.language_model
    mg_language_model = mg_model.language_model
    args = get_args()
    mg_language_model.embedding.word_embeddings.weight.data.copy_(language_model.embed_tokens.weight)
    if args.untie_embeddings_and_output_weights:
        mg_language_model.output_layer.weight.data.copy_(hf_model.lm_head.weight)
    mg_language_model.decoder.final_layernorm.weight.data.copy_(language_model.norm.weight)
    for layer_idx in range(args.num_layers):
        set_layer_state_hf2mcore(args, mg_language_model, language_model, layer_idx)
    mg_model.visual.visual.load_state_dict(hf_model.model.visual.state_dict())


def convert_mcore2hf_qwen3_vl(hf_model, mg_model):
    language_model = hf_model.model.language_model
    mg_language_model = mg_model.language_model
    args = get_args()
    language_model.embed_tokens.weight.data.copy_(mg_language_model.embedding.word_embeddings.weight)
    if args.untie_embeddings_and_output_weights:
        lm_head_weight = hf_model.score.weight if args.task_type == 'seq_cls' else hf_model.lm_head.weight
        lm_head_weight.data.copy_(mg_language_model.output_layer.weight)
    language_model.norm.weight.data.copy_(mg_language_model.decoder.final_layernorm.weight)
    for layer_idx in range(args.num_layers):
        set_layer_state_mcore2hf(args, mg_language_model, language_model, layer_idx)
    hf_model.model.visual.load_state_dict(mg_model.visual.visual.state_dict())


class Qwen3VL_Vit(HuggingFaceModule):
    module_mapping = {'visual': 'visual'}
    _vision_tower = ['visual']
    _aligner = ['visual.merger', 'visual.deepstack_merger_list']

    def __init__(self, config):
        from transformers.models.qwen3_vl import Qwen3VLTextModel
        from transformers.models.qwen3_vl_moe import Qwen3VLMoeTextModel
        super().__init__(config, [Qwen3VLTextModel, Qwen3VLMoeTextModel])

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        return Qwen3Omni_Vit._get_inputs_embeds(inputs_embeds, kwargs, self.visual, self.processor, self.model_config)


register_megatron_model(
    MMGPTMegatronModelMeta(
        MegatronModelType.qwen3_vl, [
            ModelType.qwen3_vl,
            ModelType.qwen3_moe_vl,
        ],
        convert_hf2mcore=convert_hf2mcore_qwen3_vl,
        convert_mcore2hf=convert_mcore2hf_qwen3_vl,
        model_cls=Qwen3VLGPTModel,
        visual_cls=Qwen3VL_Vit))
