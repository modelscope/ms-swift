"""
Circle-RoPE Implementation for Qwen2.5-VL (Based on Latest Transformers)

This implementation adapts Circle-RoPE to the latest transformers architecture where:
- Qwen2_5_VLModel contains both visual and language_model components
- get_rope_index is in Qwen2_5_VLModel (line 956-1139 in new version)
- rope_deltas is cached in Qwen2_5_VLModel (line 939)
- Qwen2_5_VLTextModel handles the language model forward pass (line 769-923)

Key Features:
1. Circle-RoPE position encoding for vision tokens
2. Dual-index mode: supports both circle_index and m_index (original)
3. AGE (Alternating Grouped Encoding) mode for per-layer position encoding

Architecture:
- Qwen2_5_VLConfig_CircleRoPE: Extended config
- Qwen2_5_VLModel_CircleRoPE: Base layer with modified get_rope_index
- Qwen2_5_VLModel_AGE: AGE layer with per-layer position_embeddings
- Qwen2_5_VLForConditionalGeneration_CircleRoPE: Wrapper layer
"""

from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLModel
from transformers.cache_utils import Cache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLCausalLMOutputWithPast,
    Qwen2_5_VLModelOutputWithPast,
    Qwen2_5_VLPreTrainedModel,
)
from transformers.utils import is_torchdynamo_compiling, logging

from .circle_rope_imp import get_circle_rope_index


logger = logging.get_logger(__name__)

# AGE mode strategies: True = use circle_index, False = use m_index (original)
AGE_index_dict = {
    'strategy_2': [True] * 18 + [False] * 18,
    'strategy_3': [False] * 18 + [True] * 18,
    'strategy_4': [True, False] * 18,
}


class Qwen2_5_VLConfig_CircleRoPE(Qwen2_5_VLConfig):
    """Extended config with circle_rope parameter"""
    # Keep model_type same as parent to ensure template compatibility
    model_type = "qwen2_5_vl"

    def __init__(self, circle_rope=None, **kwargs):
        super().__init__(**kwargs)
        self.circle_rope = circle_rope


class Qwen2_5_VLModel_CircleRoPE(Qwen2_5_VLModel):
    """
    Qwen2.5-VL Model with Circle-RoPE support (Base Layer).

    Key changes from original:
    - get_rope_index method supports both circle_rope and original m_index
    - Minimal modification: only vision position calculation is changed
    """
    config_class = Qwen2_5_VLConfig_CircleRoPE

    def _get_circle_index(self, llm_grid_t, llm_grid_h, llm_grid_w, time_tensor):
        """Calculate circle rope index using vertical circular plane mapping"""
        # Reshape time_tensor to 3D grid
        t_index = time_tensor.long().view(-1, llm_grid_h, llm_grid_w)
        h_index = torch.arange(llm_grid_h, device=t_index.device).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w)
        w_index = torch.arange(llm_grid_w, device=t_index.device).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1)

        # Convert ori grid index to vertical circular plane
        llm_pos_ids = get_circle_rope_index(w_index, h_index, t_index, self.config)
        return llm_pos_ids

    def _get_m_index(self, llm_grid_t, llm_grid_h, llm_grid_w, time_tensor):
        """Calculate original Qwen2.5-VL rope index (for reference or AGE mode)"""
        time_tensor_long = time_tensor.long()

        t_index = time_tensor_long.flatten()
        h_index = torch.arange(llm_grid_h, device=t_index.device).view(1, -1, 1).expand(llm_grid_t, -1,
                                                                                        llm_grid_w).flatten()
        w_index = torch.arange(llm_grid_w, device=t_index.device).view(1, 1, -1).expand(llm_grid_t, llm_grid_h,
                                                                                        -1).flatten()

        llm_pos_ids = torch.stack([t_index, h_index, w_index])
        return llm_pos_ids

    def get_rope_index(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
            second_per_grid_ts: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            use_m_index: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index with Circle-RoPE support.

        This method is copied from the latest transformers Qwen2_5_VLModel.get_rope_index
        (line 956-1139) with minimal modifications to support Circle-RoPE.

        Args:
            use_m_index: If True, use original m_index; if False, use circle_index

        Returns:
            position_ids: 3D position indices [3, batch_size, seq_len]
            mrope_position_deltas: Position deltas for caching
        """
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []

        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is not None:
                attention_mask = attention_mask == 1
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            ).to(torch.float)
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                if attention_mask is not None:
                    input_ids = input_ids[attention_mask[i]]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        second_per_grid_t = 0
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image

                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        if second_per_grid_ts is not None:
                            second_per_grid_t = second_per_grid_ts[video_index]
                        else:
                            second_per_grid_t = 1.0
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video

                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                    # normalize type, send to device.
                    second_per_grid_t = torch.as_tensor(
                        second_per_grid_t, dtype=range_tensor.dtype, device=range_tensor.device
                    )

                    time_tensor = expanded_range * second_per_grid_t * self.config.vision_config.tokens_per_second

                    # ==== MODIFICATION: Choose rope index mode (circle_rope or original m_index) ====
                    if use_m_index:
                        llm_pos_ids = self._get_m_index(llm_grid_t, llm_grid_h, llm_grid_w, time_tensor)
                    else:
                        llm_pos_ids = self._get_circle_index(llm_grid_t, llm_grid_h, llm_grid_w, time_tensor)
                    # ==================================================================================

                    llm_pos_ids_list.append(llm_pos_ids + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                if attention_mask is not None:
                    position_ids[..., i, attention_mask[i]] = llm_positions.to(position_ids.device)
                else:
                    position_ids[..., i, :] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas).unsqueeze(1).to(device=input_ids.device)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas


class Qwen2_5_VLModel_AGE(Qwen2_5_VLModel_CircleRoPE):
    """
    Qwen2.5-VL Model with AGE (Alternating Grouped Encoding) mode support.

    AGE mode uses different position_ids for different layers based on AGE strategy.
    This forward method is copied from Qwen2_5_VLModel.forward (line 1216-1326 in new version)
    with modifications to support per-layer position_embeddings.
    """

    def get_pos_id_list(self, input_ids, image_grid_thw, video_grid_thw, second_per_grid_ts, attention_mask):
        circle_position_ids, rope_deltas = self.get_rope_index(
            input_ids,
            image_grid_thw,
            video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            attention_mask=attention_mask,
            use_m_index=False,  # Circle-RoPE
        )
        ori_position_ids, _ = self.get_rope_index(
            input_ids,
            image_grid_thw,
            video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            attention_mask=attention_mask,
            use_m_index=True,  # Original m_index
        )
        self.rope_deltas = rope_deltas

        # Build position_ids_list based on AGE strategy
        if hasattr(self.config, 'circle_rope') and 'AGE_mode' in self.config.circle_rope:
            AGE_mode = self.config.circle_rope['AGE_mode']
            AGE_index = AGE_index_dict[AGE_mode]
            position_ids_list = [
                circle_position_ids if flag else ori_position_ids
                for flag in AGE_index
            ]
        else:
            # Fallback: use circle_position_ids for all layers
            position_ids_list = [circle_position_ids] * self.config.text_config.num_hidden_layers
        position_ids = circle_position_ids  # Use circle for default

        return position_ids_list, position_ids

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            pixel_values: Optional[torch.Tensor] = None,
            pixel_values_videos: Optional[torch.FloatTensor] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
            rope_deltas: Optional[torch.LongTensor] = None,
            cache_position: Optional[torch.LongTensor] = None,
            second_per_grid_ts: Optional[torch.Tensor] = None,
            position_ids_list: Optional[List[torch.LongTensor]] = None,
            **kwargs,
    ) -> Union[Tuple, Qwen2_5_VLModelOutputWithPast]:
        """
        Forward pass with AGE mode support.

        Key difference from parent: Uses different position_ids for different layers (AGE mode).

        Args:
            position_ids_list: List of position_ids for each layer (provided by CircleRoPETemplate).
                              If None, falls back to using the same position_ids for all layers.

        Note:
            When using CircleRoPEQwen2_5VLTemplate, position_ids_list is automatically generated
            in the template's _data_collator stage, before input_ids is converted to inputs_embeds.
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # ========== Copied from Qwen2_5_VLModel.forward (line 1253-1270) ==========
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        # ========== MODIFICATION: Initialize position_ids_list if not provided ==========
        # If CircleRoPETemplate is used, position_ids_list is already generated in _data_collator
        # If position_ids_list is None, fall back to standard behavior (all layers use same position_ids)
        if position_ids_list is None:
            if position_ids is not None:
                # Use the same position_ids for all layers
                position_ids_list = [position_ids] * self.config.text_config.num_hidden_layers
            else:
                # Generate position_ids using parent's logic (for non-template usage)
                # This is a fallback - normally template should provide position_ids
                prefill_stage = (
                        (cache_position is not None and cache_position[0] == 0)
                        or (past_key_values is None or past_key_values.get_seq_length() == 0)
                )
                if prefill_stage or self.rope_deltas is None:
                    # Generate circle_rope position_ids
                    position_ids, rope_deltas = self.get_rope_index(
                        input_ids,
                        image_grid_thw,
                        video_grid_thw,
                        second_per_grid_ts=second_per_grid_ts,
                        attention_mask=attention_mask,
                        use_m_index=False,  # Circle-RoPE
                    )
                    self.rope_deltas = rope_deltas
                    position_ids_list = [position_ids] * self.config.text_config.num_hidden_layers
                else:
                    # Use cached rope_deltas for generation
                    batch_size, seq_length, _ = inputs_embeds.shape
                    position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                    position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
                    if cache_position is not None:
                        delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    else:
                        delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
                    position_ids = position_ids + delta.to(position_ids.device)
                    position_ids_list = [position_ids] * self.config.text_config.num_hidden_layers

        # ========== AGE-specific: Manual layer iteration instead of calling language_model.forward ==========
        # Prepare cache_position (from TextModel.forward line 830-834)
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # Extract text_position_ids for mask generation (from TextModel.forward line 852-857)
        if position_ids_list[0] is not None and position_ids_list[0].ndim == 3 and position_ids_list[0].shape[0] == 4:
            text_position_ids = position_ids_list[0][0]
            position_ids_list = [pos[1:] if pos is not None else None for pos in position_ids_list]
        else:
            text_position_ids = None

        # Create causal_mask_mapping (from TextModel.forward line 860-876)
        if not isinstance(attention_mask, dict):
            mask_kwargs = {
                "config": self.config.text_config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": text_position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if self.language_model.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
        else:
            causal_mask_mapping = attention_mask

        # Manual layer iteration (from TextModel.forward line 883-906)
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for layer_idx, decoder_layer in enumerate(self.language_model.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # AGE modification: Generate position_embeddings per layer
            _position_ids = position_ids_list[layer_idx] if layer_idx < len(position_ids_list) else position_ids
            position_embeddings = self.language_model.rotary_emb(hidden_states, _position_ids)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # Apply norm (from TextModel.forward line 908)
        hidden_states = self.language_model.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Return with rope_deltas (from Qwen2_5_VLModel.forward line 1319-1326)
        output = Qwen2_5_VLModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            rope_deltas=self.rope_deltas,
        )
        return output if return_dict else output.to_tuple()


class Qwen2_5_VLForConditionalGeneration_CircleRoPE(Qwen2_5_VLForConditionalGeneration):
    """
    Qwen2.5-VL For Conditional Generation with Circle-RoPE support.

    Compatible with latest transformers architecture.
    Automatically selects between standard CircleRoPE and AGE mode based on config.

    Forward method is copied from parent class (line 1405-1514) with no modifications.
    AGE logic is handled in Qwen2_5_VLModel_AGE.forward.
    """
    config_class = Qwen2_5_VLConfig_CircleRoPE
    _checkpoint_conversion_mapping = {
        "^visual": "model.visual",
        r"^model(?!\.(language_model|visual))": "model.language_model",
    }
    _tied_weights_keys = ["lm_head.weight"]
    accepts_loss_kwargs = False

    def __init__(self, config):
        # Initialize PreTrainedModel base (skip parent __init__)
        Qwen2_5_VLPreTrainedModel.__init__(self, config)

        # Create Circle-RoPE model based on config
        if hasattr(config, 'circle_rope') and config.circle_rope and 'AGE_mode' in config.circle_rope:
            self.model = Qwen2_5_VLModel_AGE(config)
        else:
            self.model = Qwen2_5_VLModel_CircleRoPE(config)

        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    # Make modules available through conditional class for BC
    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    # ========== Copied from Qwen2_5_VLForConditionalGeneration.forward (line 1405-1514) ==========
    # NO MODIFICATIONS - AGE logic is handled in Qwen2_5_VLModel_AGE.forward
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            pixel_values: Optional[torch.Tensor] = None,
            pixel_values_videos: Optional[torch.FloatTensor] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
            rope_deltas: Optional[torch.LongTensor] = None,
            cache_position: Optional[torch.LongTensor] = None,
            second_per_grid_ts: Optional[torch.Tensor] = None,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            position_ids_list=None,
            **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            position_ids_list=position_ids_list,
            **kwargs,
        )

        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )


__all__ = [
    "Qwen2_5_VLConfig_CircleRoPE",
    "Qwen2_5_VLModel_CircleRoPE",
    "Qwen2_5_VLModel_AGE",
    "Qwen2_5_VLForConditionalGeneration_CircleRoPE",
    "AGE_index_dict",
]
