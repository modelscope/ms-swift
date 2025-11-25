"""
Circle-RoPE Custom Template for ms-swift

This template extends Qwen2_5VLTemplate to support Circle-RoPE and AGE mode.
It overrides _get_position_ids to generate position_ids_list for AGE mode in the data_collator stage,
before input_ids is converted to inputs_embeds.
"""
import inspect
from typing import Any, Dict, List, Optional
import torch
from packaging import version
from torch import nn

from swift.llm import to_device
from swift.llm.template.template.qwen import Qwen2_5VLTemplate
from .modular_qwen2_5_vl_circle_rope import AGE_index_dict


class CircleRoPEQwen2_5VLTemplate(Qwen2_5VLTemplate):
    """
    Custom template for Circle-RoPE with AGE mode support.

    This template ensures that:
    1. position_ids are generated in _data_collator stage (when input_ids is still available)
    2. For AGE mode, generates position_ids_list with different position_ids per layer
    3. For standard mode, generates circle_rope position_ids

    Usage:
        Register this template and use it with --template_type qwen2_5_vl_circle_rope
    """

    def pre_forward_hook(self, model: nn.Module, args, kwargs):
        old_kwargs = to_device(kwargs, model.device)
        kwargs = to_device(self._post_encode(model, old_kwargs), model.device)
        for k, v in old_kwargs.items():
            if k in {
                'input_ids', 'attention_mask', 'labels', 'position_ids', 'output_hidden_states', 'logits_to_keep',
                'max_length_q', 'max_length_k', 'cu_seq_lens_q', 'cu_seq_lens_k',
                'position_ids_list'
            } and k not in kwargs:
                kwargs[k] = v

        if 'inputs_embeds' in kwargs:
            kwargs.pop('input_ids', None)

        base_model = self.get_base_model(model)
        parameters = inspect.signature(base_model.forward).parameters
        if 'position_ids' not in parameters:
            kwargs.pop('position_ids', None)
        return args, kwargs

    def _get_position_ids(self, inputs: Dict[str, Any]):
        """
        Generate position_ids for Circle-RoPE, with AGE mode support.

        This method is called in _data_collator (line 448 in qwen.py) during training,
        at which point input_ids is still available (before _post_encode converts it to inputs_embeds).

        For AGE mode:
            Returns stacked position_ids: [num_layers+1, 3, batch, seq_len]
            - First num_layers: position_ids for each layer (alternating circle/original)
            - Last one: text_position_ids (used for mask generation)

        For standard Circle-RoPE mode:
            Returns standard 3D position_ids: [3, batch, seq_len] (concatenated with text_position_ids)
        """
        base_model = self.get_base_model(self._get_model())

        # Get the actual model that has get_rope_index
        # For CircleRoPE, it's either base_model or base_model.model
        if hasattr(base_model, 'get_rope_index'):
            rope_model = base_model
        elif hasattr(base_model, 'model') and hasattr(base_model.model, 'get_rope_index'):
            rope_model = base_model.model
        else:
            # Fallback to parent implementation if no Circle-RoPE model found
            return super()._get_position_ids(inputs)

        # Check if AGE mode is enabled
        config = rope_model.config
        is_age_mode = (hasattr(config, 'circle_rope') and
                       config.circle_rope and
                       'AGE_mode' in config.circle_rope)

        # Prepare kwargs for get_rope_index
        kwargs = {}
        if self.version == 'v2_5':
            kwargs = {'second_per_grid_ts': inputs.get('second_per_grid_ts')}

        attention_mask = inputs.get('attention_mask_2d')
        if attention_mask is None:
            attention_mask = inputs.get('attention_mask')

        if is_age_mode:
            # ========== AGE Mode: Generate position_ids_list ==========
            # Generate both circle and original position_ids
            circle_position_ids, _ = rope_model.get_rope_index(
                inputs['input_ids'],
                inputs.get('image_grid_thw'),
                inputs.get('video_grid_thw'),
                attention_mask=attention_mask,
                use_m_index=False,  # Circle-RoPE
                **kwargs)

            ori_position_ids, _ = rope_model.get_rope_index(
                inputs['input_ids'],
                inputs.get('image_grid_thw'),
                inputs.get('video_grid_thw'),
                attention_mask=attention_mask,
                use_m_index=True,  # Original m_index
                **kwargs)

            # Build position_ids_list based on AGE strategy
            AGE_mode = config.circle_rope['AGE_mode']
            AGE_index = AGE_index_dict[AGE_mode]

            # Create position_ids for each layer and concat with text_position_ids
            position_ids_list_with_text = [
                self._concat_text_position_ids(circle_position_ids if flag else ori_position_ids)
                for flag in AGE_index
            ]

            # Stack all position_ids with text_position_ids prepended
            # Shape: [num_layers, 4, batch, seq_len]
            # - [i, 0]: text_position_ids for layer i
            # - [i, 1:]: mrope position_ids (3 dims) for layer i
            all_position_ids = torch.stack(position_ids_list_with_text, dim=0)

            return all_position_ids
        else:
            # ========== Standard Circle-RoPE Mode ==========
            # Generate circle_rope position_ids (use_m_index=False)
            position_ids, _ = rope_model.get_rope_index(
                inputs['input_ids'],
                inputs.get('image_grid_thw'),
                inputs.get('video_grid_thw'),
                attention_mask=attention_mask,
                use_m_index=False,  # Circle-RoPE
                **kwargs)

            # Concatenate with text_position_ids (same as parent)
            return self._concat_text_position_ids(position_ids)

    def packing_row(self, row: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Override packing_row to handle AGE mode's 4D position_ids.

        For packing, we need to:
        1. Generate position_ids for each sample
        2. Concatenate them along seq_len dimension
        3. For AGE mode (4D), extract position_ids_list immediately

        Standard mode: [4, 1, seq_len] -> concat -> [4, 1, total_seq_len]
        AGE mode: [num_layers, 4, 1, seq_len] -> concat -> [num_layers, 4, 1, total_seq_len]
        """
        position_ids = []
        for r in row:
            r = r.copy()
            r['input_ids'] = torch.tensor(r['input_ids'])[None]
            position_ids.append(self._get_position_ids(r))

        # Call grandparent's packing_row (Qwen2VLTemplate's parent = base.py)
        # to pack input_ids, labels, etc.
        packed = super(Qwen2_5VLTemplate, self).packing_row(row)

        # Concatenate position_ids along the last dimension (seq_len)
        concatenated = torch.concat(position_ids, dim=-1)

        # Check if AGE mode (4D)
        if concatenated.ndim == 4:
            # AGE mode: [num_layers, 4, 1, total_seq_len]
            num_layers = concatenated.shape[0]

            # Extract position_ids_list for each layer (without text_position_ids)
            # Squeeze batch dimension as it's always 1 in packing
            packed['position_ids_list'] = [concatenated[i, 1:].squeeze(1) for i in range(num_layers)]

            # Extract text_position_ids (all layers have same text_position_ids)
            # Squeeze batch dimension
            packed['text_position_ids'] = concatenated[0, 0].squeeze(0)  # [total_seq_len]

            # Default position_ids (for non-AGE code paths) use first layer
            # Squeeze batch dimension
            # packed['position_ids'] = concatenated[0, 1:].squeeze(1)  # [3, total_seq_len]
            packed['position_ids'] = concatenated[0, ...].squeeze(1)  # [3, total_seq_len]

        else:
            # Standard mode: [4, 1, total_seq_len]
            # Don't process here, let parent's _data_collator handle it
            packed['position_ids'] = concatenated

        return packed

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        """
        Override data_collator to handle position_ids_list for AGE mode.

        Two scenarios:
        1. Packing mode (padding_free=True):
           - position_ids_list, text_position_ids, position_ids already extracted in packing_row
           - Just need to ensure they're passed through correctly

        2. Non-packing mode (padding_free=False):
           - position_ids is 4D: [num_layers, 4, batch, seq_len] (AGE)
           - position_ids is 3D: [4, batch, seq_len] (standard)
           - Extract position_ids_list, text_position_ids, position_ids
        """
        # # Check if packing mode and position_ids_list already exists
        # if self.padding_free and len(batch) == 1 and 'position_ids_list' in batch[0]:
        #     # AGE mode with packing: position_ids_list already extracted in packing_row
        #     # Just call parent to handle other fields
        #     res = super()._data_collator(batch, padding_to=padding_to)
        #     # position_ids_list, text_position_ids should already be in res
        #     return res

        # Non-packing mode or standard mode: use existing logic
        res = super(Qwen2_5VLTemplate, self)._data_collator(batch, padding_to=padding_to)

        # if 'position_ids' in res:
        # position_ids = batch['position_ids']
        #
        # # Check if it's AGE mode (4D stacked position_ids)
        # if position_ids.ndim == 4:
        #     # AGE Mode: [num_layers, 4, batch, seq_len]
        #     num_layers = position_ids.shape[0]
        #
        #     # Extract position_ids_list for each layer (mrope part: 3 dims)
        #     res['position_ids_list'] = [position_ids[i, 1:] for i in range(num_layers)]
        #
        #     # Extract text_position_ids from first layer (all layers have same text_position_ids)
        #     res['text_position_ids'] = text_position_ids = position_ids[0, 0]  # [batch, seq_len]
        #
        #     # Default position_ids (for non-AGE code paths) use first layer
        #     res['position_ids'] = position_ids[0, 1:]  # [3, batch, seq_len]
        # else:
        #     # Standard mode: 3D [4, batch, seq_len] (with text prepended)
        #     # Parent class handles this, but we need to ensure correct extraction
        #     if position_ids.shape[0] == 4:
        #         # With text_position_ids prepended
        #         res['position_ids'] = position_ids[1:]  # [3, batch, seq_len]
        #         res['text_position_ids'] = text_position_ids = position_ids[0]
        #     elif position_ids.shape[0] == 3:
        #         # Already extracted by parent
        #         pass
        #
        # # Handle packed sequence params if needed
        # if 'text_position_ids' in res:
        #     text_position_ids = res['text_position_ids']
        #     if self.transformers_version >= version.parse('4.53.0.dev') and text_position_ids.shape[0] == 1:
        #         # https://github.com/huggingface/transformers/pull/40194
        #         from swift.llm.template.utils import get_packed_seq_params
        #         res.update(get_packed_seq_params(text_position_ids))

        _position_ids_list = []
        _position_ids_list_list = []
        _text_position_ids_list = []
        for data in batch:
            _position_ids_list.append(data['position_ids'])
            _position_ids_list_list.append(data['position_ids_list'])  # Keep as list
            _text_position_ids_list.append(data['text_position_ids'])

        # Stack position_ids and text_position_ids across batch dimension
        res['position_ids'] = torch.stack(_position_ids_list, dim=1)
        res['text_position_ids'] = torch.stack(_text_position_ids_list, dim=1)

        # For position_ids_list, we need to transpose to [num_layers] x [3, batch, seq_len]
        # Currently _position_ids_list_list is [batch] x [num_layers] x [3, seq_len]
        # We want [num_layers] x [3, batch, seq_len]
        num_layers = len(_position_ids_list_list[0])
        position_ids_list = []
        for layer_idx in range(num_layers):
            # Collect position_ids for this layer from all batch items
            layer_position_ids = [batch_item[layer_idx] for batch_item in _position_ids_list_list]
            # Stack along batch dimension: [3, seq_len] x batch -> [3, batch, seq_len]
            position_ids_list.append(torch.stack(layer_position_ids, dim=1))

        res['position_ids_list'] = position_ids_list  # List of tensors, not stacked tensor

        return res
