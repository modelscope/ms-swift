# Copyright (c) ModelScope Contributors. All rights reserved.
"""Patch a Qwen3-TTS model's ``forward`` to implement dual-channel training loss.

Qwen3-TTS (from the external ``qwen_tts`` package, registered into transformers' Auto classes) ships
an inference-oriented ``forward`` with no trainable loss path. This patch replaces ``forward`` with a
training variant that mirrors legacy swift's ``_patch_qwen3_tts_forward`` (swift/model/models/qwen.py):

* ``input_ids`` is dual-channel ``[B, T, 2]`` -- channel 0 is text tokens, channel 1 is codec tokens;
  they are embedded separately, masked, summed, with the speaker embedding injected at position 6.
* sub-talker codec embeddings (code-predictor layers 1..15) are added into the input embeddings.
* the ``talker`` computes the main codec-0 loss; a sub-talker cross-entropy is computed from the
  talker hidden states at codec positions and added with weight ``sub_talker_loss_weight`` (0.3).

The replacement is reversible via ``unpatch``, so the model's native inference ``forward`` is
restored on context exit -- unlike swift legacy's permanent ``MethodType`` replacement.
"""
from types import MethodType

from twinkle.patch import Patch

_MARKER = '_swift_qwen3_tts_patched'
# Speaker embedding is injected at this fixed codec position; sub-talker uses code-predictor
# embedding layers 1..15. Both are properties of the Qwen3-TTS architecture, not tunables.
_SPEAKER_POSITION = 6
_SUB_TALKER_LAYERS = range(1, 16)


class Qwen3TTSTrainingPatch(Patch):
    """Swap Qwen3-TTS ``model.forward`` for a dual-channel training forward. Reversible via ``unpatch``.

    ``sub_talker_loss_weight`` scales the auxiliary sub-talker cross-entropy added to the main loss.
    """

    def __init__(self, sub_talker_loss_weight: float = 0.3):
        self.sub_talker_loss_weight = sub_talker_loss_weight
        self._model = None
        self._origin_forward = None

    def __call__(self, module, *args, **kwargs):
        if getattr(module, _MARKER, False):
            return module
        import torch.nn.functional as F

        # Save the original bound forward BEFORE mutation so unpatch restores it verbatim.
        self._model = module
        self._origin_forward = module.forward
        sub_talker_loss_weight = self.sub_talker_loss_weight

        def tts_forward(self,
                        input_ids=None,
                        attention_mask=None,
                        speaker_embedding=None,
                        text_embedding_mask=None,
                        codec_embedding_mask=None,
                        codec_0_labels=None,
                        codec_ids=None,
                        codec_mask=None,
                        **kwargs):
            # Separate dual-channel input_ids: channel 0 text, channel 1 codec.
            input_text_ids = input_ids[:, :, 0]
            input_codec_ids = input_ids[:, :, 1]

            # Build text and codec embeddings.
            input_text_embedding = self.talker.text_projection(
                self.talker.model.text_embedding(input_text_ids)) * text_embedding_mask
            input_codec_embedding = self.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
            # Inject speaker embedding at the fixed codec position.
            input_codec_embedding[:, _SPEAKER_POSITION, :] = speaker_embedding

            # Sum text and codec embeddings.
            input_embeddings = input_text_embedding + input_codec_embedding

            # Add sub-talker codec embeddings (code-predictor layers 1..15).
            for i in _SUB_TALKER_LAYERS:
                codec_i_embedding = self.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                input_embeddings = input_embeddings + codec_i_embedding

            outputs = self.talker(
                inputs_embeds=input_embeddings,
                attention_mask=attention_mask,
                labels=codec_0_labels,
                output_hidden_states=True,
            )

            # Compute sub_talker_loss from hidden states at codec positions.
            hidden_states = outputs.hidden_states[0][-1][:, :-1, :]
            talker_hidden_states = hidden_states[codec_mask[:, 1:]]
            talker_codec_ids = codec_ids[codec_mask]

            sub_talker_logits, _ = self.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)
            sub_talker_loss = F.cross_entropy(
                sub_talker_logits.reshape(-1, sub_talker_logits.shape[-1]).float(),
                talker_codec_ids[:, 1:].reshape(-1).to(sub_talker_logits.device),
            )

            outputs['loss'] = outputs.loss + sub_talker_loss_weight * sub_talker_loss
            return outputs

        module.forward = MethodType(tts_forward, module)
        setattr(module, _MARKER, True)
        return module

    def unpatch(self, module, *args, **kwargs):
        origin = getattr(self, '_origin_forward', None)
        if origin is not None:
            module.forward = origin
        if hasattr(module, _MARKER):
            delattr(module, _MARKER)
        self._origin_forward = None
        self._model = None
        return module
