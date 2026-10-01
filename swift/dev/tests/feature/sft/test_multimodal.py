# Copyright (c) ModelScope Contributors. All rights reserved.
"""End-to-end multimodal SFT training (image-text causal_lm, image-text embedding, plus audio).

``test_task_types.py`` covers the non-causal-lm task_types on a text model; this file covers the
multimodal *inputs* -- a vision-language model that must carry ``pixel_values`` / ``image_grid_thw``
from the encoded row through dev's collate into the forward, for BOTH ``causal_lm`` (an ``<image>``
user turn + assistant response) and ``embedding`` (an ``<image>`` anchor plus ``<image>`` positive /
negative candidates, the multimodal InfoNCE shape), and (audio) a model that must carry
``input_features`` the same way. The assertion is the task-type-agnostic trained contract from
``conftest.assert_trained``: optimizer steps ran, the loss stayed finite and normalized, and a
self-describing checkpoint was written.

Weights are a tiny random-init checkpoint of the real architecture, built through dev's own family
loader (``build_tiny_multimodal``) -- the same convention ``tests/infer/test_multimodal_e2e.py``
established. Only config/tokenizer/processor files are downloaded (never the multi-GB weights), so the
test exercises the real vision/audio tower wiring, the real template media encoding and the real
collate without a full model download. This is not a mock: a VL model given ``<image>`` placeholder
tokens in ``input_ids`` but no ``pixel_values`` raises on the token/patch-count mismatch, so a clean,
finite-loss training run means the image tensors really reached the forward.

All tests are ``@pytest.mark.slow`` + ``@pytest.mark.accel(1)``; run with
``CUDA_VISIBLE_DEVICES=<card> pytest swift/dev/tests/feature/sft/test_multimodal.py -m slow``.
"""
import json

import pytest
import torch

from swift.dev.tests.tiny_loader import build_tiny_multimodal

pytestmark = [pytest.mark.slow, pytest.mark.accel(1)]

#: causal_lm cross-entropy over the ~151k Qwen vocab settles near ln(vocab) ~= 11.9 for a random-init
#: tiny model, so 20.0 sits above a correctly mean-reduced run yet far below a sum-reduced one (which
#: scales by the sequence length, into the hundreds) or a diverged run.
_CAUSAL_LM_MAX_LOSS = 20.0

VL_MODEL_TYPE = 'qwen2_5_vl'
VL_TEMPLATE = 'qwen2_5_vl'

#: Audio (ASR-style) would ride Qwen2.5-Omni: the only audio-capable Qwen family the environment can
#: load. Qwen2-Audio is gated to ``transformers<4.49`` (env is 5.12) and Qwen3-ASR/TTS need the
#: ``qwen-asr``/``qwen-tts`` packages, neither installed; Qwen2.5-Omni's deps (soundfile, decord,
#: qwen_omni_utils) are all present, its tiny checkpoint builds, and its thinker keeps the
#: ``.visual``/``.audio_tower``/``get_audio_features`` layout the shared template's ``_post_encode``
#: reads -- so it loads, encodes, and (now) trains.
OMNI_MODEL_TYPE = 'qwen2_5_omni'
OMNI_TEMPLATE = 'qwen2_5_omni'
OMNI_SAMPLING_RATE = 16000

#: FIXED (twinkle audio collate). Previously twinkle's ``_collate_macro_batch`` implemented only ONE
#: audio contract -- gemma4's channels-last ``input_features`` ``[plane0, plane1] -> [plane0, plane1, 1]``
#: reshape keyed on the ``input_features_mask`` field (mcore-bridge mm_gpts/gemma4.py). Qwen2.5-Omni
#: instead emits ``input_features`` ``[num_audios, freq, time]`` + ``feature_attention_mask``
#: ``[num_audios, time]`` and transformers' ``Qwen2_5OmniThinker.get_audio_features`` needs
#: ``[batch, freq, time]`` (plain dim-0 concat). The blanket ``.squeeze()`` stripped Omni's leading
#: audio dim and the channels-last reshape folded the batch into dim 0, handing get_audio_features a
#: ``[batch*freq, time, 1]`` tensor whose mask scatter raised ``IndexError``. The two contracts are told
#: apart by their mask field name (``input_features_mask`` = gemma4 channels-last vs
#: ``feature_attention_mask`` = Qwen-audio batched); twinkle now skips the squeeze and the reshape on
#: the Qwen-audio contract and leaves the gemma4 path byte-identical. This test is the Qwen-audio
#: regression guard; the gemma4 channels-last path is guarded by a collate-level component test.


@pytest.fixture(scope='module')
def vl_model(tmp_path_factory):
    """One tiny VL checkpoint for the whole module -- the snapshot download and random init are shared."""
    dest = tmp_path_factory.mktemp('vl') / 'model'
    return build_tiny_multimodal(str(dest), model_type=VL_MODEL_TYPE, model_id='Qwen/Qwen2.5-VL-3B-Instruct')


@pytest.fixture(scope='module')
def omni_model(tmp_path_factory):
    """One tiny Qwen2.5-Omni checkpoint for the whole module (thinker + talker, random weights)."""
    dest = tmp_path_factory.mktemp('omni') / 'model'
    return build_tiny_multimodal(str(dest), model_type=OMNI_MODEL_TYPE, model_id='Qwen/Qwen2.5-Omni-3B')


def _write_image(path, size=56):
    """A tiny random RGB image on disk, big enough for one vision patch grid."""
    import numpy as np
    from PIL import Image
    Image.fromarray((np.random.rand(size, size, 3) * 255).astype('uint8')).save(path)
    return str(path)


def _write_vl_sft_data(path, image_paths):
    """A jsonl SFT dataset: each row pairs an ``<image>`` user turn with an assistant response.

    The assistant turn is what supplies the training labels; the ``images`` list runs parallel to the
    ``<image>`` placeholders in the conversation, the documented dev/legacy multimodal row shape.
    """
    rows = [{
        'messages': [
            {
                'role': 'user',
                'content': '<image>这张图里有什么？'
            },
            {
                'role': 'assistant',
                'content': '这是一张随机生成的彩色图片。'
            },
        ],
        'images': [img],
    } for img in image_paths]
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    return str(path)


def test_run_sft_vl_image_text_end_to_end(tmp_path, vl_model, assert_trained):
    """Image-text causal_lm SFT: the encoded ``pixel_values`` reach the forward and a checkpoint trains.

    Four image-bearing rows, batch 2, three steps -- enough to exercise a padded multi-image batch
    through dev's collate. A text-only path would raise on the image placeholder tokens, so a finite,
    normalized loss is itself proof the vision tensors were consumed.
    """
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    from swift.dev.config import (CheckpointConfig, DatasetConfig, DistributedConfig, ModelConfig, TemplateConfig,
                                  TrainConfig)
    from swift.dev.recipe import run_sft

    imgs = [_write_image(tmp_path / f'img{i}.png') for i in range(4)]
    data = _write_vl_sft_data(tmp_path / 'vl_sft.jsonl', imgs)
    out_dir = str(tmp_path / 'out_vl_sft')

    history = run_sft(
        ModelConfig(model=vl_model, model_type=VL_MODEL_TYPE, task_type='causal_lm', torch_dtype='bfloat16'),
        TemplateConfig(template=VL_TEMPLATE, max_length=512),
        DatasetConfig(dataset=[data], dataset_shuffle=False),
        TrainConfig(learning_rate=1e-5, per_device_train_batch_size=2, gradient_accumulation_steps=1, max_steps=3),
        DistributedConfig(),
        CheckpointConfig(),
        output_dir=out_dir,
    )
    assert_trained(history, out_dir, 'causal_lm', max_loss=_CAUSAL_LM_MAX_LOSS)


def _write_vl_embedding_data(path, anchor_images, negative_images):
    """A jsonl multimodal-embedding dataset: an ``<image>`` anchor plus an ``<image>`` positive/negative.

    The documented dev/legacy multimodal-embedding row shape (docs/source/BestPractices/Embedding.md):
    ``<image>`` may appear in ``messages`` / ``positive_messages`` / ``negative_messages``, each with its
    own parallel media column -- ``images`` / ``positive_images`` / ``negative_images``. The positive and
    negative media columns are list-of-list: the outer list aligns with the candidate conversations (one
    positive is the documented constraint), the inner list with the ``<image>`` tags in that conversation.
    The anchor and its positive share an image (a genuine match for InfoNCE); the negative is a different
    one, so the contrastive objective has a real signal rather than noise.
    """
    rows = [{
        'messages': [{
            'role': 'user',
            'content': '<image>这张图的表示是什么？'
        }],
        'images': [anchor],
        'positive_messages': [[{
            'role': 'user',
            'content': '<image>同一张图。'
        }]],
        'positive_images': [[anchor]],
        'negative_messages': [[{
            'role': 'user',
            'content': '<image>另一张不同的图。'
        }]],
        'negative_images': [[negative]],
    } for anchor, negative in zip(anchor_images, negative_images)]
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    return str(path)


def test_run_embedding_multimodal_end_to_end(tmp_path, vl_model, assert_trained):
    """Image-text embedding (InfoNCE): the anchor / positive / negative ``pixel_values`` all reach the
    forward and a checkpoint trains.

    This is the multimodal-embedding dimension the text-only ``test_run_embedding_end_to_end`` cannot
    reach: each of the three candidates carries its own ``<image>`` and its own media column
    (``images`` / ``positive_images`` / ``negative_images``). It guards two seams that are easy to drop
    silently -- the dataset preprocessor must carry the positive/negative media columns through to the
    template (its ``cast_mm_data`` normalizes only ``images``/``rejected_images``), and the embedding
    collate must interleave the anchor/positive/negative vision tensors alongside their input_ids. A
    text-only path would raise on the un-filled ``<image>`` placeholder tokens, so a finite, normalized
    InfoNCE loss is itself proof the vision tensors were consumed for all three candidates.
    """
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    from swift.dev.config import (CheckpointConfig, DatasetConfig, DistributedConfig, ModelConfig, TemplateConfig,
                                  TrainConfig)
    from swift.dev.recipe import run_embedding

    # 4 anchors, each contrasted against a distinct negative image (the positive reuses the anchor image).
    anchors = [_write_image(tmp_path / f'emb_anchor{i}.png') for i in range(4)]
    negatives = [_write_image(tmp_path / f'emb_neg{i}.png') for i in range(4)]
    data = _write_vl_embedding_data(tmp_path / 'vl_embedding.jsonl', anchors, negatives)
    out_dir = str(tmp_path / 'out_vl_embedding')

    history = run_embedding(
        ModelConfig(model=vl_model, model_type=VL_MODEL_TYPE, task_type='embedding', torch_dtype='bfloat16'),
        TemplateConfig(template=VL_TEMPLATE, max_length=512),
        DatasetConfig(dataset=[data], dataset_shuffle=False),
        TrainConfig(
            learning_rate=1e-5,
            loss='infonce',
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            max_steps=3),
        DistributedConfig(),
        CheckpointConfig(),
        output_dir=out_dir,
    )
    assert_trained(history, out_dir, 'embedding', max_loss=20.0)


def _write_audio(path, seconds=1.0, sampling_rate=OMNI_SAMPLING_RATE):
    """A tiny mono wav on disk: enough samples for the audio tower's feature extractor to emit patches."""
    import numpy as np
    import soundfile as sf
    samples = (np.random.rand(int(seconds * sampling_rate)) * 2 - 1).astype('float32') * 0.01
    sf.write(str(path), samples, sampling_rate)
    return str(path)


def _write_audio_sft_data(path, audio_paths):
    """A jsonl SFT dataset: each row pairs an ``<audio>`` user turn with an assistant transcript.

    ``<audio>`` is swift's media tag; the Omni template expands it to the ``<|audio_bos|><|AUDIO|>
    <|audio_eos|>`` placeholder run and loads the parallel ``audios`` entry into ``input_features``.
    """
    rows = [{
        'messages': [
            {
                'role': 'user',
                'content': '<audio>这段音频说了什么？'
            },
            {
                'role': 'assistant',
                'content': '这是一段随机生成的音频。'
            },
        ],
        'audios': [aud],
    } for aud in audio_paths]
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    return str(path)


def test_run_sft_omni_audio_end_to_end(tmp_path, omni_model, assert_trained):
    """Audio causal_lm SFT: the encoded ``input_features`` reach the thinker's audio tower and train.

    The Omni ``_post_encode`` runs ``get_audio_features`` over the row's ``input_features`` and scatters
    the audio embeds into the ``<|AUDIO|>`` slots; a text-only path leaves those placeholder tokens
    un-filled, so a finite, normalized loss is proof the audio features were consumed. The talker /
    token2wav generator heads ride along in the checkpoint but take no gradient on a text-target run.

    This is the Qwen-audio regression guard for the twinkle audio-collate contract (see the module-level
    FIXED note): per-row ``input_features`` ``[N, freq, time]`` + ``feature_attention_mask`` ``[N, time]``
    must survive the collate un-squeezed and un-reshaped so ``get_audio_features`` gets ``[batch, freq,
    time]``. batch_size=2 on purpose -- a size-1 batch skips twinkle's ``_collate_macro_batch`` entirely
    and would not exercise the concat path that used to fold the batch into dim 0.
    """
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    from swift.dev.config import (CheckpointConfig, DatasetConfig, DistributedConfig, ModelConfig, TemplateConfig,
                                  TrainConfig)
    from swift.dev.recipe import run_sft

    auds = [_write_audio(tmp_path / f'aud{i}.wav') for i in range(4)]
    data = _write_audio_sft_data(tmp_path / 'audio_sft.jsonl', auds)
    out_dir = str(tmp_path / 'out_omni_audio')

    history = run_sft(
        ModelConfig(model=omni_model, model_type=OMNI_MODEL_TYPE, task_type='causal_lm', torch_dtype='bfloat16'),
        TemplateConfig(template=OMNI_TEMPLATE, max_length=512),
        DatasetConfig(dataset=[data], dataset_shuffle=False),
        TrainConfig(learning_rate=1e-5, per_device_train_batch_size=2, gradient_accumulation_steps=1, max_steps=3),
        DistributedConfig(),
        CheckpointConfig(),
        output_dir=out_dir,
    )
    assert_trained(history, out_dir, 'causal_lm', max_loss=_CAUSAL_LM_MAX_LOSS)
