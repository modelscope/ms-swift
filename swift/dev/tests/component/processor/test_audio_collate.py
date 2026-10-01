"""Guards for twinkle's two mutually-exclusive audio ``input_features`` collate contracts.

Both audio families put their features in a field named ``input_features``; they are told apart by the
mask field that rides alongside it:

  * Qwen2.5-Omni / Qwen-audio -> ``feature_attention_mask``. transformers'
    ``Qwen2_5OmniThinker.get_audio_features`` needs per-audio-batched ``[N, freq, time]`` features and
    ``[N, time]`` mask, produced by a plain dim-0 concat of the un-squeezed per-row tensors.
  * gemma4 (mcore-bridge mm_gpts/gemma4.py) -> ``input_features_mask``. Its audio tower is
    channels-last: per-sample 2-D ``[plane0, plane1]`` features get a trailing singleton and the 1-D
    mask is expanded to the feature plane, then the batch is folded into dim 0.

The bug this pins: twinkle used to apply the gemma4 channels-last reshape to ``input_features``
UNCONDITIONALLY (keyed only on the field name) and blanket-``.squeeze()`` every tensor, which stripped
Qwen-audio's leading audio dim and handed ``get_audio_features`` a ``[batch*freq, time, 1]`` tensor ->
IndexError against the ``[batch, time]`` mask. The Qwen-audio side is also covered end-to-end by
``test_multimodal.py::test_run_sft_omni_audio_end_to_end``; this is the fast, GPU-free contract guard
for BOTH sides (the gemma4 e2e needs a gemma4 checkpoint, so its layout is pinned here instead).

Calls ``collate_fn`` directly with CPU tensors and audio-only rows (batch=2, because a size-1 batch
short-circuits before ``_collate_macro_batch``), so no device, template or model is needed.
"""
import torch

from swift.dev.processor import InputProcessor


def _collate(rows):
    return InputProcessor(framework='transformers').collate_fn([dict(r) for r in rows])[0]


def test_qwen_audio_contract_keeps_batched_layout():
    """feature_attention_mask present -> per-row [N, freq, time] survives un-squeezed, plain dim-0 concat."""
    freq, time = 8, 16
    f0 = torch.arange(1 * freq * time, dtype=torch.float32).reshape(1, freq, time)
    f1 = torch.full((1, freq, time), -1.0)
    rows = [
        {
            'input_features': f0.clone(),
            'feature_attention_mask': torch.ones(1, time, dtype=torch.long)
        },
        {
            'input_features': f1.clone(),
            'feature_attention_mask': torch.ones(1, time, dtype=torch.long)
        },
    ]
    collated = _collate(rows)
    feats = collated['input_features']
    mask = collated['feature_attention_mask']
    # get_audio_features wants [batch, freq, time] + [batch, time] -- NOT the channels-last fold.
    assert feats.shape == (2, freq, time), tuple(feats.shape)
    assert mask.shape == (2, time), tuple(mask.shape)
    # Values preserved exactly (no reshape / no leading-dim squeeze).
    assert torch.equal(feats[0], f0[0])
    assert torch.equal(feats[1], f1[0])


def test_qwen_audio_contract_keeps_multi_audio_leading_dim():
    """A row with N>1 audios keeps its [N, freq, time]; the gemma4 2-D assert must not fire on it."""
    n, freq, time = 3, 8, 16
    feats_in = torch.randn(n, freq, time)
    rows = [{
        'input_features': feats_in.clone(),
        'feature_attention_mask': torch.ones(n, time, dtype=torch.long)
    }, {
        'input_features': torch.zeros(1, freq, time),
        'feature_attention_mask': torch.ones(1, time, dtype=torch.long)
    }]
    collated = _collate(rows)
    # dim-0 concat: 3 audios + 1 audio.
    assert collated['input_features'].shape == (n + 1, freq, time), tuple(collated['input_features'].shape)
    assert collated['feature_attention_mask'].shape == (n + 1, time)
    assert torch.equal(collated['input_features'][:n], feats_in)


def test_gemma4_channels_last_contract_preserved():
    """input_features_mask present -> the original channels-last reshape stays byte-identical."""
    p0, p1 = 6, 4
    f0 = torch.arange(p0 * p1, dtype=torch.float32).reshape(p0, p1)
    f1 = torch.full((p0, p1), -2.0)
    rows = [
        {
            'input_features': f0.clone(),
            'input_features_mask': torch.ones(p0, dtype=torch.long)
        },
        {
            'input_features': f1.clone(),
            'input_features_mask': torch.ones(p0, dtype=torch.long)
        },
    ]
    collated = _collate(rows)
    feats = collated['input_features']
    mask = collated['input_features_mask']
    # 2-D per-sample -> trailing singleton, batch folded into dim 0.
    assert feats.shape == (2 * p0, p1, 1), tuple(feats.shape)
    assert mask.shape == (2 * p0, p1), tuple(mask.shape)
    assert torch.equal(feats[:p0, :, 0], f0)
    assert torch.equal(feats[p0:, :, 0], f1)
    assert torch.equal(mask, torch.ones(2 * p0, p1, dtype=torch.long))
