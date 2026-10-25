# Copyright (c) ModelScope Contributors. All rights reserved.
"""Dimension ④ -- multimodal (vision-language) RL trained end to end, plus a pure-text control
(RL_PLAN §四 test_multimodal_e2e).

A VL policy rolled out and trained through the production CLI lifecycle (``run_rl``): real Ray actors, a real
vLLM colocate sampler that conditions generation on the image, LoRA on a tiny Qwen2.5-VL checkpoint. No stubs,
no degenerate single-sample shortcuts -- a green test means the image travelled dataset row -> rollout
``Trajectory`` -> the sampler's ``multi_modal_data`` -> generation -> ``_merge_multimodal_keys`` lifting
``pixel_values``/``image_grid_thw`` into the training feature -> the VL forward -> a policy-gradient /
best-of-n step that moved ``lora_B``.

TWO independent oracles, neither a self-consistency round-trip:

1. Model side (fail-loudly coupling -- the oracle ``infer/test_multimodal_e2e`` already relies on): a VL
   forward given ``<image>`` pad tokens in ``input_ids`` but no ``pixel_values`` RAISES on the token/patch
   count mismatch, and the template rejects an ``<image>`` placeholder with no matching image. So a clean run
   that trains ``lora_B`` proves the vision tensors really reached BOTH ``generate`` and the training forward
   -- the image was consumed, not silently dropped to a text-only prompt.

2. Rollout side (positive): the VL reward asserts every scored sample still carried its ``images`` media
   column (threaded through ``prompt_extras``), so the media reference survived dataset -> rollout -> reward
   scoring. The pure-text control runs the SAME recipe with no media column and a text policy, proving the
   no-media branch of ``_prompt_trajectory`` still trains and that the VL runs are not text runs in disguise.

Gated ``slow`` (a real Ray session per test) + ``accel(1)`` (vLLM colocate on one card). Run on a free card
with ``CUDA_VISIBLE_DEVICES=<card> pytest swift/dev/tests/feature/rl/test_multimodal_e2e.py -m slow``.
"""
import pytest

from swift.dev.tests.feature.rl.conftest import rl_configs, run_rl
from twinkle.data_format import last_assistant_text


def _image_bearing_paths(value):
    """Normalise a row's ``images`` entry to its path strings.

    HF datasets round-trips a bare path into ``{'bytes': None, 'path': ...}``; accept either shape (and a bare
    string, or a list of either) so the assertion is about the media reference surviving, not its
    serialisation -- mirrors ``infer/test_multimodal_e2e._image_paths``.
    """
    if value is None:
        return []
    if isinstance(value, (str, dict)):
        value = [value]
    return [(im.get('path') if isinstance(im, dict) else im) for im in value]


def _vl_content_reward(trajectories, **kwargs):
    """Content-varied ORM reward that ALSO asserts every scored sample carried its image (rollout oracle).

    Two jobs. (1) The char-sum-mod-11 content reward gives a GRPO group a non-zero advantage ``std`` -- a
    constant reward across a prompt's ``num_generations`` completions gives std=0 -> advantage 0 -> no gradient
    -> ``lora_B`` never moves (a LENGTH reward is constant here, a tiny model never emits EOS, so every
    completion runs to ``max_completion_length``). (2) The ``images`` assertion is the POSITIVE rollout-side
    oracle: the media column threaded from the dataset row through ``prompt_extras`` into each trajectory's
    flat extras, so the prompt was sampled WITH its image, not as bare text. The model-side oracle is the
    fail-loudly coupling documented in the module header (a clean VL step proves ``pixel_values`` reached the
    forward).

    A rule reward is invoked ``scorer(rows)`` -- one positional, no column kwargs -- and each row IS a
    trajectory dict (``ScoreWorker._score_rows`` spreads ``sample.extra`` into it), so ``images`` is read
    per-trajectory, not off a column list.
    """
    for trajectory in trajectories:
        sample_images = trajectory.get('images')
        assert _image_bearing_paths(sample_images), (
            f'VL rollout dropped the image: a scored sample carried no media reference '
            f'(keys={sorted(trajectory)}, images={sample_images!r})')
    return [float(sum(ord(ch) for ch in last_assistant_text(t)) % 11) / 11.0 for t in trajectories]


def _varied_content_reward(trajectories, **kwargs):
    """The pure-text control's reward: content-varied, and asserts NO media column rode the text rollout.

    The contrast that makes the VL image assertion meaningful: a text row threads no ``images`` extra, so the
    same recipe that carries media on a VL run carries none here. This proves the VL runs really did thread an
    image (not that every run trivially has one) and that the no-media branch of ``_prompt_trajectory`` still
    trains a text policy. The char-sum-mod-11 content reward keeps the GRPO advantage ``std`` non-zero so the
    control genuinely moves ``lora_B`` too.
    """
    for trajectory in trajectories:
        assert not trajectory.get('images'), (
            f'pure-text control unexpectedly carried a media column: images={trajectory.get("images")!r}')
    return [float(sum(ord(ch) for ch in last_assistant_text(t)) % 11) / 11.0 for t in trajectories]


@pytest.mark.slow
@pytest.mark.accel(1)
def test_vl_grpo_rollout_consumes_image(tiny_vl, rl_data, tmp_path, assert_rl_trained):
    """VL GRPO end to end: a real vLLM rollout conditioned on the image feeds a real policy-gradient step.

    Every dataset row carries an ``<image>`` placeholder plus its ``images`` column, so the rollout threads the
    media onto the ``Trajectory``, vLLM generates from ``multi_modal_data={'image': ...}``, and
    ``_merge_multimodal_keys`` lifts ``pixel_values``/``image_grid_thw`` into the training feature. The VL
    forward then either consumes them or raises on the token/patch mismatch -- so a run that trains ``lora_B``
    (read back from ``checkpoint-final``) is proof the image reached the forward, and ``_vl_content_reward``
    independently proves it reached scoring. Post-conditions beyond the universal ones: the GRPO
    policy-gradient fingerprint (``train/approx_kl`` / ``train/clip_ratio`` -- absent if GRPO degraded to a
    non-rollout loop) and a non-zero group-relative advantage moving the adapter. A colocate run rides the
    driver-less plain loop, so it emits no StreamingDriver telemetry (``stream_publishes`` / ``version_span_mean``
    are disaggregated-only); the policy-gradient metrics are the on-rollout fingerprint here.
    """
    out_dir = str(tmp_path / 'out_vl_grpo')
    configs = rl_configs(
        rl_type='grpo',
        model=tiny_vl,
        model_type='qwen2_5_vl',
        template='qwen2_5_vl',
        dataset=rl_data.multimodal(n_images=2),
        out_dir=out_dir,
        max_steps=2,
        rlhf_over={'num_generations': 4}, reward_over={'orm': [_vl_content_reward]},
        # temperature>0 so a prompt's group completions differ (greedy -> identical -> advantage std=0 -> no
        # gradient); a short completion budget keeps the VL rollout cheap on one card.
        generation_over={'temperature': 1.0, 'top_p': 1.0, 'max_new_tokens': 16},
    )
    history = run_rl(configs)
    assert_rl_trained(
        history, configs, 'grpo', max_loss=20.0, require_keys=('train/approx_kl', 'train/clip_ratio'))


@pytest.mark.slow
@pytest.mark.accel(1)
def test_vl_rft_rollout_consumes_image(tiny_vl, rl_data, tmp_path, assert_rl_trained):
    """VL RFT end to end: roll out on the image, keep best-of-n, SFT the kept completions.

    RFT reuses GRPO's rollout + reward machinery but sets the plain SFT ``cross_entropy`` loss on the KEPT
    completions -- no advantage, no reference, no importance ratio. This drives the SAME multimodal wiring
    (``_prompt_trajectory`` -> vLLM ``multi_modal_data`` -> ``_merge_multimodal_keys``) through the RFT branch,
    so a green run proves the vision tensors are lifted on the RFT path too, not only GRPO's. ``best_of_n``
    always keeps exactly one completion per prompt, so a round never comes up empty and the run always takes
    real steps on a tiny model whose gibberish completions score all-zero accuracy; the kept-set SFT gradient
    moves ``lora_B`` regardless, which is the point -- RFT trains on the selected sequences. The image oracle
    (fail-loudly coupling + ``_vl_content_reward``'s media assertion) is identical to the GRPO case.
    """
    out_dir = str(tmp_path / 'out_vl_rft')
    configs = rl_configs(
        rl_type='rft',
        model=tiny_vl,
        model_type='qwen2_5_vl',
        template='qwen2_5_vl',
        dataset=rl_data.multimodal(n_images=2),
        out_dir=out_dir,
        max_steps=2,
        rlhf_over={
            'rft_num_samples': 2,
            'rft_select': 'best_of_n',
            'rft_iterations': 1
        },
        reward_over={'orm': [_vl_content_reward]},
        generation_over={'temperature': 1.0, 'top_p': 1.0, 'max_new_tokens': 16},
    )
    history = run_rl(configs)
    assert_rl_trained(history, configs, 'rft', max_loss=20.0)


@pytest.mark.slow
@pytest.mark.accel(1)
def test_text_grpo_control_no_media(tiny_qwen2_5, rl_data, tmp_path, assert_rl_trained):
    """Pure-text control: the SAME GRPO recipe with no media column trains a text policy.

    The contrastive half of dimension ④. A text row threads no ``images`` column, so ``_prompt_trajectory``
    yields a plain text trajectory (its documented "no media -> unchanged behaviour" branch) and the rollout
    carries no vision tensors. ``_varied_content_reward`` asserts the score columns have NO ``images`` key --
    the mirror of the VL reward's assertion -- so together the two prove the media column is threaded exactly
    when the row has media (not always, not never). Training a text policy to a moved ``lora_B`` here shows the
    multimodal wiring is additive: it neither breaks the text path nor is required for GRPO to train. The
    colocate run rides the driver-less plain loop, so the on-rollout fingerprint is the GRPO policy-gradient
    pair (``train/approx_kl`` / ``train/clip_ratio``), not the disaggregated-only StreamingDriver telemetry.
    """
    out_dir = str(tmp_path / 'out_text_control')
    configs = rl_configs(
        rl_type='grpo',
        model=tiny_qwen2_5,
        model_type='qwen2',
        template='qwen2_5',
        dataset=rl_data.prompt_only(n=4),
        out_dir=out_dir,
        max_steps=2,
        rlhf_over={'num_generations': 4}, reward_over={'orm': [_varied_content_reward]},
        generation_over={'temperature': 1.0, 'top_p': 1.0, 'max_new_tokens': 16},
    )
    history = run_rl(configs)
    assert_rl_trained(
        history, configs, 'grpo', max_loss=20.0, require_keys=('train/approx_kl', 'train/clip_ratio'))
