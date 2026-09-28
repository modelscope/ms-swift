# Copyright (c) ModelScope Contributors. All rights reserved.
"""End-to-end ``merge_lora`` coverage for ``swift deploy``.

``--merge_lora`` folds a single LoRA adapter into the base weights before serving, so the deployment runs
one merged model with no per-request adapter routing (the alternative -- serving the adapter live -- is
what ``adapter_mapping`` without ``merge_lora`` does, and needs two or more adapters to be worth routing).
The merge runs inside the spawned ``run_deploy`` subprocess (``_merge_single_adapter`` ->
``run_merge_lora`` on CPU), then vLLM serves the merged checkpoint.

These tests really build a LoRA over ``Qwen3.5-4B``, deploy with ``merge_lora=True``, and assert the
observable contract: the merged weight directory is produced next to the adapter, the gateway comes up on
the merged model, and it serves a normal completion. A green module means ``swift deploy --merge_lora``
works end to end, not that a stub wrote a directory.

All tests are ``@pytest.mark.slow`` + ``@pytest.mark.accel(1)``; run with ``-m slow``.
"""
import gc
import os

import pytest

from swift.dev.tests.deploy.conftest import MODEL, MODEL_TYPE

pytestmark = [pytest.mark.slow, pytest.mark.accel(1)]

SERVED = 'policy'


@pytest.fixture(scope='module')
def lora_adapter(tmp_path_factory):
    """A real (randomly-initialised) LoRA over ``Qwen3.5-4B``, saved as a PEFT adapter directory.

    The weights are untrained -- merging is a structural operation, so what matters is that the adapter
    targets real modules of the base architecture and folds in cleanly, not what it learned. The base
    model is loaded on CPU, wrapped, saved, and released before the deployment subprocess reloads it.
    """
    import torch
    from peft import LoraConfig, get_peft_model

    from swift.dev.builders import load_model_processor
    from swift.dev.config import ModelConfig

    dest = str(tmp_path_factory.mktemp('lora') / 'adapter')
    model_config = ModelConfig(model=MODEL, model_type=MODEL_TYPE, torch_dtype='bfloat16', device_map='cpu')
    model, _ = load_model_processor(model_config, load_model=True)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=['q_proj', 'v_proj'],
        bias='none',
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.save_pretrained(dest)
    # Release the CPU copy of the base before the subprocess loads its own for the merge.
    del peft_model
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    assert os.path.isfile(os.path.join(dest, 'adapter_config.json'))
    return dest


def test_merge_lora_serves_the_merged_model(live_server, lora_adapter):
    """``merge_lora=True`` merges the adapter and serves the result as the single ``policy`` model."""
    with live_server(
            backend='vllm',
            adapter_mapping={SERVED: lora_adapter},
            deploy_overrides={'merge_lora': True},
    ) as srv:
        # The merge wrote a self-contained checkpoint next to the adapter (run_merge_lora's default
        # ``{adapter}-merged``), which is what vLLM was handed rather than the base + live adapter.
        merged_dir = f'{lora_adapter}-merged'
        assert os.path.isdir(merged_dir), f'merge_lora did not produce {merged_dir}'
        assert os.path.isfile(os.path.join(merged_dir, 'config.json'))

        models = srv.get('/models').json()
        assert SERVED in [m['id'] for m in models['data']]

        body = srv.chat([{'role': 'user', 'content': 'Say hello in one short sentence.'}], max_tokens=24)
        assert body['object'] == 'chat.completion' and body['model'] == SERVED
        assert body['choices'][0]['message']['content']
        assert body['usage']['completion_tokens'] > 0


def test_merge_lora_rejects_multiple_adapters():
    """Merging bakes ONE adapter into the weights, so two adapters cannot be routed afterwards.

    ``build_server_config`` fails loudly on the adapter count before loading anything, rather than
    silently merging only the first; it is the public config-assembly entry ``run_deploy`` calls, so the
    guard is asserted through it with placeholder paths (no real adapter, no server).
    """
    from swift.dev.config import DeployConfig, ModelConfig, TemplateConfig
    from swift.dev.recipe.run_deploy import build_server_config

    model_config = ModelConfig(model=MODEL, model_type=MODEL_TYPE, torch_dtype='bfloat16')
    template_config = TemplateConfig(template=MODEL_TYPE)
    deploy_config = DeployConfig(served_model_name=SERVED, merge_lora=True)
    with pytest.raises(ValueError, match='cannot serve 2 adapters'):
        build_server_config(
            model_config,
            template_config,
            adapter_mapping={'a': '/nonexistent/a',
                             'b': '/nonexistent/b'},
            deploy_config=deploy_config,
        )
