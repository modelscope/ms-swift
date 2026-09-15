# Copyright (c) ModelScope Contributors. All rights reserved.
import pytest
from transformers import HfArgumentParser
from types import SimpleNamespace

from swift.arguments.base_args.generation_args import GenerationArguments
from swift.infer_engine import RequestConfig
from swift.pipelines.infer.deploy import SwiftDeploy


@pytest.mark.parametrize('request_value, expected', [(None, False), (False, False), (True, True)])
def test_deploy_detokenize_default_and_override(request_value, expected):
    args, = HfArgumentParser(GenerationArguments).parse_args_into_dataclasses(['--detokenize', 'false'])
    args.task_type = 'causal_lm'
    request = RequestConfig(detokenize=request_value)
    SwiftDeploy._set_request_config(SimpleNamespace(args=args), request)
    assert request.detokenize is expected


@pytest.mark.parametrize('value', [None, False, True])
def test_vllm_detokenize_sampling_and_stops(value):
    pytest.importorskip('vllm')
    from vllm import SamplingParams

    from swift.infer_engine.vllm_engine import VllmEngine

    engine = object.__new__(VllmEngine)
    engine.generation_config = SamplingParams()
    engine.template = SimpleNamespace(template_meta=SimpleNamespace(stop_words=['END', [42]]))
    engine._get_stop_words = lambda words: [word for word in words if isinstance(word, str)]
    engine._get_stop_token_ids = lambda words: [42]
    request = RequestConfig(max_tokens=1, detokenize=value, prompt_logprobs=64)
    config = engine._prepare_generation_config(request)
    assert config.detokenize is (True if value is None else value)
    assert config.prompt_logprobs == 64
    engine._add_stop_words(config, request)
    assert config.stop == ([] if value is False else ['END'])
    assert config.stop_token_ids == [42]
    if value is False:
        request.stop = ['explicit stop']
        with pytest.raises(ValueError, match='detokenize=True'):
            engine._add_stop_words(config, request)
