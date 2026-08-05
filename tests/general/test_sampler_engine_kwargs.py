import torch

from swift.pipelines.sampling.vanilla_sampler import _pop_engine_torch_dtype


def _engine_stub(*args, torch_dtype=None, **kwargs):
    return torch_dtype, kwargs


def test_engine_kwargs_torch_dtype_no_crash():
    # dtype in engine_kwargs was the workaround while --torch_dtype was ignored;
    # it must survive the explicit argument now instead of raising TypeError.
    cleaned = _pop_engine_torch_dtype({'torch_dtype': 'bfloat16', 'max_model_len': 4096})
    torch_dtype, kwargs = _engine_stub('model', torch_dtype=torch.bfloat16, **cleaned)
    assert torch_dtype == torch.bfloat16
    assert kwargs == {'max_model_len': 4096}


def test_engine_kwargs_passthrough():
    assert _pop_engine_torch_dtype({'max_model_len': 4096}) == {'max_model_len': 4096}
    assert _pop_engine_torch_dtype({'torch_dtype': None}) == {}
    assert _pop_engine_torch_dtype({}) == {}


def test_duplicate_torch_dtype_would_raise():
    try:
        _engine_stub('model', torch_dtype=torch.bfloat16, **{'torch_dtype': 'bfloat16'})
    except TypeError:
        return
    raise AssertionError('expected TypeError without the pop')


if __name__ == '__main__':
    test_engine_kwargs_torch_dtype_no_crash()
    test_engine_kwargs_passthrough()
    test_duplicate_torch_dtype_would_raise()
