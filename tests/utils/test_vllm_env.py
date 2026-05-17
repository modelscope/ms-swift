import os

from swift.utils.env import configure_vllm_allreduce_env


def test_configure_vllm_allreduce_env_sets_default_for_tensor_parallel(monkeypatch):
    monkeypatch.delenv('VLLM_ALLREDUCE_USE_SYMM_MEM', raising=False)

    configure_vllm_allreduce_env(2)

    assert os.environ['VLLM_ALLREDUCE_USE_SYMM_MEM'] == '0'


def test_configure_vllm_allreduce_env_preserves_explicit_value(monkeypatch):
    monkeypatch.setenv('VLLM_ALLREDUCE_USE_SYMM_MEM', '1')

    configure_vllm_allreduce_env(2)

    assert os.environ['VLLM_ALLREDUCE_USE_SYMM_MEM'] == '1'


def test_configure_vllm_allreduce_env_skips_single_tensor_parallel(monkeypatch):
    monkeypatch.delenv('VLLM_ALLREDUCE_USE_SYMM_MEM', raising=False)

    configure_vllm_allreduce_env(1)

    assert 'VLLM_ALLREDUCE_USE_SYMM_MEM' not in os.environ


def test_configure_vllm_allreduce_env_skips_missing_tensor_parallel(monkeypatch):
    monkeypatch.delenv('VLLM_ALLREDUCE_USE_SYMM_MEM', raising=False)

    configure_vllm_allreduce_env(None)

    assert 'VLLM_ALLREDUCE_USE_SYMM_MEM' not in os.environ
