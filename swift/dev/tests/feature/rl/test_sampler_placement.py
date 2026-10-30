"""RL sampler placement must preserve independent trainer and rollout parallelism."""
from types import SimpleNamespace

import unittest
from unittest.mock import patch


def check_replica(colocate, tp, pp, dp):
    from swift.dev import builders
    from swift.dev.config import ModelConfig, RolloutConfig
    from swift.dev.recipe.utils import build_rl_sampler, plan_device_groups

    config = RolloutConfig(vllm_tensor_parallel_size=tp, vllm_pipeline_parallel_size=pp,
                           vllm_data_parallel_size=dp)
    captured = {}

    def capture(model_config, **kwargs):
        captured.update(kwargs)
        return captured

    with patch.object(builders, 'build_sampler', capture):
        build_rl_sampler(model_config=ModelConfig(model='unused'), rollout_config=config,
                         template=None, tuner_config=None, colocate=colocate)
    groups, _ = plan_device_groups(8, 'colocate' if colocate else 'disaggregated', tp * pp * dp, [])
    ranks = dict(groups)[captured['remote_group']]
    assert captured['remote_group'] == 'sampler'
    assert len(ranks) // (tp * pp) == dp
    assert captured['device_mesh'].data_world_size == dp
    assert captured['device_mesh'].world_size == dp
    assert captured['engine_args']['tensor_parallel_size'] == tp
    assert captured['engine_args']['pipeline_parallel_size'] == pp
    assert captured['engine_args']['data_parallel_size'] == 1
    assert captured['engine_args'].get('enable_sleep_mode', False) == colocate
    assert config.vllm_data_parallel_size == dp
    if colocate:
        assert set(ranks).issubset(range(8))
    else:
        assert set(ranks).isdisjoint(range(8))


def check_planned():
    from swift.dev import builders
    from swift.dev.config import ModelConfig, RolloutConfig
    from swift.dev.recipe.utils import build_rl_sampler

    with patch.object(builders, 'build_sampler', lambda model_config, **kwargs: kwargs):
        result = build_rl_sampler(
            model_config=ModelConfig(model='unused'),
            rollout_config=RolloutConfig(vllm_tensor_parallel_size=2, vllm_data_parallel_size=4),
            template=None, tuner_config=None,
            resource_plan=SimpleNamespace(sampler=SimpleNamespace(replica_count=4)))
    assert result['remote_group'] == 'sampler'
    assert result['device_mesh'].data_world_size == 4
    assert result['engine_args']['data_parallel_size'] == 1


class SamplerPlacementTests(unittest.TestCase):
    def test_independent_parallel_layouts(self):
        for colocate in (True, False):
            for tp, pp, dp in ((1, 1, 8), (2, 1, 4), (4, 1, 2), (2, 2, 2)):
                with self.subTest(colocate=colocate, tp=tp, pp=pp, dp=dp):
                    check_replica(colocate, tp, pp, dp)

    def test_planned_layout(self):
        check_planned()


if __name__ == '__main__':
    unittest.main()
