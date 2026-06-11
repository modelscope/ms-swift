import unittest


class TestMegatronArgs(unittest.TestCase):
    """Megatron import / args smoke test (GPU and NPU adapted).

    Covers: MegatronSftArguments initialization, MegatronRLHFArguments,
    MegatronArguments field validation.

    Why these tests are needed:
    - tests/megatron/test_train.py and test_lora.py have top-level functions
      that require multi-GPU and mcore models, too heavy for CI.
    - Megatron argument construction is a common entry point that should be
      validated even without a full training run.
    - On NPU, Megatron dependencies (mcore, MindSpeed) may not be installed,
      so we gracefully skip.
    """

    @classmethod
    def setUpClass(cls):
        try:
            from swift.megatron import (MegatronArguments, MegatronExportArguments, MegatronPretrainArguments,
                                        MegatronRLHFArguments, MegatronSftArguments)
            cls._megatron_available = True
            cls.MegatronArguments = MegatronArguments
            cls.MegatronSftArguments = MegatronSftArguments
            cls.MegatronRLHFArguments = MegatronRLHFArguments
        except (ImportError, RuntimeError) as e:
            cls._megatron_available = False
            cls._skip_reason = str(e)

    def _skip_if_no_megatron(self):
        if not self._megatron_available:
            self.skipTest(f'Megatron dependencies not available: {self._skip_reason}')

    def test_megatron_import(self):
        self._skip_if_no_megatron()

    def test_megatron_sft_args_construction(self):
        self._skip_if_no_megatron()

        args = self.MegatronSftArguments(
            mcore_model='Qwen2-7B-Instruct-mcore',
            dataset=['AI-ModelScope/alpaca-gpt4-data-zh#20'],
            split_dataset_ratio=0.01,
            tensor_model_parallel_size=1,
            train_iters=1,
            skip_megatron_init=True,
        )
        self.assertEqual(args.train_iters, 1)
        self.assertEqual(args.tensor_model_parallel_size, 1)

    def test_megatron_rlhf_args_construction(self):
        self._skip_if_no_megatron()

        args = self.MegatronRLHFArguments(
            rlhf_type='grpo',
            mcore_model='Qwen2-7B-Instruct-mcore',
            dataset=['AI-ModelScope/alpaca-gpt4-data-zh#20'],
            reward_funcs=['format'],
            num_generations=2,
            max_completion_length=128,
            tensor_model_parallel_size=1,
            train_iters=1,
            skip_megatron_init=True,
        )
        self.assertEqual(args.rlhf_type, 'grpo')
        self.assertIn('format', args.reward_funcs)

    def test_megatron_base_args_fields(self):
        self._skip_if_no_megatron()

        expected_fields = [
            'tensor_model_parallel_size',
            'pipeline_model_parallel_size',
            'context_parallel_size',
            'sequence_parallel_size',
            'train_iters',
            'micro_batch_size',
            'global_batch_size',
            'lr',
            'min_lr',
            'bf16',
        ]
        from dataclasses import fields
        field_names = {f.name for f in fields(self.MegatronArguments)}
        for field_name in expected_fields:
            self.assertIn(field_name, field_names, f'MegatronArguments missing field: {field_name}')


if __name__ == '__main__':
    unittest.main()
