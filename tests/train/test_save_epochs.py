import unittest
from types import SimpleNamespace

from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.trainer_utils import IntervalStrategy


class TestSaveEpochs(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            from swift.ray.megatron.driver_utils import compute_iter_params
            from swift.trainers.patcher import DefaultFlowCallbackNew
            cls.compute_iter_params = compute_iter_params
            cls.DefaultFlowCallbackNew = DefaultFlowCallbackNew
        except (ImportError, RuntimeError) as e:
            raise unittest.SkipTest(f'ms-swift dependencies are not available: {e}')

    @staticmethod
    def _training_args(**kwargs):
        args = dict(
            eval_strategy=IntervalStrategy.EPOCH,
            logging_strategy=IntervalStrategy.NO,
            eval_delay=0,
            save_strategy=IntervalStrategy.EPOCH,
            save_epochs=2,
            max_epochs=None,
        )
        args.update(kwargs)
        return SimpleNamespace(**args)

    def test_hf_callback_aligns_epoch_evaluation_with_saves(self):
        callback = self.DefaultFlowCallbackNew()
        args = self._training_args()

        first_epoch = callback.on_epoch_end(args, TrainerState(epoch=1), TrainerControl())
        self.assertFalse(first_epoch.should_save)
        self.assertFalse(first_epoch.should_evaluate)

        second_epoch = callback.on_epoch_end(args, TrainerState(epoch=2), TrainerControl())
        self.assertTrue(second_epoch.should_save)
        self.assertTrue(second_epoch.should_evaluate)

    def test_ray_compute_iter_params_uses_save_epochs(self):
        data_info = {
            'micro_batch_size': 2,
            'global_batch_size': 8,
            'num_generations': 1,
            'train_dataset': list(range(100)),
            'val_dataset': None,
            'save_strategy': 'epoch',
            'save_epochs': 3,
            'train_iters': None,
            'num_train_epochs': 2,
            'eval_iters': 0,
        }
        result = self.compute_iter_params(data_info, dp_size=1)
        self.assertEqual(result['save_steps'], 36)
        self.assertEqual(result['eval_steps'], 36)
        self.assertEqual(result['train_iters'], 25)

    def test_ray_streaming_save_epochs_is_rejected(self):

        class StreamingDataset:

            def __iter__(self):
                return iter(())

        data_info = {
            'micro_batch_size': 2,
            'global_batch_size': 8,
            'num_generations': 1,
            'train_dataset': StreamingDataset(),
            'val_dataset': None,
            'save_strategy': 'epoch',
            'save_epochs': 2,
            'train_iters': 10,
            'num_train_epochs': None,
            'eval_iters': 0,
        }
        with self.assertRaisesRegex(ValueError, 'streaming dataset'):
            self.compute_iter_params(data_info, dp_size=1)

    def test_ray_existing_streaming_epoch_strategy_remains_compatible(self):

        class StreamingDataset:

            def __iter__(self):
                return iter(())

        data_info = {
            'micro_batch_size': 2,
            'global_batch_size': 8,
            'num_generations': 1,
            'train_dataset': StreamingDataset(),
            'val_dataset': None,
            'save_strategy': 'epoch',
            'save_epochs': None,
            'train_iters': 10,
            'num_train_epochs': None,
            'eval_iters': 0,
        }
        result = self.compute_iter_params(data_info, dp_size=1)
        self.assertEqual(result['train_iters'], 10)
        self.assertNotIn('save_steps', result)


if __name__ == '__main__':
    unittest.main()
