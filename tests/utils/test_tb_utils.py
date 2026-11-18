import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from swift.utils.tb_utils import plot_images, read_tensorboard_file


class TestTBUtils(unittest.TestCase):

    def test_plot_images_with_relative_tensorboard_dir(self):
        event_file = 'events.out.tfevents.test'
        tb_dir = 'runs'
        matplotlib = types.ModuleType('matplotlib')
        pyplot = types.ModuleType('matplotlib.pyplot')
        matplotlib.pyplot = pyplot

        with patch.dict(sys.modules, {'matplotlib': matplotlib, 'matplotlib.pyplot': pyplot}), \
                patch('swift.utils.tb_utils.os.path.exists', return_value=True), \
                patch('swift.utils.tb_utils.os.makedirs'), \
                patch('swift.utils.tb_utils.os.walk', return_value=[(tb_dir, [], [event_file])]), \
                patch('swift.utils.tb_utils.read_tensorboard_file', return_value={}) as mock_read:
            plot_images('images', tb_dir)

        mock_read.assert_called_once_with(tb_dir)

    @staticmethod
    def _write_events(tb_dir, values, restart_step=None):
        from tensorboard.compat.proto.event_pb2 import Event, SessionLog
        from tensorboard.compat.proto.summary_pb2 import Summary
        from tensorboard.summary.writer.event_file_writer import EventFileWriter

        writer = EventFileWriter(tb_dir)
        if restart_step is not None:
            writer.add_event(Event(step=restart_step, session_log=SessionLog(status=SessionLog.START)))
        for tag, step, value in values:
            writer.add_event(Event(step=step, summary=Summary(value=[Summary.Value(tag=tag, simple_value=value)])))
        writer.flush()
        writer.close()

    def test_plot_images_reads_all_event_files_in_run(self):
        import matplotlib
        matplotlib.use('Agg')

        with tempfile.TemporaryDirectory() as folder:
            tb_dir = os.path.join(folder, 'runs')
            images_dir = os.path.join(folder, 'images')
            self._write_events(tb_dir, [('train/loss', 1, 2.0)])
            self._write_events(tb_dir, [('eval/loss', 2, 1.0)])
            plot_images(images_dir, tb_dir)
            self.assertEqual(sorted(os.listdir(images_dir)), ['eval_loss.png', 'train_loss.png'])

    def test_read_directory_preserves_tensorboard_resume_semantics(self):
        with tempfile.TemporaryDirectory() as tb_dir:
            self._write_events(tb_dir, [('train/loss', 1, 3.0), ('train/loss', 2, 2.0)])
            self._write_events(tb_dir, [('train/loss', 2, 1.0), ('train/loss', 3, 0.5)], restart_step=2)
            expected = [{'step': 1, 'value': 3.0}, {'step': 2, 'value': 1.0}, {'step': 3, 'value': 0.5}]
            self.assertEqual(read_tensorboard_file(tb_dir)['train/loss'], expected)
            self.assertEqual(read_tensorboard_file(tb_dir)['train/loss'], expected)

    def test_read_single_event_file(self):
        with tempfile.TemporaryDirectory() as tb_dir:
            self._write_events(tb_dir, [('train/loss', 1, 2.0)])
            event_path = os.path.join(tb_dir, os.listdir(tb_dir)[0])
            self.assertEqual(read_tensorboard_file(event_path), {'train/loss': [{'step': 1, 'value': 2.0}]})

    def test_plot_images_without_scalar_events(self):
        with tempfile.TemporaryDirectory() as folder:
            tb_dir = os.path.join(folder, 'runs')
            os.mkdir(tb_dir)
            images_dir = os.path.join(folder, 'images')
            plot_images(images_dir, tb_dir)
            self.assertEqual(os.listdir(images_dir), [])
            self._write_events(tb_dir, [])
            self.assertEqual(read_tensorboard_file(tb_dir), {})
            plot_images(images_dir, tb_dir)
            self.assertEqual(os.listdir(images_dir), [])


if __name__ == '__main__':
    unittest.main()
