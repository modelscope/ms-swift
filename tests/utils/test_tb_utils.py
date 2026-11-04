import os
import sys
import types
import unittest
from unittest.mock import patch

from swift.utils.tb_utils import plot_images


class TestTBUtils(unittest.TestCase):

    def test_plot_images_with_relative_tensorboard_dir(self):
        event_file = 'events.out.tfevents.test'
        tb_dir = 'runs'
        event_path = os.path.join(tb_dir, event_file)
        matplotlib = types.ModuleType('matplotlib')
        pyplot = types.ModuleType('matplotlib.pyplot')
        matplotlib.pyplot = pyplot

        with patch.dict(sys.modules, {'matplotlib': matplotlib, 'matplotlib.pyplot': pyplot}), \
                patch('swift.utils.tb_utils.os.path.exists', return_value=True), \
                patch('swift.utils.tb_utils.os.makedirs'), \
                patch('swift.utils.tb_utils.os.walk', return_value=[(tb_dir, [], [event_file])]), \
                patch('swift.utils.tb_utils.read_tensorboard_file', return_value={}) as mock_read:
            plot_images('images', tb_dir)

        mock_read.assert_called_once_with(event_path)


if __name__ == '__main__':
    unittest.main()
