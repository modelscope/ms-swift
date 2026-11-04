import unittest

from swift.utils import format_time


class TestFormatTime(unittest.TestCase):

    def test_carries_rounded_seconds(self):
        # A rounded-up second must carry into the next unit instead of printing
        # an impossible '60s'.
        self.assertEqual(format_time(59.7), '1m 0s')
        self.assertEqual(format_time(119.6), '2m 0s')
        self.assertEqual(format_time(3599.8), '1h 0m 0s')
        self.assertEqual(format_time(3659.7), '1h 1m 0s')
        self.assertEqual(format_time(86399.7), '1d 0h 0m 0s')

    def test_normal_values_unchanged(self):
        self.assertEqual(format_time(30.2), '30s')
        self.assertEqual(format_time(90.0), '1m 30s')
        self.assertEqual(format_time(3600), '1h 0m 0s')


if __name__ == '__main__':
    unittest.main()
