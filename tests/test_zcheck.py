"""
Unit tests for the main entrypoint.
"""
import unittest
from contextlib import redirect_stdout
import io
from onnxcustom import check


class TestCheck(unittest.TestCase):

    def test_check(self):
        f = io.StringIO()
        with redirect_stdout(f):
            res = check(verbose=1)
        self.assertIsInstance(res, list)
        self.assertEqual(len(res), 0)

    def test__main__(self):
        import onnxcustom.__main__  # noqa


if __name__ == '__main__':
    unittest.main()
