"""
@brief      test log(time=0s)
"""
import unittest
from contextlib import redirect_stdout
import io
from pyquickhelper.pycode import ExtTestCase
from onnxcustom import check


class TestCheck(ExtTestCase):
    """Test style."""

    def test_check(self):
        test = check()
        self.assertEmpty(test)

    def test_check_out(self):
        f = io.StringIO()
        with redirect_stdout(f):
            res = check(verbose=1)
        self.assertIsInstance(res, list)
        if len(res) > 0:
            raise AssertionError(res)

    def test__main__(self):
        import onnxcustom.__main__  # pylint: disable=W0611

if __name__ == "__main__":
    unittest.main()
