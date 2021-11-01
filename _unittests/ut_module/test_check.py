"""
@brief      test log(time=0s)
"""
import unittest
from pyquickhelper.pycode import ExtTestCase
from onnxcustom import check


class TestCheck(ExtTestCase):
    """Test style."""

    def test_check(self):
        test = check()
        self.assertEmpty(test)


if __name__ == "__main__":
    unittest.main()
