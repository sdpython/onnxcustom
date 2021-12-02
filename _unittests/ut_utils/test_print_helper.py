"""
@brief      test log(time=3s)
"""

import unittest
import numpy
from onnxruntime import OrtValue
from pyquickhelper.pycode import ExtTestCase
from onnxcustom.utils import str_ortvalue

class TestUtilsPrintHelper(unittest.TestCase):

    def test_print_ortvalue(self):
        expected = (
            "device=Cpu dtype=dtype('float32') shape=(1, 4) "
            "value=[0.0, 1.0, 4.0, 4.5]")
        value = numpy.array([[0, 1, 4, 4.5]], dtype=numpy.float32)
        ort = OrtValue.ortvalue_from_numpy(value, 'cpu', 0)
        text = str_ortvalue(ort)
        self.assertEqual(expected, text)
        text = str_ortvalue(ort._ortvalue)
        self.assertEqual(expected, text)

        expected = (
            "device=Cpu dtype=dtype('int32') shape=(100,) "
            "value=[0, 1, 2, 3, 4, '...', 95, 96, 97, 98, 99]")
        value = numpy.arange(100)
        ort = OrtValue.ortvalue_from_numpy(value, 'cpu', 0)
        text = str_ortvalue(ort._ortvalue)
        self.assertEqual(expected, text)


if __name__ == "__main__":
    unittest.main()
