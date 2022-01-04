"""
@brief      test log(time=3s)
"""

import unittest
import numpy
from onnxruntime import OrtValue
from onnxruntime.capi._pybind_state import (  # pylint: disable=E0611
    OrtValue as C_OrtValue)
from pyquickhelper.pycode import ExtTestCase
from onnxcustom.utils import str_ortvalue
from onnxcustom.utils.onnxruntime_helper import get_ort_device


class TestUtilsPrintHelper(ExtTestCase):

    def test_print_ortvalue(self):
        expected = (
            "device=Cpu dtype=dtype('float32') shape=(1, 4) "
            "value=[0.0, 1.0, 4.0, 4.5]")
        value = numpy.array([[0, 1, 4, 4.5]], dtype=numpy.float32)
        dev = get_ort_device('cpu')
        ort = C_OrtValue.ortvalue_from_numpy(value, dev)
        text = str_ortvalue(ort)
        self.assertEqual(expected, text)
        text = str_ortvalue(ort)  # pylint: disable=W0212
        self.assertEqual(expected, text)

        expected = (
            "device=Cpu dtype=dtype('int64') shape=(100,) "
            "value=[0, 1, 2, 3, 4, '...', 95, 96, 97, 98, 99]")
        value = numpy.arange(100).astype(numpy.int64)
        ort = C_OrtValue.ortvalue_from_numpy(value, dev)
        text = str_ortvalue(ort)  # pylint: disable=W0212
        self.assertEqual(expected, text)

    def test_print_py_ortvalue(self):
        expected = (
            "device=Cpu dtype=dtype('float32') shape=(1, 4) "
            "value=[0.0, 1.0, 4.0, 4.5]")
        value = numpy.array([[0, 1, 4, 4.5]], dtype=numpy.float32)
        ort = OrtValue.ortvalue_from_numpy(value, 'cpu')
        text = str_ortvalue(ort)
        self.assertEqual(expected, text)
        text = str_ortvalue(ort)  # pylint: disable=W0212
        self.assertEqual(expected, text)

        expected = (
            "device=Cpu dtype=dtype('int64') shape=(100,) "
            "value=[0, 1, 2, 3, 4, '...', 95, 96, 97, 98, 99]")
        value = numpy.arange(100).astype(numpy.int64)
        ort = OrtValue.ortvalue_from_numpy(value, 'cpu')
        text = str_ortvalue(ort)  # pylint: disable=W0212
        self.assertEqual(expected, text)


if __name__ == "__main__":
    unittest.main()
