"""
@brief      test log(time=1s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase
from onnxruntime.capi._pybind_state import (  # pylint: disable=E0611
    OrtValue as C_OrtValue)
from onnxcustom.utils.onnxruntime_helper import (
    provider_to_device, get_ort_device_type,
    get_ort_device, ort_device_to_string,
    device_to_providers, numpy_to_ort_value)


class TestOnnxRuntimeHelper(ExtTestCase):

    def test_provider_to_device(self):
        self.assertEqual(provider_to_device('CPUExecutionProvider'), 'cpu')
        self.assertEqual(provider_to_device('CUDAExecutionProvider'), 'cuda')
        self.assertRaise(lambda: provider_to_device('NONE'), ValueError)

    def test_device_to_provider(self):
        self.assertEqual(device_to_providers('cpu'), ['CPUExecutionProvider'])
        self.assertEqual(device_to_providers('gpu'), ['CUDAExecutionProvider'])
        self.assertRaise(lambda: device_to_providers('NONE'), ValueError)

    def test_get_ort_device_type(self):
        self.assertEqual(get_ort_device_type('cpu'), 0)
        self.assertEqual(get_ort_device_type('cuda'), 1)
        self.assertRaise(lambda: get_ort_device_type('none'), ValueError)

    def test_get_ort_device_type_exc(self):
        self.assertRaise(
            lambda: get_ort_device_type(['cpu']),
            TypeError)
        self.assertRaise(
            lambda: get_ort_device_type('upc'),
            ValueError)

    def test_ort_device_to_string(self):
        for value in ['cpu', 'cuda', ('gpu', 'cuda'),
                      ('gpu:0', 'cuda'), ('cuda:0', 'cuda'),
                      ('gpu:1', 'cuda:1'), 'cuda:1']:
            with self.subTest(device=value):
                if isinstance(value, str):
                    a, b = value, value
                else:
                    a, b = value
                dev = get_ort_device(a)
                back = ort_device_to_string(dev)
                self.assertEqual(b, back)

    def test_ort_device_to_string_exc(self):
        self.assertRaise(lambda: ort_device_to_string('gg'), TypeError)

    def test_numpy_to_ort_value(self):
        res = numpy_to_ort_value(numpy.array([0]))
        self.assertIsInstance(res, C_OrtValue)


if __name__ == "__main__":
    unittest.main()
