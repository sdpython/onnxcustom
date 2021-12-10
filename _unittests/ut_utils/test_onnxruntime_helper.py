"""
@brief      test log(time=1s)
"""
import unittest
from pyquickhelper.pycode import ExtTestCase
from onnxcustom.utils.onnxruntime_helper import (
    device_to_provider, provider_to_device, get_ort_device_type)


class TestOnnxRuntimeHelper(ExtTestCase):

    def test_provider_to_device(self):
        self.assertEqual(provider_to_device('CPUExecutionProvider'), 'cpu')
        self.assertEqual(provider_to_device('CUDAExecutionProvider'), 'cuda')
        self.assertRaise(lambda: provider_to_device('NONE'), ValueError)

    def test_device_to_provider(self):
        self.assertEqual(device_to_provider('cpu'), 'CPUExecutionProvider')
        self.assertEqual(device_to_provider('gpu'), 'CUDAExecutionProvider')
        self.assertRaise(lambda: device_to_provider('NONE'), ValueError)

    def test_get_ort_device_type(self):
        self.assertEqual(get_ort_device_type('cpu'), 0)
        self.assertEqual(get_ort_device_type('cuda'), 1)
        self.assertRaise(lambda: get_ort_device_type('none'), ValueError)


if __name__ == "__main__":
    unittest.main()
