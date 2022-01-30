"""
@brief      test log(time=0s)
"""
import unittest
import logging
import numpy
from scipy.special import expit  # pylint: disable=E0611
from pyquickhelper.pycode import ExtTestCase
from skl2onnx.algebra.onnx_ops import OnnxSigmoid  # pylint: disable=E0611
from skl2onnx.common.data_types import FloatTensorType
from mlprodict.onnxrt import OnnxInference
from onnxcustom import get_max_opset


class TestOnnxRuntimes(ExtTestCase):
    """Test style."""

    @classmethod
    def setUpClass(cls):
        logger = logging.getLogger('skl2onnx')
        logger.setLevel(logging.WARNING)
        logging.basicConfig(level=logging.WARNING)

    def test_check(self):
        opset = get_max_opset()
        min_values = [-41.621277, -40.621277, -30.621277, -20.621277,
                      -19, -18, -17, -15, -14, -13, -12, -11, -10, -5, -2]
        data = numpy.array(
            [[0]],
            dtype=numpy.float32)

        node = OnnxSigmoid('X', op_version=opset, output_names=['Y'])
        onx = node.to_onnx({'X': FloatTensorType()},
                           {'Y': FloatTensorType()},
                           target_opset=opset)
        rts = ['numpy', 'python', 'onnxruntime1']
        for mv in min_values:
            data[:, 0] = mv
            for rt in rts:
                if rt == 'numpy':
                    y = expit(data)
                else:
                    oinf = OnnxInference(onx, runtime=rt)
                    y = oinf.run({'X': data})['Y']
                self.assertNotEmpty(y)


if __name__ == "__main__":
    unittest.main()
