"""
@brief      test log(time=9s)
"""
import unittest
import logging
from pyquickhelper.pycode import ExtTestCase
import numpy
from onnxruntime import InferenceSession
try:
    from onnxruntime import TrainingSession
except ImportError:
    # onnxruntime not training
    TrainingSession = None
from skl2onnx.algebra.onnx_ops import OnnxAdd  # pylint: disable=E0611
from skl2onnx.common.data_types import FloatTensorType
from onnxcustom import __max_supported_opset__ as opset
from onnxcustom.training.grad_helper import onnx_derivative


class TestGradHelper(ExtTestCase):

    @classmethod
    def setUpClass(cls):
        logger = logging.getLogger('skl2onnx')
        logger.setLevel(logging.WARNING)
        logging.basicConfig(level=logging.WARNING)

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_grad_helper(self):
        opv = opset
        node = OnnxAdd('X', numpy.array([1], dtype=numpy.float32),
                       op_version=opv, output_names=['Y'])
        onx = node.to_onnx({'X': FloatTensorType([None, None])},
                           {'Y': FloatTensorType([None, None])},
                           target_opset=opv)

        new_onx = onnx_derivative(onx)
        sess = InferenceSession(new_onx.SerializeToString())
        self.assertNotEmpty(sess)


if __name__ == "__main__":
    unittest.main()
