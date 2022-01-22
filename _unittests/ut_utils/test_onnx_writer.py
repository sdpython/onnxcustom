"""
@brief      test log(time=9s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxReciprocal, OnnxDiv)
from mlprodict.onnxrt import OnnxInference
from onnxcustom import get_max_opset
from onnxcustom.utils.onnx_rewriter import onnx_rewrite_operator


class TestOnnxWriter(ExtTestCase):

    def test_onnx_rewrite_operator(self):
        opset = get_max_opset()
        node1 = OnnxReciprocal('X', output_names=['Y'],
                               op_version=opset)
        onx1 = node1.to_onnx(
            inputs={'X': FloatTensorType()},
            outputs={'Y': FloatTensorType()},
            target_opset=opset)
        onx1.graph.name = "jjj"
        oinf1 = OnnxInference(onx1)

        node2 = OnnxDiv(numpy.array([1], dtype=numpy.float32),
                        'X', output_names=['Y'],
                        op_version=opset)
        onx2 = node2.to_onnx(
            inputs={'X': FloatTensorType()},
            outputs={'Y': FloatTensorType()},
            target_opset=opset)
        oinf2 = OnnxInference(onx2)
        X = numpy.array([[5, 6]], dtype=numpy.float32)
        y1 = oinf1.run({'X': X})['Y']
        y2 = oinf2.run({'X': X})['Y']
        self.assertEqualArray(y1, y2)

        onx3 = onnx_rewrite_operator(onx1, 'Reciprocal', onx2)
        self.assertNotIn('Reciprocal', str(onx3))
        oinf3 = OnnxInference(onx3)
        y3 = oinf3.run({'X': X})['Y']
        self.assertEqualArray(y1, y3)


if __name__ == "__main__":
    unittest.main()
