"""
@brief      test log(time=3s)
"""
import unittest
import os
import numpy
from pyquickhelper.pycode import ExtTestCase
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxRelu, OnnxMatMul)
from onnxcustom.utils.onnx_helper import onnx_rename_weights


class TestOnnxHelper(ExtTestCase):

    def test_onnx_rename_weights(self):
        N, D_in, D_out, H = 3, 3, 3, 3
        var = [('X', FloatTensorType([N, D_in]))]
        w1 = numpy.random.randn(D_in, H).astype(numpy.float32)
        w2 = numpy.random.randn(H, D_out).astype(numpy.float32)
        opv = 14
        onx_alg = OnnxMatMul(
            OnnxRelu(OnnxMatMul(*var, w1, op_version=opv),
                     op_version=opv),
            w2, op_version=opv, output_names=['Y'])
        onx = onx_alg.to_onnx(
            var, target_opset=opv, outputs=[('Y', FloatTensorType())])

        onx = onnx_rename_weights(onx)
        names = [init.name for init in onx.graph.initializer]
        self.assertEqual(['I0_Ma_MatMulcst', 'I1_Ma_MatMulcst1'], names)


if __name__ == "__main__":
    unittest.main()
