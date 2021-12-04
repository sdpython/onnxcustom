"""
@brief      test log(time=4s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase
from skl2onnx.common.data_types import FloatTensorType, DoubleTensorType
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxRelu, OnnxMatMul)
from onnxcustom.utils.onnx_helper import (
    onnx_rename_weights, proto_type_to_dtype, dtype_to_var_type,
    get_onnx_opset)


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
        self.assertEqual(get_onnx_opset(onx), 14)
        self.assertRaise(lambda: get_onnx_opset(onx, "H"), ValueError)

    def test_dtype_to_var_type(self):
        self.assertEqual(dtype_to_var_type(numpy.float32), FloatTensorType)
        self.assertEqual(dtype_to_var_type(numpy.float64), DoubleTensorType)
        self.assertRaise(lambda: dtype_to_var_type(numpy.int64), ValueError)

    def test_proto_type_to_dtype(self):
        self.assertEqual(proto_type_to_dtype(1), numpy.float32)
        self.assertEqual(proto_type_to_dtype(11), numpy.float64)
        self.assertRaise(lambda: proto_type_to_dtype(9), ValueError)


if __name__ == "__main__":
    unittest.main()
