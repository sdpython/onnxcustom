"""
@brief      test log(time=9s)
"""
import unittest
import numpy
from onnxruntime import InferenceSession
from pyquickhelper.pycode import ExtTestCase
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnx_tools.onnx_export import export2onnx
from onnxcustom import get_max_opset
from onnxcustom.utils.onnx_function import function_onnx_graph, get_supported_functions
from onnxcustom.utils.onnxruntime_helper import device_to_providers


class TestOnnxFunction(ExtTestCase):

    def common_check(self, name, fct):
        onx = function_onnx_graph(
            name, target_opset=get_max_opset(),
            dtype=numpy.float32)
        expected = numpy.random.randn(10, 1).astype(numpy.float32)
        predicted = numpy.random.randn(10, 1).astype(numpy.float32)
        fin = fct(expected, predicted)

        oinf = OnnxInference(onx)
        got = oinf.run({'X1': expected, 'X2': predicted})
        self.assertEqualArray(fin, got['Y'], decimal=5)

        providers = device_to_providers('cpu')
        sess = InferenceSession(onx.SerializeToString(), providers=providers)
        got = sess.run(None, {'X1': expected, 'X2': predicted})
        self.assertEqualArray(fin, got[0], decimal=5)

    def test_exc(self):
        self.assertRaise(lambda: function_onnx_graph("H"))

    def test_onnx_square_error(self):
        self.common_check("square_error", lambda x1,
                          x2: ((x1 - x2) ** 2).sum())

    def test_grad_onnx_square_error(self):
        self.common_check("grad_square_error", lambda x1, x2: (x1 - x2) * (-2))

    def test_get_supported_functions(self):
        res = get_supported_functions()
        self.assertIsInstance(res, dict)
        self.assertIn("square_error", res)
        self.assertIn("grad_square_error", res)

    def common_check_alpha(self, name, fct):
        onx = function_onnx_graph(
            name, target_opset=get_max_opset(),
            dtype=numpy.float32)
        x1 = numpy.random.randn(10, 1).astype(numpy.float32)
        x2 = numpy.random.randn(10, 1).astype(numpy.float32)
        alpha = numpy.random.randn(1).astype(numpy.float32)
        fin = fct(x1, x2, alpha)

        oinf = OnnxInference(onx)
        got = oinf.run({'X1': x1, 'X2': x2, 'alpha': alpha})
        self.assertEqualArray(fin, got['Y'], decimal=5)

        providers = device_to_providers('cpu')
        sess = InferenceSession(onx.SerializeToString(), providers=providers)
        got = sess.run(None, {'X1': x1, 'X2': x2, 'alpha': alpha})
        self.assertEqualArray(fin, got[0], decimal=5)

    def test_grad_onnx_axpy(self):
        self.common_check_alpha("axpy", lambda x1, x2, alpha: x1 * alpha + x2)

    def common_check_2(self, name, fct):
        onx = function_onnx_graph(
            name, target_opset=get_max_opset(),
            dtype=numpy.float32)
        x1 = numpy.random.randn(10, 1).astype(numpy.float32)
        x2 = numpy.random.randn(10, 1).astype(numpy.float32)
        exp_loss, exp_grad = fct(x1, x2)

        oinf = OnnxInference(onx)
        got = oinf.run({'X1': x1, 'X2': x2})
        self.assertEqualArray(exp_loss, got['Y'], decimal=5)
        self.assertEqualArray(exp_grad, got['Z'], decimal=5)

        providers = device_to_providers('cpu')
        sess = InferenceSession(onx.SerializeToString(), providers=providers)
        got = sess.run(None, {'X1': x1, 'X2': x2})
        self.assertEqualArray(exp_loss, got[0], decimal=5)
        self.assertEqualArray(exp_grad, got[1], decimal=5)

    def test_loss_grad_onnx_square_error(self):
        self.common_check_2(
            "grad_loss_square_error",
            lambda x1, x2: (((x1 - x2) ** 2).sum(),
                            (x1 - x2) * (-2)))

    def common_check_3(self, name, fct):
        onx = function_onnx_graph(
            name, target_opset=get_max_opset(),
            dtype=numpy.float32)
        x = numpy.random.randn(10, 1).astype(numpy.float32)
        a = numpy.random.randn(10, 1).astype(numpy.float32).T
        b = numpy.random.randn(10, 1).astype(numpy.float32)
        y = fct(x, a, b)

        code = export2onnx(onx)
        self.assertIn("'OnnxAdd'", code)

        oinf = OnnxInference(onx)
        got = oinf.run({'X': x, 'A': a, 'B': b})
        self.assertEqualArray(y, got['Y'], decimal=5)

        providers = device_to_providers('cpu')
        sess = InferenceSession(onx.SerializeToString(), providers=providers)
        got = sess.run(None, {'X': x, 'A': a, 'B': b})
        self.assertEqualArray(y, got[0], decimal=5)

    def test_linear_regression(self):
        self.common_check_3(
            "linear_regression", lambda x, a, b: x @ a + b)


if __name__ == "__main__":
    unittest.main()
