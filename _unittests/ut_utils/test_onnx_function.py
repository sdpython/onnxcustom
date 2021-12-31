"""
@brief      test log(time=9s)
"""
import unittest
import numpy
from onnxruntime import InferenceSession, SessionOptions
from pyquickhelper.pycode import ExtTestCase
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnx_tools.onnx_export import export2onnx
from onnxcustom import get_max_opset
from onnxcustom.utils.onnx_function import (
    function_onnx_graph, get_supported_functions)
from onnxcustom.utils.onnxruntime_helper import device_to_providers


class TestOnnxFunction(ExtTestCase):

    def common_check(self, name, fct, weight_name=None):
        onx = function_onnx_graph(
            name, target_opset=get_max_opset(),
            dtype=numpy.float32, weight_name=weight_name)
        expected = numpy.random.randn(10, 1).astype(numpy.float32)
        predicted = numpy.random.randn(10, 1).astype(numpy.float32)
        w = numpy.random.rand(10).astype(numpy.float32)
        if weight_name is None:
            fin = fct(expected, predicted)
        else:
            fin = fct(expected, predicted, w)

        oinf = OnnxInference(onx)
        if weight_name is None:
            got = oinf.run({'X1': expected, 'X2': predicted})
        else:
            got = oinf.run({'X1': expected, 'X2': predicted, 'weight': w})
        self.assertEqualArray(fin, got['Y'], decimal=5)
        if weight_name is not None:
            got = oinf.run({'X1': expected, 'X2': predicted})
            fin1 = fct(
                expected, predicted, numpy.array([1], dtype=expected.dtype))
            self.assertEqualArray(fin1, got['Y'], decimal=5)

        providers = device_to_providers('cpu')
        so = SessionOptions()
        so.log_severity_level = 4
        sess = InferenceSession(
            onx.SerializeToString(), so, providers=providers)
        if weight_name is None:
            got = sess.run(None, {'X1': expected, 'X2': predicted})
        else:
            got = sess.run(
                None, {'X1': expected, 'X2': predicted, 'weight': w})
        self.assertEqualArray(fin, got[0], decimal=5)
        if weight_name is not None:
            got = sess.run(None, {'X1': expected, 'X2': predicted})
            fin1 = fct(
                expected, predicted, numpy.array([1], dtype=expected.dtype))
            self.assertEqualArray(fin1, got[0], decimal=5)

    def test_exc(self):
        self.assertRaise(lambda: function_onnx_graph("H"))

    def test_onnx_square_error(self):
        self.common_check(
            "square_error", lambda x1, x2: ((x1 - x2) ** 2).sum())

    def test_onnx_square_error_w(self):
        self.common_check(
            "square_error", lambda x1, x2, w:
                ((x1 - x2) ** 2 * w.reshape((-1, 1))).sum(),
            weight_name='weight')

    def test_grad_onnx_square_error(self):
        self.common_check("grad_square_error", lambda x1, x2: (x1 - x2) * (-2))

    def test_grad_onnx_square_error_w(self):
        self.common_check(
            "grad_square_error", lambda x1, x2, w:
                (x1 - x2) * (-2) * w.reshape((-1, 1)),
            weight_name='weight')

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
        so = SessionOptions()
        so.log_severity_level = 4
        sess = InferenceSession(
            onx.SerializeToString(), so, providers=providers)
        got = sess.run(None, {'X1': x1, 'X2': x2, 'alpha': alpha})
        self.assertEqualArray(fin, got[0], decimal=5)

    def test_grad_onnx_axpy(self):
        self.common_check_alpha("axpy", lambda x1, x2, alpha: x1 * alpha + x2)

    def common_check_2(self, name, fct, weight_name=None):
        onx = function_onnx_graph(
            name, target_opset=get_max_opset(),
            dtype=numpy.float32, weight_name=weight_name)
        x1 = numpy.random.randn(10, 1).astype(numpy.float32)
        x2 = numpy.random.randn(10, 1).astype(numpy.float32)
        w = numpy.random.rand(10).astype(numpy.float32)
        if weight_name is None:
            exp_loss, exp_grad = fct(x1, x2)
        else:
            exp_loss, exp_grad = fct(x1, x2, w)

        oinf = OnnxInference(onx)
        if weight_name is None:
            got = oinf.run({'X1': x1, 'X2': x2})
        else:
            got = oinf.run({'X1': x1, 'X2': x2, 'weight': w})
        self.assertEqualArray(exp_loss, got['Y'], decimal=5)
        self.assertEqualArray(exp_grad, got['Z'], decimal=5)

        providers = device_to_providers('cpu')
        so = SessionOptions()
        so.log_severity_level = 4
        sess = InferenceSession(
            onx.SerializeToString(), so, providers=providers)
        if weight_name is None:
            got = sess.run(None, {'X1': x1, 'X2': x2})
        else:
            got = sess.run(None, {'X1': x1, 'X2': x2, 'weight': w})
        self.assertEqualArray(exp_loss, got[0], decimal=5)
        self.assertEqualArray(exp_grad, got[1], decimal=5)
        if weight_name is not None:
            got = sess.run(None, {'X1': x1, 'X2': x2})
            exp_loss, exp_grad = fct(
                x1, x2, numpy.array([1], dtype=x1.dtype))
            self.assertEqualArray(exp_loss, got[0], decimal=5)
            self.assertEqualArray(exp_grad, got[1], decimal=5)

    def test_loss_grad_onnx_square_error(self):
        self.common_check_2(
            "grad_loss_square_error",
            lambda x1, x2: (((x1 - x2) ** 2).sum(),
                            (x1 - x2) * (-2)))

    def test_loss_grad_onnx_square_error_w(self):
        self.common_check_2(
            "grad_loss_square_error",
            lambda x1, x2, w: (((x1 - x2) ** 2 * w.reshape((-1, 1))).sum(),
                               (x1 - x2) * (-2) * w.reshape((-1, 1))),
            weight_name='weight')

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
        so = SessionOptions()
        so.log_severity_level = 4
        sess = InferenceSession(
            onx.SerializeToString(), so, providers=providers)
        got = sess.run(None, {'X': x, 'A': a, 'B': b})
        self.assertEqualArray(y, got[0], decimal=5)

    def test_linear_regression(self):
        self.common_check_3(
            "linear_regression", lambda x, a, b: x @ a + b)

    def test_251(self):
        onx = function_onnx_graph(
            "grad_loss_square_error",
            target_opset=get_max_opset(),
            dtype=numpy.float32, weight_name='weight')
        expected = numpy.random.randn(25, 1).astype(numpy.float32)
        predicted = numpy.random.randn(25, 1).astype(numpy.float32)

        oinf = OnnxInference(onx)
        got1 = oinf.run({'X1': expected, 'X2': predicted})
        so = SessionOptions()
        so.log_severity_level = 4
        sess = InferenceSession(onx.SerializeToString(), so)
        got2 = sess.run(None, {'X1': expected, 'X2': predicted})
        self.assertEqualArray(got1['Y'], got2[0], decimal=5)
        self.assertEqualArray(got1['Z'], got2[1])

    def common_unary(self, name, fct):
        onx = function_onnx_graph(
            name, target_opset=get_max_opset(),
            dtype=numpy.float32)
        x = numpy.random.randn(10, 1).astype(numpy.float32)
        fin = fct(x)

        oinf = OnnxInference(onx)
        got = oinf.run({'X': x})
        self.assertEqualArray(fin, got['Y'], decimal=5)

        providers = device_to_providers('cpu')
        so = SessionOptions()
        so.log_severity_level = 4
        sess = InferenceSession(
            onx.SerializeToString(), so, providers=providers)
        got = sess.run(None, {'X': x})
        self.assertEqualArray(fin, got[0], decimal=5)

    def test_copy(self):
        self.common_unary("copy", lambda x: x)

    def test_zero(self):
        self.common_unary("zero", lambda x: x * 0)

    def common_check_alpha_beta(self, name, fct):
        onx = function_onnx_graph(
            name, target_opset=get_max_opset(),
            dtype=numpy.float32)
        x1 = numpy.random.randn(10, 1).astype(numpy.float32)
        x2 = numpy.random.randn(10, 1).astype(numpy.float32)
        g = numpy.random.randn(10, 1).astype(numpy.float32)
        alpha = numpy.random.randn(1).astype(numpy.float32)
        beta = numpy.random.randn(1).astype(numpy.float32)
        y, z = fct(x1, x2, g, alpha, beta)

        oinf = OnnxInference(onx)
        got = oinf.run({'X1': x1, 'X2': x2, 'alpha': alpha,
                        'beta': beta, 'G': g})
        self.assertEqualArray(y, got['Y'], decimal=5)
        self.assertEqualArray(z, got['Z'], decimal=5)

        providers = device_to_providers('cpu')
        so = SessionOptions()
        so.log_severity_level = 4
        sess = InferenceSession(
            onx.SerializeToString(), so, providers=providers)
        got = sess.run(None, {'X1': x1, 'X2': x2, 'alpha': alpha,
                              'beta': beta, 'G': g})
        self.assertEqualArray(y, got[0], decimal=5)
        self.assertEqualArray(z, got[1], decimal=5)

    def test_grad_onnx_axpyw(self):
        self.common_check_alpha_beta(
            "axpyw", lambda x1, x2, g, alpha, beta:
                (x1 * alpha + x2 + beta * g,
                 x1 * alpha + beta * g))

    def test_grad_onnx_axpyw2(self):
        self.common_check_alpha_beta(
            "axpyw2", lambda x1, x2, g, alpha, beta:
                (x1 * alpha + x2 + beta * (x1 * alpha + beta * g),
                 x1 * alpha + beta * g))


if __name__ == "__main__":
    unittest.main()
