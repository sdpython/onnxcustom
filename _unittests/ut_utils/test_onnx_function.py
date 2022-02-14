"""
@brief      test log(time=9s)
"""
import unittest
import numpy
from scipy.special import expit  # pylint: disable=E0611
from onnxruntime import InferenceSession, SessionOptions
from pyquickhelper.pycode import ExtTestCase
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnx_tools.onnx_export import export2onnx
from onnxcustom import get_max_opset
from onnxcustom.utils.onnx_function import (
    function_onnx_graph, get_supported_functions)
from onnxcustom.utils.onnxruntime_helper import device_to_providers
from onnxcustom.utils.onnx_rewriter import unreduced_onnx_loss


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

    def common_check_2(self, name, fct, weight_name=None,
                       verbose=0, classification=False, rnd=True,
                       **kwargs):
        onx = function_onnx_graph(
            name, target_opset=get_max_opset(),
            dtype=numpy.float32, weight_name=weight_name,
            **kwargs)
        if verbose > 0:
            with open(name + ".onnx", "wb") as f:
                f.write(onx.SerializeToString())
        if classification:
            N = 10
            p = numpy.random.randn(N, 1).astype(numpy.float32)
            p[0, :] = 0
            p[1, :] = 100
            p[2, :] = -100
            p[3, :] = 1
            p[4, :] = -1
            y = (numpy.random.randn(N, 1).astype(numpy.float32) > 0).astype(
                numpy.int64)
            x2 = p
            x1 = y
        else:
            if rnd:
                x1 = numpy.random.randn(10, 1).astype(numpy.float32)
                x2 = numpy.random.randn(10, 1).astype(numpy.float32)
            else:
                x1 = numpy.zeros((10, 1), dtype=numpy.float32)
                x2 = numpy.zeros((10, 1), dtype=numpy.float32) + 1
        if rnd:
            w = numpy.random.rand(10).astype(numpy.float32)
        else:
            w = numpy.zeros(10, dtype=numpy.float32) + 0.2
        if weight_name is None:
            exp_loss, exp_grad = fct(x1, x2)
        else:
            exp_loss, exp_grad = fct(x1, x2, w.reshape((-1, 1)))

        oinf = OnnxInference(onx)
        run_params = dict(verbose=verbose, fLOG=print) if verbose > 0 else {}
        if verbose > 0:
            print("\n+++++ name(1)=%r" % name)
        if weight_name is None:
            got = oinf.run({'X1': x1, 'X2': x2}, **run_params)
        else:
            got = oinf.run({'X1': x1, 'X2': x2, 'weight': w}, **run_params)
        self.assertEqual(len(exp_grad.shape), 2)
        self.assertEqual(exp_grad.shape[-1], 1)
        self.assertEqualArray(exp_grad, got['Z'], decimal=5)
        self.assertEqualArray(exp_loss, got['Y'], decimal=5)

        providers = device_to_providers('cpu')
        so = SessionOptions()
        so.log_severity_level = 0 if verbose > 0 else 4
        so.log_verbosity_level = 0 if verbose > 0 else 4
        sess = InferenceSession(
            onx.SerializeToString(), so, providers=providers)
        if verbose > 0:
            print("+++ run")
        if weight_name is None:
            got = sess.run(None, {'X1': x1, 'X2': x2})
        else:
            got = sess.run(None, {'X1': x1, 'X2': x2, 'weight': w})
        self.assertEqualArray(exp_loss, got[0], decimal=5)
        self.assertEqualArray(exp_grad, got[1], decimal=5)
        if weight_name is not None:
            if verbose > 0:
                print("+++ run*")
            got = sess.run(None, {'X1': x1, 'X2': x2})
            exp_loss2, exp_grad2 = fct(
                x1, x2, numpy.array([1], dtype=x1.dtype))
            self.assertEqualArray(exp_loss2, got[0], decimal=5)
            self.assertEqualArray(exp_grad2, got[1], decimal=5)

        if 'grad' in name:
            rew = unreduced_onnx_loss(onx)
            if 'ReduceSum' in str(rew):
                raise AssertionError("Isse with:\n%r" % rew)
            if verbose > 0:
                with open(name + ".unreduced.onnx", "wb") as f:
                    f.write(rew.SerializeToString())

            if verbose > 0:
                print("\n+++++ name(2)=%r" % name)
            oinf = OnnxInference(rew)
            if weight_name is None:
                got = oinf.run({'X1': x1, 'X2': x2}, **run_params)
            else:
                got = oinf.run({'X1': x1, 'X2': x2, 'weight': w}, **run_params)
            score = got['score']
            self.assertEqual(len(score.shape), 2)
            self.assertEqual(score.shape[0], 10)
            self.assertEqual(score.shape[1], 1)
            self.assertEqualFloat(exp_loss, score.sum())

            sess = InferenceSession(
                rew.SerializeToString(), so, providers=providers)
            if verbose > 0:
                print("+++ run")
            if weight_name is None:
                got = sess.run(None, {'X1': x1, 'X2': x2})
            else:
                got = sess.run(None, {'X1': x1, 'X2': x2, 'weight': w})
            score = got[0]
            self.assertEqual(len(score.shape), 2)
            self.assertEqual(score.shape[0], 10)
            self.assertEqual(score.shape[1], 1)
            self.assertEqualFloat(exp_loss, score.sum())

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

    def test_loss_grad_onnx_absolute_error(self):
        self.common_check_2(
            "grad_loss_absolute_error",
            lambda x1, x2: (numpy.abs(x1 - x2).sum(),
                            numpy.sign(x1 - x2)))

    def test_loss_grad_onnx_absolute_error_w(self):
        self.common_check_2(
            "grad_loss_absolute_error",
            lambda x1, x2, w: ((numpy.abs(x1 - x2) * w.reshape((-1, 1))).sum(),
                               numpy.sign(x1 - x2) * w.reshape((-1, 1))),
            weight_name='weight')

    def test_loss_grad_onnx_absolute_error_w_norand(self):
        self.common_check_2(
            "grad_loss_absolute_error",
            lambda x1, x2, w: ((numpy.abs(x1 - x2) * w.reshape((-1, 1))).sum(),
                               numpy.sign(x1 - x2) * w.reshape((-1, 1))),
            weight_name='weight', verbose=0, rnd=False)

    def test_loss_grad_onnx_elastic_error(self):
        self.common_check_2(
            "grad_loss_elastic_error",
            lambda x1, x2: (
                numpy.abs(x1 - x2).sum() * 0.1 + ((x1 - x2) ** 2).sum() * 0.9,
                numpy.sign(x1 - x2) * 0.1 - 2 * 0.9 * (x1 - x2)
            ),
            l1_weight=0.1, l2_weight=0.9, verbose=0)

    def test_loss_grad_onnx_elastic_error_w(self):
        self.common_check_2(
            "grad_loss_elastic_error",
            lambda x1, x2, w: (
                (numpy.abs(x1 - x2) * w.reshape((-1, 1))).sum() * 0.1 +
                ((x1 - x2) ** 2 * w.reshape((-1, 1))).sum() * 0.9,
                numpy.sign(x1 - x2) * w.reshape((-1, 1)) * 0.1 +
                (x1 - x2) * (-2) * w.reshape((-1, 1)) * 0.9
            ),
            weight_name='weight', l1_weight=0.1, l2_weight=0.9)

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

    def common_check_1(self, name, fct, weight_name=None, **kwargs):
        onx = function_onnx_graph(
            name, target_opset=get_max_opset(),
            dtype=numpy.float32, weight_name=weight_name,
            **kwargs)
        x = numpy.random.randn(10, 1).astype(numpy.float32)
        exp_loss, exp_grad = fct(x)

        oinf = OnnxInference(onx)
        got = oinf.run({'X': x})
        self.assertEqualArray(exp_loss, got['Y'], decimal=5)
        self.assertEqualArray(exp_grad, got['Z'], decimal=5)

        providers = device_to_providers('cpu')
        so = SessionOptions()
        so.log_severity_level = 4
        sess = InferenceSession(
            onx.SerializeToString(), so, providers=providers)
        got = sess.run(None, {'X': x})
        self.assertEqualArray(exp_loss, got[0], decimal=5)
        self.assertEqualArray(exp_grad, got[1], decimal=5)

    def test_penalty_grad_onnx_elastic_error(self):
        self.common_check_1(
            "grad_penalty_elastic_error",
            lambda x: (
                numpy.abs(x).sum() * 0.1 + ((x) ** 2).sum() * 0.9,
                numpy.sign(x) * 0.1 + 2 * 0.9 * x
            ),
            l1_weight=0.1, l2_weight=0.9)

    def test_penalty_3(self):
        loss = numpy.random.randn(1, 1).astype(numpy.float32)
        w1 = numpy.random.randn(10, 1).astype(numpy.float32)
        w2 = numpy.random.randn(5, 1).astype(numpy.float32)

        def fct(x):
            return numpy.abs(x).sum() * 0.1 + ((x) ** 2).sum() * 0.9

        exp_loss = loss + fct(w1) + fct(w2)

        onx = function_onnx_graph(
            'n_penalty_elastic_error', target_opset=get_max_opset(),
            dtype=numpy.float32, n_tensors=2,
            l1_weight=0.1, l2_weight=0.9)

        oinf = OnnxInference(onx)
        got = oinf.run({'loss': loss, 'W0': w1, 'W1': w2})
        self.assertEqualArray(exp_loss.reshape((-1, )), got['Y'], decimal=5)

        providers = device_to_providers('cpu')
        so = SessionOptions()
        so.log_severity_level = 4
        sess = InferenceSession(
            onx.SerializeToString(), so, providers=providers)
        got = sess.run(None, {'loss': loss, 'W0': w1, 'W1': w2})
        self.assertEqualArray(exp_loss.reshape((-1, )), got[0], decimal=5)

    def test_penalty_3w(self):
        loss = numpy.random.randn(1, 1).astype(numpy.float32)
        w1 = numpy.random.randn(10, 1).astype(numpy.float32)
        w2 = numpy.random.randn(5, 1).astype(numpy.float32)

        def fct(x):
            return numpy.abs(x).sum() * 0.1 + ((x) ** 2).sum() * 0.9

        exp_loss = loss + fct(w1) + fct(w2)

        onx = function_onnx_graph(
            'n_penalty_elastic_error', target_opset=get_max_opset(),
            dtype=numpy.float32, n_tensors=2,
            l1_weight=0.1, l2_weight=0.9, weight_name='weight')

        oinf = OnnxInference(onx)
        got = oinf.run({'loss': loss, 'W0': w1, 'W1': w2})
        self.assertEqualArray(exp_loss.reshape((-1, )), got['Y'], decimal=5)

        providers = device_to_providers('cpu')
        so = SessionOptions()
        so.log_severity_level = 4
        sess = InferenceSession(
            onx.SerializeToString(), so, providers=providers)
        got = sess.run(None, {'loss': loss, 'W0': w1, 'W1': w2})
        self.assertEqualArray(exp_loss.reshape((-1, )), got[0], decimal=5)

    def test_penalty_update(self):
        x = numpy.random.randn(10, 1).astype(numpy.float32)

        def fct(x):
            return numpy.sign(x) * 0.1 + (x * 0.9 * 2)

        exp_loss = x - fct(x)

        onx = function_onnx_graph(
            'update_penalty_elastic_error', target_opset=get_max_opset(),
            dtype=numpy.float32, l1=0.1, l2=0.9)

        oinf = OnnxInference(onx)
        got = oinf.run({'X': x})
        self.assertEqualArray(exp_loss, got['Y'], decimal=5)

        providers = device_to_providers('cpu')
        so = SessionOptions()
        so.log_severity_level = 4
        sess = InferenceSession(
            onx.SerializeToString(), so, providers=providers)
        got = sess.run(None, {'X': x})
        self.assertEqualArray(exp_loss, got[0], decimal=5)

    def test_grad_sigmoid_neg_log_loss_error(self):

        def loss(x1, x2, eps=1e-5):
            pr = expit(x2)
            cl = numpy.clip(pr, eps, 1 - eps)
            lo = - (1 - x1) * numpy.log(1 - cl) - x1 * numpy.log(cl)
            return lo

        self.common_check_2(
            "grad_sigmoid_neg_log_loss_error",
            lambda x1, x2: (loss(x1, x2).sum(), expit(x2) - x1),
            classification=True, verbose=0)

    def test_grad_sigmoid_neg_log_loss_error_weight(self):

        def loss(x1, x2, w, eps=1e-5):
            pr = expit(x2)
            cl = numpy.clip(pr, eps, 1 - eps)
            lo = - (1 - x1) * numpy.log(1 - cl) - x1 * numpy.log(cl)
            return lo * w.reshape((-1, 1))

        def grad(x1, x2, w):
            r = - (x1 - expit(x2)) * w.reshape((-1, 1))
            return r

        self.common_check_2(
            "grad_sigmoid_neg_log_loss_error",
            lambda x1, x2, w:
                (loss(x1, x2, w).sum(),
                 grad(x1, x2, w)),
            classification=True, weight_name='weight', verbose=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
