"""
@brief      test log(time=8s)
"""
import unittest
import logging
import numpy
from onnx import TensorProto
from pyquickhelper.pycode import ExtTestCase
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, SGDRegressor
from mlprodict.onnx_conv import to_onnx
from mlprodict.plotting.text_plot import onnx_simple_text_plot
from mlprodict.onnx_tools.onnx_manipulations import select_model_inputs_outputs
# from mlprodict.onnxrt import OnnxInference
try:
    from onnxruntime import TrainingSession
except ImportError:
    # onnxruntime not training
    TrainingSession = None
from onnxcustom import __max_supported_opset__ as opset
from onnxcustom.training.excs import ConvergenceError


class TestOptimizersNan(ExtTestCase):

    @classmethod
    def setUpClass(cls):
        logger = logging.getLogger('skl2onnx')
        logger.setLevel(logging.WARNING)
        logger = logging.getLogger('onnxcustom')
        logger.setLevel(logging.WARNING)
        logging.basicConfig(level=logging.WARNING)
        ExtTestCase.setUpClass()

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_ort_gradient_optimizers_reg(self):
        self.wtest_ort_gradient_optimizers_reg(False)

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_ort_gradient_optimizers_reg_w(self):
        self.wtest_ort_gradient_optimizers_reg(True)

    def wtest_ort_gradient_optimizers_reg(self, use_weight=False):
        from onnxcustom.utils.orttraining_helper import add_loss_output
        from onnxcustom.training.optimizers import OrtGradientOptimizer
        X = numpy.arange(60).astype(numpy.float32).reshape((-1, 3))
        y = numpy.arange(X.shape[0]).astype(numpy.float32).reshape((-1, 1))
        y[0, 0] += 1
        y[-1, 0] += 1
        w = (numpy.random.rand(X.shape[0]) + 1).astype(numpy.float32)
        X_train, _, y_train, __, w_train, ___ = train_test_split(X, y, w)
        reg = SGDRegressor()
        reg.fit(X_train, y_train.ravel())
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'})
        onx_loss = add_loss_output(
            onx, 'squared_error',
            weight_name='weight' if use_weight else None)
        inits = ['intercept', 'coef']
        inputs = onx_loss.graph.input
        self.assertEqual(len(inputs), 3 if use_weight else 2)
        train_session = OrtGradientOptimizer(
            onx_loss, inits, learning_rate=1e9)
        self.assertRaise(lambda: train_session.get_state(), AttributeError)
        if use_weight:
            self.assertRaise(
                lambda: train_session.fit(
                    X_train, y_train.reshape((-1, 1)),
                    w_train.reshape((-1, 1)), use_numpy=False),
                ConvergenceError)
        else:
            self.assertRaise(
                lambda: train_session.fit(
                    X_train, y_train.reshape((-1, 1)), use_numpy=False),
                ConvergenceError)
        state_tensors = train_session.get_state()
        self.assertEqual(len(state_tensors), 2)
        r = repr(train_session)
        self.assertIn("OrtGradientOptimizer(model_onnx=", r)
        self.assertIn("learning_rate='invscaling'", r)
        losses = train_session.train_losses_
        self.assertGreater(len(losses), 1)
        if any(map(numpy.isnan, losses)):
            raise AssertionError(losses)

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_ort_gradient_optimizers_binary(self):
        self.wtest_ort_gradient_optimizers_binary(False)

    def wtest_ort_gradient_optimizers_binary(self, use_weight=False):
        from onnxcustom.utils.orttraining_helper import add_loss_output
        from onnxcustom.training.optimizers import OrtGradientOptimizer
        X = numpy.arange(60).astype(numpy.float32).reshape((-1, 3))
        y = numpy.arange(X.shape[0]).astype(
            numpy.float32).reshape((-1, 1)) > 10
        X = X.astype(numpy.float32)
        y = y.astype(numpy.int64)
        y[0, 0] = 0
        y[-1, 0] = 1
        w = (numpy.random.rand(X.shape[0]) + 1).astype(numpy.float32)
        X_train, _, y_train, __, w_train, ___ = train_test_split(X, y, w)
        reg = SGDClassifier(loss='log')
        reg.fit(X_train, y_train.ravel())
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearClassifier'},
                      options={'zipmap': False})
        onx_loss = add_loss_output(
            onx, 'log', output_index=1,
            weight_name='weight' if use_weight else None)
        inits = ['intercept', 'coef']
        inputs = onx_loss.graph.input
        self.assertEqual(len(inputs), 3 if use_weight else 2)
        dt = inputs[1].type.tensor_type.elem_type
        self.assertEqual(TensorProto.INT64, dt)  # pylint: disable=E1101
        train_session = OrtGradientOptimizer(
            onx_loss, inits, learning_rate=1e9)
        self.assertRaise(lambda: train_session.get_state(), AttributeError)
        if use_weight:
            train_session.fit(
                X_train, y_train.reshape((-1, 1)),
                w_train.reshape((-1, 1)), use_numpy=False)
        else:
            train_session.fit(
                X_train, y_train.reshape((-1, 1)), use_numpy=False)
        state_tensors = train_session.get_state()
        self.assertEqual(len(state_tensors), 2)
        r = repr(train_session)
        self.assertIn("OrtGradientOptimizer(model_onnx=", r)
        self.assertIn("learning_rate='invscaling'", r)
        losses = train_session.train_losses_
        self.assertGreater(len(losses), 1)
        if any(map(numpy.isnan, losses)):
            raise AssertionError(losses)

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_ort_gradient_optimizers_fw_sgd_binary(self):
        self.wtest_ort_gradient_optimizers_fw_sgd_binary(False)

    def wtest_ort_gradient_optimizers_fw_sgd_binary(self, use_weight):
        from onnxcustom.training.optimizers_partial import (
            OrtGradientForwardBackwardOptimizer)
        from onnxcustom.training.sgd_learning_rate import (
            LearningRateSGD)
        from onnxcustom.training.sgd_learning_loss import NegLogLearningLoss
        X = numpy.arange(60).astype(numpy.float32).reshape((-1, 3))
        y = numpy.arange(X.shape[0]).astype(
            numpy.float32).reshape((-1, 1)) > 10
        X = X.astype(numpy.float32)
        y = y.astype(numpy.int64)
        y[0, 0] = 0
        y[-1, 0] = 1
        w = (numpy.random.rand(y.shape[0]) + 1).astype(numpy.float32)
        X_train, _, y_train, __, w_train, ___ = train_test_split(X, y, w)
        reg = SGDClassifier(loss='log')
        if use_weight:
            reg.fit(X_train, y_train.ravel(),
                    sample_weight=w_train.astype(numpy.float64))
        else:
            reg.fit(X_train, y_train.ravel())
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'},
                      options={'zipmap': False,
                               'raw_scores': True})
        onx = select_model_inputs_outputs(onx, outputs=['score'])
        self.assertIn("output: name='score'",
                      onnx_simple_text_plot(onx))
        inits = ['coef', 'intercept']

        train_session = OrtGradientForwardBackwardOptimizer(
            onx, inits, weight_name='weight' if use_weight else None,
            learning_rate=LearningRateSGD(1e10),
            learning_loss=NegLogLearningLoss(),
            warm_start=False, max_iter=100, batch_size=10,
            enable_logging=False)
        self.assertIsInstance(train_session.learning_loss, NegLogLearningLoss)
        self.assertEqual(train_session.learning_loss.eps, 1e-5)
        y_train = y_train.reshape((-1, 1))
        if use_weight:
            train_session.fit(X_train, y_train, w_train.reshape((-1, 1)))
        else:
            train_session.fit(X_train, y_train)
        losses = train_session.train_losses_
        self.assertGreater(len(losses), 1)
        if any(map(numpy.isnan, losses)):
            raise AssertionError(losses)

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_ort_gradient_optimizers_fw_sgd_reg(self):
        self.wtest_ort_gradient_optimizers_fw_sgd_reg(False)

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_ort_gradient_optimizers_fw_sgd_reg_weight(self):
        self.wtest_ort_gradient_optimizers_fw_sgd_reg(True)

    def wtest_ort_gradient_optimizers_fw_sgd_reg(self, use_weight):
        from onnxcustom.training.optimizers_partial import (
            OrtGradientForwardBackwardOptimizer)
        from onnxcustom.training.sgd_learning_rate import (
            LearningRateSGD)
        from onnxcustom.training.sgd_learning_loss import SquareLearningLoss
        X = numpy.arange(60).astype(numpy.float32).reshape((-1, 3))
        y = numpy.arange(X.shape[0]).astype(numpy.float32).reshape((-1, 1))
        y[0, 0] += 1
        y[-1, 0] += 1
        w = (numpy.random.rand(y.shape[0]) + 1).astype(numpy.float32)
        X_train, _, y_train, __, w_train, ___ = train_test_split(X, y, w)
        reg = SGDRegressor()
        if use_weight:
            reg.fit(X_train, y_train.ravel(),
                    sample_weight=w_train.astype(numpy.float64))
        else:
            reg.fit(X_train, y_train.ravel())
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'})
        inits = ['coef', 'intercept']

        train_session = OrtGradientForwardBackwardOptimizer(
            onx, inits, weight_name='weight' if use_weight else None,
            learning_rate=LearningRateSGD(1e10),
            learning_loss=SquareLearningLoss(),
            warm_start=False, max_iter=100, batch_size=10,
            enable_logging=False)
        self.assertIsInstance(train_session.learning_loss, SquareLearningLoss)
        y_train = y_train.reshape((-1, 1))
        if use_weight:
            self.assertRaise(
                lambda: train_session.fit(
                    X_train, y_train, w_train.reshape((-1, 1))),
                ConvergenceError)
        else:
            self.assertRaise(
                lambda: train_session.fit(X_train, y_train),
                ConvergenceError)
        losses = train_session.train_losses_
        self.assertLess(len(losses), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
