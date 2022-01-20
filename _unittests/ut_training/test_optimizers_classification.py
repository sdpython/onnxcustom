"""
@brief      test log(time=8s)
"""
import unittest
from pyquickhelper.pycode import ExtTestCase
import numpy
from onnx.helper import set_model_props
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from mlprodict.onnx_conv import to_onnx
from mlprodict.plotting.text_plot import onnx_simple_text_plot
# from mlprodict.onnxrt import OnnxInference
from onnxcustom import __max_supported_opset__ as opset
from onnxcustom.training.sgd_learning_rate import (
    LearningRateSGDNesterov)
from onnxcustom.training.sgd_learning_loss import (
    BaseLearningLoss, NegLogLearningLoss)
try:
    from onnxruntime import TrainingSession
except ImportError:
    # onnxruntime not training
    TrainingSession = None


class TestOptimizersClassification(ExtTestCase):

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_ort_gradient_optimizers_binary(self):
        from onnxcustom.utils.orttraining_helper import add_loss_output
        from onnxcustom.training.optimizers import OrtGradientOptimizer
        X, y = make_classification(  # pylint: disable=W0632
            100, n_features=10, random_state=0)
        X = X.astype(numpy.float32)
        y = y.astype(numpy.int64)
        X_train, _, y_train, __ = train_test_split(X, y)
        reg = SGDClassifier(loss='log')
        reg.fit(X_train, y_train)
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearClassifier'},
                      options={'zipmap': False})
        set_model_props(onx, {'info': 'unit test'})
        onx_loss = add_loss_output(onx, 'log', output_index=1)
        inits = ['intercept', 'coef']
        train_session = OrtGradientOptimizer(
            onx_loss, inits, learning_rate=1e-3)
        self.assertRaise(lambda: train_session.get_state(), AttributeError)
        train_session.fit(X_train, y_train.reshape((-1, 1)), use_numpy=True)
        state_tensors = train_session.get_state()
        self.assertEqual(len(state_tensors), 2)
        r = repr(train_session)
        self.assertIn("OrtGradientOptimizer(model_onnx=", r)
        self.assertIn("learning_rate='invscaling'", r)
        losses = train_session.train_losses_
        self.assertGreater(len(losses), 1)
        self.assertFalse(any(map(numpy.isnan, losses)))

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_ort_gradient_optimizers_fw_nesterov_binary(self):
        from onnxcustom.training.optimizers_partial import OrtGradientForwardBackwardOptimizer
        X, y = make_classification(  # pylint: disable=W0632
            100, n_features=10, random_state=0)
        X = X.astype(numpy.float32)
        y = y.astype(numpy.int64)
        X_train, _, y_train, __ = train_test_split(X, y)
        reg = SGDClassifier(loss='log')
        reg.fit(X_train, y_train)
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'},
                      options={'zipmap': False,
                               'raw_scores': True})
        print(onnx_simple_text_plot(onx))
        set_model_props(onx, {'info': 'unit test'})
        inits = ['coef', 'intercept']

        train_session = OrtGradientForwardBackwardOptimizer(
            onx, inits,
            learning_rate=LearningRateSGDNesterov(
                1e-4, nesterov=False, momentum=0.9),
            learning_loss=NegLogLearningLoss(),
            warm_start=False, max_iter=100, batch_size=10)
        self.assertIsInstance(train_session.learning_loss, BaseLearningLoss)
        self.assertIsInstance(train_session.learning_loss, NegLogLearningLoss)
        self.assertEqual(train_session.learning_loss.eps, 1e-5)
        train_session.fit(X, y)


if __name__ == "__main__":
    unittest.main()
