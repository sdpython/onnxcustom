"""
@brief      test log(time=8s)
"""
import unittest
import logging
import numpy
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDRegressor
from mlprodict.onnx_conv import to_onnx
# from mlprodict.onnxrt import OnnxInference
# from mlprodict.plotting.text_plot import onnx_simple_text_plot
try:
    from onnxruntime import TrainingSession
except ImportError:
    # onnxruntime not training
    TrainingSession = None
from onnxcustom import __max_supported_opset__ as opset
# from onnxcustom.training.excs import ConvergenceError
from onnxcustom.utils.onnx_helper import onnx_rename_weights


class TestOptimizersGrid(ExtTestCase):

    @classmethod
    def setUpClass(cls):
        logger = logging.getLogger('skl2onnx')
        logger.setLevel(logging.WARNING)
        logger = logging.getLogger('onnxcustom')
        logger.setLevel(logging.WARNING)
        logging.basicConfig(level=logging.WARNING)

    @unittest.skipIf(TrainingSession is None, reason="not training")
    @ignore_warnings((ConvergenceWarning, DeprecationWarning))
    def test_ort_gradient_optimizers_score_reg(self):
        self.wtest_ort_gradient_optimizers_score_reg(False)

    def wtest_ort_gradient_optimizers_score_reg(self, use_weight=False):
        from onnxcustom.training.optimizers_partial import (
            OrtGradientForwardBackwardOptimizer)
        from onnxcustom.training.sgd_learning_rate import (
            LearningRateSGD)
        from onnxcustom.training.sgd_learning_loss import SquareLearningLoss
        X = numpy.arange(60).astype(numpy.float32).reshape((-1, 3))
        y = numpy.arange(X.shape[0]).astype(numpy.float32).reshape((-1, 1))
        y[0, 0] += 1
        y[-1, 0] += 1
        w = (numpy.random.rand(X.shape[0]) + 1).astype(numpy.float32)
        X_train, _, y_train, __, w_train, ___ = train_test_split(X, y, w)
        reg = SGDRegressor(max_iter=20)
        reg.fit(X_train, y_train.ravel())
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'})
        onx = onnx_rename_weights(onx)
        inits = ['I0_coef', 'I1_intercept']

        model = OrtGradientForwardBackwardOptimizer(
            onx, inits, weight_name='weight' if use_weight else None,
            learning_rate=LearningRateSGD(1e-3),
            learning_loss=SquareLearningLoss(),
            warm_start=False, max_iter=20, batch_size=10)
        if use_weight:
            model.fit(X_train, y_train, w_train)
            score = model.score(X_train, y_train, w_train)
        else:
            model.fit(X_train, y_train)
            score = model.score(X_train, y_train)
        self.assertEqual(score.shape, y_train.shape)
        self.assertFalse(any(map(numpy.isnan, score)))

    @unittest.skipIf(TrainingSession is None, reason="not training")
    @ignore_warnings((ConvergenceWarning, DeprecationWarning))
    def test_ort_gradient_optimizers_grid_reg(self):
        self.wtest_ort_gradient_optimizers_grid_reg(False)

    def wtest_ort_gradient_optimizers_grid_reg(self, use_weight=False):
        from onnxcustom.training.optimizers_partial import (
            OrtGradientForwardBackwardOptimizer)
        from onnxcustom.training.sgd_learning_rate import (
            LearningRateSGD)
        from onnxcustom.training.sgd_learning_loss import SquareLearningLoss
        values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
        X = numpy.arange(60).astype(numpy.float32).reshape((-1, 3))
        y = numpy.arange(X.shape[0]).astype(numpy.float32).reshape((-1, 1))
        y[0, 0] += 1
        y[-1, 0] += 1
        X_train, _, y_train, __ = train_test_split(X, y)
        reg = GridSearchCV(
            SGDRegressor(max_iter=20),
            param_grid={'eta0': values})
        reg.fit(X_train, y_train.ravel())
        self.assertIsInstance(reg.best_params_, dict)
        self.assertIn('eta0', reg.best_params_)
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'})
        onx = onnx_rename_weights(onx)
        inits = ['I0_coef', 'I1_intercept']

        grid = GridSearchCV(
            OrtGradientForwardBackwardOptimizer(
                onx, inits, weight_name='weight' if use_weight else None,
                learning_rate=LearningRateSGD(1e10),
                learning_loss=SquareLearningLoss(),
                warm_start=False, max_iter=20, batch_size=10,
                enable_logging=False),
            param_grid={'learning_rate': values})
        if use_weight:
            grid.fit(X_train, y_train)
        else:
            grid.fit(X_train, y_train)
        print(grid.best_params_)


if __name__ == "__main__":
    unittest.main()
