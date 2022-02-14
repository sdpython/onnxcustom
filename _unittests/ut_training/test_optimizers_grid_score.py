"""
@brief      test log(time=20s)
"""
import unittest
import logging
import numbers
import numpy
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDRegressor, SGDClassifier
from sklearn.metrics import log_loss, make_scorer, mean_squared_error
from mlprodict.onnx_conv import to_onnx
from mlprodict.onnx_tools.onnx_manipulations import select_model_inputs_outputs
# from mlprodict.onnxrt import OnnxInference
# from mlprodict.plotting.text_plot import onnx_simple_text_plot
try:
    from onnxruntime import TrainingSession
except ImportError:
    # onnxruntime not training
    TrainingSession = None
from onnxcustom import __max_supported_opset__ as opset
from onnxcustom.training.excs import ConvergenceWarning as MyConvergenceWarning
from onnxcustom.utils.onnx_helper import onnx_rename_weights


class TestOptimizersGrid(ExtTestCase):

    @classmethod
    def setUpClass(cls):
        logger = logging.getLogger('skl2onnx')
        logger.setLevel(logging.WARNING)
        logger = logging.getLogger('onnxcustom')
        logger.setLevel(logging.WARNING)
        logging.basicConfig(level=logging.WARNING)
        ExtTestCase.setUpClass()

    @unittest.skipIf(TrainingSession is None, reason="not training")
    @ignore_warnings((ConvergenceWarning, DeprecationWarning))
    def test_ort_gradient_optimizers_score_reg(self):
        self.wtest_ort_gradient_optimizers_score_reg(False)

    @unittest.skipIf(TrainingSession is None, reason="not training")
    @ignore_warnings((
        ConvergenceWarning, DeprecationWarning, MyConvergenceWarning))
    def test_ort_gradient_optimizers_score_reg_w(self):
        self.wtest_ort_gradient_optimizers_score_reg(True)

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
            learning_rate=LearningRateSGD(1e-4),
            learning_loss=SquareLearningLoss(),
            warm_start=False, max_iter=20, batch_size=10)
        if use_weight:
            model.fit(X_train, y_train, w_train)
            losses = model.losses(X_train, y_train, w_train)
            score = model.score(X_train, y_train, w_train)
        else:
            model.fit(X_train, y_train)
            losses = model.losses(X_train, y_train)
            score = model.score(X_train, y_train)
        self.assertEqual(losses.shape[0], y_train.shape[0])
        self.assertFalse(any(map(numpy.isnan, losses)))
        self.assertIsInstance(score, numbers.Number)
        params = model.get_params()
        self.assertIsInstance(params['device'], str)

    @unittest.skipIf(TrainingSession is None, reason="not training")
    @ignore_warnings((ConvergenceWarning, DeprecationWarning,
                      MyConvergenceWarning, UserWarning))
    def test_ort_gradient_optimizers_grid_reg(self):
        self.wtest_ort_gradient_optimizers_grid_reg(False)

    def wtest_ort_gradient_optimizers_grid_reg(self, use_weight=False):
        from onnxcustom.training.optimizers_partial import (
            OrtGradientForwardBackwardOptimizer)
        from onnxcustom.training.sgd_learning_rate import (
            LearningRateSGD)
        from onnxcustom.training.sgd_learning_loss import SquareLearningLoss
        values = [1e-6, 1e-5, 5e-5, 8e-5, 1e-4, 2e-4, 5e-4,
                  1e-3, 1e-2, 1e-1, 1]
        X = numpy.random.randn(30, 3).astype(numpy.float32)
        y = X.sum(axis=1).reshape((-1, 1))
        y += numpy.random.randn(y.shape[0]).astype(
            numpy.float32).reshape((-1, 1)) / 10
        X_train, _, y_train, __ = train_test_split(X, y)
        scorer = make_scorer(
            lambda y_true, y_pred: (
                -mean_squared_error(y_true, y_pred)))  # pylint: disable=E1130
        reg = GridSearchCV(
            SGDRegressor(max_iter=20),
            param_grid={'eta0': values},
            scoring=scorer, cv=3, error_score='raise')
        reg.fit(X_train, y_train.ravel())
        self.assertIsInstance(reg.best_params_, dict)
        self.assertIn('eta0', reg.best_params_)
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'})
        onx = onnx_rename_weights(onx)
        inits = ['I0_coef', 'I1_intercept']

        cvalues = [LearningRateSGD(v) for v in values]
        grid = GridSearchCV(
            OrtGradientForwardBackwardOptimizer(
                onx, inits, weight_name='weight' if use_weight else None,
                learning_rate=LearningRateSGD(1e-4),
                learning_loss=SquareLearningLoss(),
                warm_start=False, max_iter=20, batch_size=10,
                enable_logging=False, exc=False),
            param_grid={'learning_rate': cvalues}, cv=3)
        if use_weight:
            grid.fit(X_train, y_train)
        else:
            grid.fit(X_train, y_train)
        self.assertIsInstance(grid.best_params_, dict)
        self.assertEqual(len(grid.best_params_), 1)
        self.assertIsInstance(
            grid.best_params_['learning_rate'], LearningRateSGD)
        # print('\nREG', reg.best_params_, reg.best_score_,
        #       grid.best_params_['learning_rate'], grid.best_score_)
        # print(reg.cv_results_['mean_test_score'])
        # print(grid.cv_results_['mean_test_score'])

    @unittest.skipIf(TrainingSession is None, reason="not training")
    @ignore_warnings((
        ConvergenceWarning, DeprecationWarning, MyConvergenceWarning))
    def test_ort_gradient_optimizers_grid_cls(self):
        self.wtest_ort_gradient_optimizers_grid_cls(False)

    def wtest_ort_gradient_optimizers_grid_cls(self, use_weight=False):
        from onnxcustom.training.optimizers_partial import (
            OrtGradientForwardBackwardOptimizer)
        from onnxcustom.training.sgd_learning_rate import (
            LearningRateSGD)
        from onnxcustom.training.sgd_learning_loss import NegLogLearningLoss
        values = [1e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4,
                  5e-4, 1e-3, 5e-3, 1e-2, 1e-1, 1,
                  10, 100, 1000]
        X = numpy.random.randn(30, 3).astype(numpy.float32)
        y = (X.sum(axis=1) >= 0).astype(
            numpy.int64).reshape((-1, 1))
        X += numpy.random.randn(30, 3).astype(numpy.float32) / 10
        X_train, _, y_train, __ = train_test_split(X, y)
        scorer = make_scorer(
            lambda y_true, y_pred: (
                -log_loss(y_true, y_pred)))  # pylint: disable=E1130
        reg = GridSearchCV(
            SGDClassifier(max_iter=20),
            param_grid={'eta0': values},
            scoring=scorer, cv=3)
        reg.fit(X_train, y_train.ravel())
        self.assertIsInstance(reg.best_params_, dict)
        self.assertIn('eta0', reg.best_params_)
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearClassifier'},
                      options={'zipmap': False})
        onx = select_model_inputs_outputs(
            onx, outputs=['score'])
        onx = onnx_rename_weights(onx)
        inits = ['I0_coef', 'I1_intercept']

        cvalues = [LearningRateSGD(v) for v in values]
        grid = GridSearchCV(
            OrtGradientForwardBackwardOptimizer(
                onx, inits, weight_name='weight' if use_weight else None,
                learning_rate=LearningRateSGD(1e-4),
                learning_loss=NegLogLearningLoss(),
                warm_start=False, max_iter=20, batch_size=10,
                enable_logging=False, exc=False),
            param_grid={'learning_rate': cvalues}, cv=3)
        if use_weight:
            grid.fit(X_train, y_train)
        else:
            grid.fit(X_train, y_train)
        self.assertIsInstance(grid.best_params_, dict)
        self.assertEqual(len(grid.best_params_), 1)
        self.assertIsInstance(
            grid.best_params_['learning_rate'], LearningRateSGD)
        # print('\nCLS', reg.best_params_, reg.best_score_,
        #       grid.best_params_['learning_rate'], grid.best_score_)
        # print(reg.cv_results_['mean_test_score'])
        # print(grid.cv_results_['mean_test_score'])


if __name__ == "__main__":
    unittest.main(verbosity=2)
