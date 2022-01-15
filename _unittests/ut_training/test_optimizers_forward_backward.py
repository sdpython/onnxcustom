"""
@brief      test log(time=10s)
"""

import unittest
import io
import pickle
import logging
from pyquickhelper.pycode import (
    ExtTestCase, ignore_warnings, skipif_appveyor,
    get_temp_folder)
import numpy
import onnx
from onnx.helper import set_model_props
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
from mlprodict.onnx_conv import to_onnx
from onnxcustom import __max_supported_opset__ as opset
from onnxcustom.training.sgd_learning_rate import (
    LearningRateSGD, LearningRateSGDNesterov)
from onnxcustom.training.sgd_learning_loss import (
    BaseLearningLoss, SquareLearningLoss, AbsoluteLearningLoss,
    ElasticLearningLoss)
from onnxcustom.training.sgd_learning_penalty import (
    BaseLearningPenalty, NoLearningPenalty, ElasticLearningPenalty)
from onnxcustom.utils.onnx_helper import onnx_rename_weights
from onnxcustom.training import ConvergenceError
try:
    from onnxruntime import TrainingSession
except ImportError:
    # onnxruntime not training
    TrainingSession = None


class TestOptimizersForwardBackward(ExtTestCase):

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_ort_gradient_optimizers_use_numpy_zero(self):
        from onnxcustom.training.optimizers_partial import OrtGradientForwardBackwardOptimizer
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=10, bias=2, random_state=0)
        X = X.astype(numpy.float32)
        y[:] = 10
        y = y.astype(numpy.float32)
        X_train, _, y_train, __ = train_test_split(X, y)
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        reg.coef_ = reg.coef_.reshape((1, -1))
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'})
        # onx = onnx_rename_weights(onx)
        set_model_props(onx, {'info': 'unit test'})
        inits = ['coef', 'intercept']
        train_session = OrtGradientForwardBackwardOptimizer(onx, inits)
        self.assertIsInstance(train_session.learning_loss, SquareLearningLoss)
        self.assertRaise(lambda: train_session.get_state(), AttributeError)
        train_session.fit(X_train, y_train, use_numpy=True)
        state_tensors = train_session.get_state()
        self.assertEqual(len(state_tensors), 2)
        r = repr(train_session)
        self.assertIn("OrtGradientForwardBackwardOptimizer(model_onnx=", r)
        self.assertIn("learning_rate='invscaling'", r)
        losses = train_session.train_losses_
        self.assertGreater(len(losses), 1)
        self.assertFalse(any(map(numpy.isnan, losses)))

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_ort_gradient_optimizers_use_numpy_zero_w(self):
        from onnxcustom.training.optimizers_partial import OrtGradientForwardBackwardOptimizer
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=10, bias=2, random_state=0)
        X = X.astype(numpy.float32)
        y[:] = 10
        y = y.astype(numpy.float32)
        w = (numpy.random.rand(y.shape[0]) + 1).astype(X.dtype)
        X_train, _, y_train, __, w_train, ___ = train_test_split(
            X, y, w)
        reg = LinearRegression()
        reg.fit(X_train, y_train, w_train)
        reg.coef_ = reg.coef_.reshape((1, -1))
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'})
        # onx = onnx_rename_weights(onx)
        set_model_props(onx, {'info': 'unit test'})
        inits = ['coef', 'intercept']
        train_session = OrtGradientForwardBackwardOptimizer(
            onx, inits, weight_name='weight')
        self.assertRaise(lambda: train_session.get_state(), AttributeError)
        train_session.fit(X_train, y_train, w_train, use_numpy=True)
        state_tensors = train_session.get_state()
        self.assertEqual(len(state_tensors), 2)
        r = repr(train_session)
        self.assertIn("OrtGradientForwardBackwardOptimizer(model_onnx=", r)
        self.assertIn("learning_rate='invscaling'", r)
        losses = train_session.train_losses_
        self.assertGreater(len(losses), 1)
        self.assertFalse(any(map(numpy.isnan, losses)))

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_ort_gradient_optimizers_use_numpy(self):
        from onnxcustom.training.optimizers_partial import OrtGradientForwardBackwardOptimizer
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=2, bias=2, random_state=0)
        X[:10, :] = 0
        X = X.astype(numpy.float32)
        y = (X.sum(axis=1) + y / 1000).astype(numpy.float32)
        X_train, _, y_train, __ = train_test_split(X, y)
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        reg.coef_ = reg.coef_.reshape((1, -1))
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'})
        # onx = onnx_rename_weights(onx)
        set_model_props(onx, {'info': 'unit test'})
        inits = ['coef', 'intercept']
        train_session = OrtGradientForwardBackwardOptimizer(
            onx, inits, enable_logging=False)
        self.assertRaise(lambda: train_session.get_state(), AttributeError)
        train_session.fit(X_train, y_train, use_numpy=True)
        state_tensors = train_session.get_state()
        self.assertEqual(len(state_tensors), 2)
        r = repr(train_session)
        self.assertIn("OrtGradientForwardBackwardOptimizer(model_onnx=", r)
        self.assertIn("learning_rate='invscaling'", r)
        losses = train_session.train_losses_
        self.assertGreater(len(losses), 1)
        self.assertFalse(any(map(numpy.isnan, losses)))

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_ort_gradient_optimizers_use_numpy_w(self):
        from onnxcustom.training.optimizers_partial import OrtGradientForwardBackwardOptimizer
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=2, bias=2, random_state=0)
        X[:10, :] = 0
        X = X.astype(numpy.float32)
        y = (X.sum(axis=1) + y / 1000).astype(numpy.float32)
        w = (numpy.random.rand(y.shape[0]) + 1).astype(X.dtype)
        X_train, _, y_train, __, w_train, ___ = train_test_split(
            X, y, w)
        reg = LinearRegression()
        reg.fit(X_train, y_train, w_train)
        reg.coef_ = reg.coef_.reshape((1, -1))
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'})
        # onx = onnx_rename_weights(onx)
        set_model_props(onx, {'info': 'unit test'})
        inits = ['coef', 'intercept']
        train_session = OrtGradientForwardBackwardOptimizer(
            onx, inits, enable_logging=False, weight_name='weight')
        self.assertRaise(lambda: train_session.get_state(), AttributeError)
        train_session.fit(X_train, y_train, w_train, use_numpy=True)
        state_tensors = train_session.get_state()
        self.assertEqual(len(state_tensors), 2)
        r = repr(train_session)
        self.assertIn("OrtGradientForwardBackwardOptimizer(model_onnx=", r)
        self.assertIn("learning_rate='invscaling'", r)
        losses = train_session.train_losses_
        self.assertGreater(len(losses), 1)
        self.assertFalse(any(map(numpy.isnan, losses)))

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_ort_gradient_optimizers_use_numpy_exc(self):
        from onnxcustom.training.optimizers_partial import OrtGradientForwardBackwardOptimizer
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=2, bias=2, random_state=0)
        X[:10, :] = 0
        X = X.astype(numpy.float32)
        y = (X.sum(axis=1) + y / 1000).astype(numpy.float32)
        X_train, _, y_train, __ = train_test_split(X, y)
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        reg.coef_ = reg.coef_.reshape((1, -1))
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'})
        # onx = onnx_rename_weights(onx)
        set_model_props(onx, {'info': 'unit test'})
        inits = ['coef', 'intercept']
        train_session = OrtGradientForwardBackwardOptimizer(
            onx, inits, enable_logging=False, learning_rate=1e3)
        self.assertRaise(
            lambda: train_session.fit(X_train, y_train, use_numpy=True),
            ConvergenceError)

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_ort_gradient_optimizers_use_numpy_exc_w(self):
        from onnxcustom.training.optimizers_partial import OrtGradientForwardBackwardOptimizer
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=2, bias=2, random_state=0)
        X[:10, :] = 0
        X = X.astype(numpy.float32)
        y = (X.sum(axis=1) + y / 1000).astype(numpy.float32)
        w = (numpy.random.rand(y.shape[0]) + 1).astype(X.dtype)
        X_train, _, y_train, __, w_train, ___ = train_test_split(
            X, y, w)
        reg = LinearRegression()
        reg.fit(X_train, y_train, w_train)
        reg.coef_ = reg.coef_.reshape((1, -1))
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'})
        # onx = onnx_rename_weights(onx)
        set_model_props(onx, {'info': 'unit test'})
        inits = ['coef', 'intercept']
        train_session = OrtGradientForwardBackwardOptimizer(
            onx, inits, enable_logging=False, learning_rate=1e3,
            weight_name='weight')
        self.assertRaise(
            lambda: train_session.fit(
                X_train, y_train, w_train, use_numpy=True),
            ConvergenceError)

    @unittest.skipIf(TrainingSession is None, reason="not training")
    @skipif_appveyor("logging issue")
    def test_ort_gradient_optimizers_use_numpy_log(self):
        from onnxcustom.training.optimizers_partial import OrtGradientForwardBackwardOptimizer
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=2, bias=2, random_state=0)
        X[:10, :] = 0
        X = X.astype(numpy.float32)
        y = (X.sum(axis=1) + y / 1000).astype(numpy.float32)
        X_train, _, y_train, __ = train_test_split(X, y)
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        reg.coef_ = reg.coef_.reshape((1, -1))
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'})
        # onx = onnx_rename_weights(onx)
        set_model_props(onx, {'info': 'unit test'})
        inits = ['coef', 'intercept']
        train_session = OrtGradientForwardBackwardOptimizer(
            onx, inits, enable_logging=True)
        res, logs = self.assertLogging(
            lambda: train_session.fit(X_train, y_train, use_numpy=True),
            'onnxcustom', level=logging.DEBUG)
        self.assertTrue(res is train_session)
        self.assertIn("[OrtGradientForwardBackwardOptimizer._iteration]", logs)

    @unittest.skipIf(TrainingSession is None, reason="not training")
    @skipif_appveyor("logging issue")
    def test_ort_gradient_optimizers_use_numpy_log_w(self):
        from onnxcustom.training.optimizers_partial import OrtGradientForwardBackwardOptimizer
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=2, bias=2, random_state=0)
        X[:10, :] = 0
        X = X.astype(numpy.float32)
        y = (X.sum(axis=1) + y / 1000).astype(numpy.float32)
        w = (numpy.random.rand(y.shape[0]) + 1).astype(X.dtype)
        X_train, _, y_train, __, w_train, ___ = train_test_split(
            X, y, w)
        reg = LinearRegression()
        reg.fit(X_train, y_train, w_train)
        reg.coef_ = reg.coef_.reshape((1, -1))
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'})
        # onx = onnx_rename_weights(onx)
        set_model_props(onx, {'info': 'unit test'})
        inits = ['coef', 'intercept']
        train_session = OrtGradientForwardBackwardOptimizer(
            onx, inits, enable_logging=True, weight_name='weight')
        res, logs = self.assertLogging(
            lambda: train_session.fit(
                X_train, y_train, w_train, use_numpy=True),
            'onnxcustom', level=logging.DEBUG)
        self.assertTrue(res is train_session)
        self.assertIn("[OrtGradientForwardBackwardOptimizer._iteration]", logs)

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_ort_gradient_optimizers_use_numpy_log_appveyor(self):
        from onnxcustom.training.optimizers_partial import OrtGradientForwardBackwardOptimizer
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=2, bias=2, random_state=0)
        X[:10, :] = 0
        X = X.astype(numpy.float32)
        y = (X.sum(axis=1) + y / 1000).astype(numpy.float32)
        X_train, _, y_train, __ = train_test_split(X, y)
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        reg.coef_ = reg.coef_.reshape((1, -1))
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'})
        # onx = onnx_rename_weights(onx)
        set_model_props(onx, {'info': 'unit test'})
        inits = ['coef', 'intercept']
        train_session = OrtGradientForwardBackwardOptimizer(
            onx, inits, enable_logging=True)
        res = train_session.fit(X_train, y_train, use_numpy=True)
        self.assertTrue(res is train_session)

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_ort_gradient_optimizers_use_numpy_log_appveyor_w(self):
        from onnxcustom.training.optimizers_partial import OrtGradientForwardBackwardOptimizer
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=2, bias=2, random_state=0)
        X[:10, :] = 0
        X = X.astype(numpy.float32)
        y = (X.sum(axis=1) + y / 1000).astype(numpy.float32)
        w = (numpy.random.rand(y.shape[0]) + 1).astype(X.dtype)
        X_train, _, y_train, __, w_train, ___ = train_test_split(
            X, y, w)
        reg = LinearRegression()
        reg.fit(X_train, y_train, w_train)
        reg.coef_ = reg.coef_.reshape((1, -1))
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'})
        # onx = onnx_rename_weights(onx)
        set_model_props(onx, {'info': 'unit test'})
        inits = ['coef', 'intercept']
        train_session = OrtGradientForwardBackwardOptimizer(
            onx, inits, enable_logging=True, weight_name='weight')
        res = train_session.fit(X_train, y_train, w_train, use_numpy=True)
        self.assertTrue(res is train_session)

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_ort_gradient_optimizers_use_numpy_pickle(self):
        from onnxcustom.training.optimizers_partial import OrtGradientForwardBackwardOptimizer
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=10, bias=2, random_state=0)
        X = X.astype(numpy.float32)
        y = y.astype(numpy.float32)
        X_train, _, y_train, __ = train_test_split(X, y)
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        reg.coef_ = reg.coef_.reshape((1, -1))
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'})
        set_model_props(onx, {'info': 'unit test'})
        inits = ['coef', 'intercept']
        train_session0 = OrtGradientForwardBackwardOptimizer(
            onx, inits, learning_rate=1e-4)

        st = io.BytesIO()
        pickle.dump(train_session0, st)
        st2 = io.BytesIO(st.getvalue())
        train_session1 = pickle.load(st2)

        train_session1.fit(X_train, y_train, use_numpy=True)

        st = io.BytesIO()
        pickle.dump(train_session1, st)
        st2 = io.BytesIO(st.getvalue())
        train_session = pickle.load(st2)
        state_tensors = train_session.get_state()
        self.assertEqual(len(state_tensors), 2)

        train_session.fit(X_train, y_train, use_numpy=True)
        state_tensors = train_session.get_state()
        self.assertEqual(len(state_tensors), 2)
        r = repr(train_session)
        self.assertIn("OrtGradientForwardBackwardOptimizer(model_onnx=", r)
        self.assertIn("learning_rate='invscaling'", r)
        losses = train_session.train_losses_
        self.assertGreater(len(losses), 1)
        self.assertFalse(any(map(numpy.isnan, losses)))

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_ort_gradient_optimizers_use_numpy_pickle_w(self):
        from onnxcustom.training.optimizers_partial import OrtGradientForwardBackwardOptimizer
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=10, bias=2, random_state=0)
        X = X.astype(numpy.float32)
        y = y.astype(numpy.float32)
        w = (numpy.random.rand(y.shape[0]) + 1).astype(X.dtype)
        X_train, _, y_train, __, w_train, ___ = train_test_split(
            X, y, w)
        reg = LinearRegression()
        reg.fit(X_train, y_train, w_train)
        reg.coef_ = reg.coef_.reshape((1, -1))
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'})
        set_model_props(onx, {'info': 'unit test'})
        inits = ['coef', 'intercept']
        train_session0 = OrtGradientForwardBackwardOptimizer(
            onx, inits, learning_rate=1e-4, weight_name='weight')

        st = io.BytesIO()
        pickle.dump(train_session0, st)
        st2 = io.BytesIO(st.getvalue())
        train_session1 = pickle.load(st2)

        train_session1.fit(X_train, y_train, w_train, use_numpy=True)

        st = io.BytesIO()
        pickle.dump(train_session1, st)
        st2 = io.BytesIO(st.getvalue())
        train_session = pickle.load(st2)
        state_tensors = train_session.get_state()
        self.assertEqual(len(state_tensors), 2)

        train_session.fit(X_train, y_train, w_train, use_numpy=True)
        state_tensors = train_session.get_state()
        self.assertEqual(len(state_tensors), 2)
        r = repr(train_session)
        self.assertIn("OrtGradientForwardBackwardOptimizer(model_onnx=", r)
        self.assertIn("learning_rate='invscaling'", r)
        losses = train_session.train_losses_
        self.assertGreater(len(losses), 1)
        self.assertFalse(any(map(numpy.isnan, losses)))

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_ort_gradient_optimizers_use_ort(self):
        from onnxcustom.training.optimizers_partial import OrtGradientForwardBackwardOptimizer
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=10, bias=2)
        X = X.astype(numpy.float32)
        y = y.astype(numpy.float32)
        X_train, _, y_train, __ = train_test_split(X, y)
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        reg.coef_ = reg.coef_.reshape((1, -1))
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'})
        set_model_props(onx, {'info': 'unit test'})
        inits = ['coef', 'intercept']
        train_session = OrtGradientForwardBackwardOptimizer(onx, inits)
        self.assertRaise(lambda: train_session.get_state(), AttributeError)
        train_session.fit(X_train, y_train, use_numpy=False)
        state_tensors = train_session.get_state()
        self.assertEqual(len(state_tensors), 2)
        r = repr(train_session)
        self.assertIn("OrtGradientForwardBackwardOptimizer(model_onnx=", r)
        self.assertIn("learning_rate='invscaling'", r)
        losses = train_session.train_losses_
        self.assertGreater(len(losses), 1)
        self.assertFalse(any(map(numpy.isnan, losses)))

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_ort_gradient_optimizers_use_ort_w(self):
        from onnxcustom.training.optimizers_partial import OrtGradientForwardBackwardOptimizer
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=10, bias=2)
        X = X.astype(numpy.float32)
        y = y.astype(numpy.float32)
        w = (numpy.random.rand(y.shape[0]) + 1).astype(X.dtype)
        X_train, _, y_train, __, w_train, ___ = train_test_split(
            X, y, w)
        reg = LinearRegression()
        reg.fit(X_train, y_train, w_train)
        reg.coef_ = reg.coef_.reshape((1, -1))
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'})
        set_model_props(onx, {'info': 'unit test'})
        inits = ['coef', 'intercept']
        train_session = OrtGradientForwardBackwardOptimizer(
            onx, inits, weight_name='weight')
        self.assertRaise(lambda: train_session.get_state(), AttributeError)
        train_session.fit(X_train, y_train, w_train, use_numpy=False)
        state_tensors = train_session.get_state()
        self.assertEqual(len(state_tensors), 2)
        r = repr(train_session)
        self.assertIn("OrtGradientForwardBackwardOptimizer(model_onnx=", r)
        self.assertIn("learning_rate='invscaling'", r)
        losses = train_session.train_losses_
        self.assertGreater(len(losses), 1)
        self.assertFalse(any(map(numpy.isnan, losses)))

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_ort_gradient_optimizers_optimal_use_numpy(self):
        from onnxcustom.training.optimizers_partial import OrtGradientForwardBackwardOptimizer
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=10, bias=2, random_state=0)
        X = X.astype(numpy.float32)
        y = y.astype(numpy.float32)
        X_train, _, y_train, __ = train_test_split(X, y)
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        reg.coef_ = reg.coef_.reshape((1, -1))
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'})
        inits = ['coef', 'intercept']
        train_session = OrtGradientForwardBackwardOptimizer(
            onx, inits, max_iter=10,
            learning_rate=LearningRateSGD(learning_rate='optimal'))
        self.assertRaise(lambda: train_session.get_state(), AttributeError)
        train_session.fit(X_train, y_train, use_numpy=True)
        state_tensors = train_session.get_state()
        self.assertEqual(len(state_tensors), 2)
        losses = train_session.train_losses_
        self.assertGreater(len(losses), 1)
        self.assertFalse(any(map(numpy.isnan, losses)))
        r = repr(train_session)
        self.assertIn("OrtGradientForwardBackwardOptimizer(model_onnx=", r)
        self.assertIn("learning_rate='optimal'", r)

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_ort_gradient_optimizers_optimal_use_numpy_w(self):
        from onnxcustom.training.optimizers_partial import OrtGradientForwardBackwardOptimizer
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=10, bias=2, random_state=0)
        X = X.astype(numpy.float32)
        y = y.astype(numpy.float32)
        w = (numpy.random.rand(y.shape[0]) + 1).astype(X.dtype)
        X_train, _, y_train, __, w_train, ___ = train_test_split(
            X, y, w)
        reg = LinearRegression()
        reg.fit(X_train, y_train, w_train)
        reg.coef_ = reg.coef_.reshape((1, -1))
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'})
        inits = ['coef', 'intercept']
        train_session = OrtGradientForwardBackwardOptimizer(
            onx, inits, max_iter=10, weight_name='weight',
            learning_rate=LearningRateSGD(learning_rate='optimal'))
        self.assertRaise(lambda: train_session.get_state(), AttributeError)
        train_session.fit(X_train, y_train, w_train, use_numpy=True)
        state_tensors = train_session.get_state()
        self.assertEqual(len(state_tensors), 2)
        losses = train_session.train_losses_
        self.assertGreater(len(losses), 1)
        self.assertFalse(any(map(numpy.isnan, losses)))
        r = repr(train_session)
        self.assertIn("OrtGradientForwardBackwardOptimizer(model_onnx=", r)
        self.assertIn("learning_rate='optimal'", r)

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_ort_gradient_optimizers_optimal_use_ort(self):
        from onnxcustom.training.optimizers_partial import OrtGradientForwardBackwardOptimizer
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=10, bias=2, random_state=0)
        X = X.astype(numpy.float32)
        y = y.astype(numpy.float32)
        X_train, _, y_train, __ = train_test_split(X, y)
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        reg.coef_ = reg.coef_.reshape((1, -1))
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'})
        inits = ['coef', 'intercept']
        train_session = OrtGradientForwardBackwardOptimizer(
            onx, inits, max_iter=10,
            learning_rate=LearningRateSGD(learning_rate='optimal'))
        self.assertRaise(lambda: train_session.get_state(), AttributeError)
        train_session.fit(X_train, y_train, use_numpy=False)
        state_tensors = train_session.get_state()
        self.assertEqual(len(state_tensors), 2)
        r = repr(train_session)
        self.assertIn("OrtGradientForwardBackwardOptimizer(model_onnx=", r)
        self.assertIn("learning_rate='optimal'", r)
        losses = train_session.train_losses_
        self.assertGreater(len(losses), 1)
        self.assertFalse(any(map(numpy.isnan, losses)))

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_ort_gradient_optimizers_optimal_use_ort_w(self):
        from onnxcustom.training.optimizers_partial import OrtGradientForwardBackwardOptimizer
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=10, bias=2, random_state=0)
        X = X.astype(numpy.float32)
        y = y.astype(numpy.float32)
        w = (numpy.random.rand(y.shape[0]) + 1).astype(X.dtype)
        X_train, _, y_train, __, w_train, ___ = train_test_split(
            X, y, w)
        reg = LinearRegression()
        reg.fit(X_train, y_train, w_train)
        reg.coef_ = reg.coef_.reshape((1, -1))
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'})
        inits = ['coef', 'intercept']
        train_session = OrtGradientForwardBackwardOptimizer(
            onx, inits, max_iter=10, weight_name='weight',
            learning_rate=LearningRateSGD(learning_rate='optimal'))
        self.assertRaise(lambda: train_session.get_state(), AttributeError)
        train_session.fit(X_train, y_train, w_train, use_numpy=False)
        state_tensors = train_session.get_state()
        self.assertEqual(len(state_tensors), 2)
        r = repr(train_session)
        self.assertIn("OrtGradientForwardBackwardOptimizer(model_onnx=", r)
        self.assertIn("learning_rate='optimal'", r)
        losses = train_session.train_losses_
        self.assertGreater(len(losses), 1)
        self.assertFalse(any(map(numpy.isnan, losses)))

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_ort_gradient_optimizers_optimal_use_ort_w_absolute(self):
        from onnxcustom.training.optimizers_partial import OrtGradientForwardBackwardOptimizer
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=10, bias=2, random_state=0)
        X = X.astype(numpy.float32)
        y = y.astype(numpy.float32)
        w = (numpy.random.rand(y.shape[0]) + 1).astype(X.dtype)
        X_train, _, y_train, __, w_train, ___ = train_test_split(
            X, y, w)
        reg = LinearRegression()
        reg.fit(X_train, y_train, w_train)
        reg.coef_ = reg.coef_.reshape((1, -1))
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'})
        inits = ['coef', 'intercept']
        train_session = OrtGradientForwardBackwardOptimizer(
            onx, inits, max_iter=10, weight_name='weight',
            learning_rate=LearningRateSGD(learning_rate='optimal'),
            learning_loss='absolute_error')
        self.assertIsInstance(
            train_session.learning_loss, AbsoluteLearningLoss)
        self.assertRaise(lambda: train_session.get_state(), AttributeError)
        train_session.fit(X_train, y_train, w_train, use_numpy=False)
        state_tensors = train_session.get_state()
        self.assertEqual(len(state_tensors), 2)
        r = repr(train_session)
        self.assertIn("OrtGradientForwardBackwardOptimizer(model_onnx=", r)
        self.assertIn("learning_rate='optimal'", r)
        losses = train_session.train_losses_
        self.assertGreater(len(losses), 1)
        self.assertFalse(any(map(numpy.isnan, losses)))

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_ort_gradient_optimizers_optimal_use_ort_w_elastic(self):
        from onnxcustom.training.optimizers_partial import OrtGradientForwardBackwardOptimizer
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=10, bias=2, random_state=0)
        X = X.astype(numpy.float32)
        y = y.astype(numpy.float32)
        w = (numpy.random.rand(y.shape[0]) + 1).astype(X.dtype)
        X_train, _, y_train, __, w_train, ___ = train_test_split(
            X, y, w)
        reg = LinearRegression()
        reg.fit(X_train, y_train, w_train)
        reg.coef_ = reg.coef_.reshape((1, -1))
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'})
        inits = ['coef', 'intercept']
        train_session = OrtGradientForwardBackwardOptimizer(
            onx, inits, max_iter=10, weight_name='weight',
            learning_rate=LearningRateSGD(learning_rate='optimal'),
            learning_loss=BaseLearningLoss.select(
                'elastic', l1_weight=0.1, l2_weight=0.9))
        self.assertIsInstance(
            train_session.learning_loss, ElasticLearningLoss)
        self.assertIsInstance(
            train_session.learning_penalty, NoLearningPenalty)
        self.assertRaise(lambda: train_session.get_state(), AttributeError)
        train_session.fit(X_train, y_train, w_train, use_numpy=False)
        state_tensors = train_session.get_state()
        self.assertEqual(len(state_tensors), 2)
        r = repr(train_session)
        self.assertIn("OrtGradientForwardBackwardOptimizer(model_onnx=", r)
        self.assertIn("learning_rate='optimal'", r)
        losses = train_session.train_losses_
        self.assertGreater(len(losses), 1)
        self.assertFalse(any(map(numpy.isnan, losses)))

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_ort_gradient_optimizers_optimal_use_ort_w_elastic_penalty(self):
        from onnxcustom.training.optimizers_partial import OrtGradientForwardBackwardOptimizer
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=10, bias=2, random_state=0)
        X = X.astype(numpy.float32)
        y = y.astype(numpy.float32)
        w = (numpy.random.rand(y.shape[0]) + 1).astype(X.dtype)
        X_train, _, y_train, __, w_train, ___ = train_test_split(
            X, y, w)
        reg = LinearRegression()
        reg.fit(X_train, y_train, w_train)
        reg.coef_ = reg.coef_.reshape((1, -1))
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'})
        inits = ['coef', 'intercept']
        train_session = OrtGradientForwardBackwardOptimizer(
            onx, inits, max_iter=10, weight_name='weight',
            learning_rate=LearningRateSGD(learning_rate='optimal'),
            learning_loss=BaseLearningLoss.select(
                'elastic', l1_weight=0.1, l2_weight=0.9),
            learning_penalty=BaseLearningPenalty.select(
                'elastic', l1=0.1, l2=0.9))
        self.assertIsInstance(
            train_session.learning_loss, ElasticLearningLoss)
        self.assertIsInstance(
            train_session.learning_penalty, ElasticLearningPenalty)
        self.assertRaise(lambda: train_session.get_state(), AttributeError)
        train_session.fit(X_train, y_train, w_train, use_numpy=False)
        state_tensors = train_session.get_state()
        self.assertEqual(len(state_tensors), 2)
        r = repr(train_session)
        self.assertIn("OrtGradientForwardBackwardOptimizer(model_onnx=", r)
        self.assertIn("learning_rate='optimal'", r)
        losses = train_session.train_losses_
        self.assertGreater(len(losses), 1)
        self.assertFalse(any(map(numpy.isnan, losses)))

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_ort_gradient_optimizers_evaluation_use_numpy(self):
        from onnxcustom.training.optimizers_partial import OrtGradientForwardBackwardOptimizer
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=10, bias=2)
        X = X.astype(numpy.float32)
        y = y.astype(numpy.float32)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        reg.coef_ = reg.coef_.reshape((1, -1))
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'})
        inits = ['coef', 'intercept']
        train_session = OrtGradientForwardBackwardOptimizer(onx, inits)
        self.assertRaise(lambda: train_session.get_state(), AttributeError)
        train_session.fit(X_train, y_train, X_val=X_test,
                          y_val=y_test, use_numpy=True)
        state_tensors = train_session.get_state()
        self.assertEqual(len(state_tensors), 2)
        r = repr(train_session)
        self.assertIn("OrtGradientForwardBackwardOptimizer(model_onnx=", r)
        self.assertIn("learning_rate='invscaling'", r)
        losses = train_session.train_losses_
        self.assertGreater(len(losses), 1)
        self.assertFalse(any(map(numpy.isnan, losses)))
        vlosses = train_session.validation_losses_
        self.assertGreater(len(vlosses), 1)
        self.assertFalse(any(map(numpy.isnan, losses)))

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_ort_gradient_optimizers_evaluation_use_numpy_w(self):
        from onnxcustom.training.optimizers_partial import OrtGradientForwardBackwardOptimizer
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=10, bias=2)
        X = X.astype(numpy.float32)
        y = y.astype(numpy.float32)
        w = (numpy.random.rand(y.shape[0]) + 1).astype(X.dtype)
        X_train, X_test, y_train, y_test, w_train, ___ = train_test_split(
            X, y, w)
        reg = LinearRegression()
        reg.fit(X_train, y_train, w_train)
        reg.coef_ = reg.coef_.reshape((1, -1))
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'})
        inits = ['coef', 'intercept']
        train_session = OrtGradientForwardBackwardOptimizer(
            onx, inits, weight_name='weight')
        self.assertRaise(lambda: train_session.get_state(), AttributeError)
        train_session.fit(
            X_train, y_train, w_train, X_val=X_test, y_val=y_test,
            use_numpy=True)
        state_tensors = train_session.get_state()
        self.assertEqual(len(state_tensors), 2)
        r = repr(train_session)
        self.assertIn("OrtGradientForwardBackwardOptimizer(model_onnx=", r)
        self.assertIn("learning_rate='invscaling'", r)
        losses = train_session.train_losses_
        self.assertGreater(len(losses), 1)
        self.assertFalse(any(map(numpy.isnan, losses)))
        vlosses = train_session.validation_losses_
        self.assertGreater(len(vlosses), 1)
        self.assertFalse(any(map(numpy.isnan, losses)))

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_ort_gradient_optimizers_evaluation_use_ort(self):
        from onnxcustom.training.optimizers_partial import OrtGradientForwardBackwardOptimizer
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=10, bias=2)
        X = X.astype(numpy.float32)
        y = y.astype(numpy.float32)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        reg.coef_ = reg.coef_.reshape((1, -1))
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'})
        inits = ['coef', 'intercept']
        train_session = OrtGradientForwardBackwardOptimizer(onx, inits)
        self.assertRaise(lambda: train_session.get_state(), AttributeError)
        train_session.fit(X_train, y_train, X_val=X_test,
                          y_val=y_test, use_numpy=False)
        state_tensors = train_session.get_state()
        self.assertEqual(len(state_tensors), 2)
        r = repr(train_session)
        self.assertIn("OrtGradientForwardBackwardOptimizer(model_onnx=", r)
        self.assertIn("learning_rate='invscaling'", r)
        losses = train_session.train_losses_
        self.assertGreater(len(losses), 1)
        self.assertFalse(any(map(numpy.isnan, losses)))
        vlosses = train_session.validation_losses_
        self.assertGreater(len(vlosses), 1)
        self.assertFalse(any(map(numpy.isnan, losses)))

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_ort_gradient_optimizers_evaluation_use_ort_w(self):
        from onnxcustom.training.optimizers_partial import OrtGradientForwardBackwardOptimizer
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=10, bias=2)
        X = X.astype(numpy.float32)
        y = y.astype(numpy.float32)
        w = (numpy.random.rand(y.shape[0]) + 1).astype(X.dtype)
        X_train, X_test, y_train, y_test, w_train, ___ = train_test_split(
            X, y, w)
        reg = LinearRegression()
        reg.fit(X_train, y_train, w_train)
        reg.coef_ = reg.coef_.reshape((1, -1))
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'})
        inits = ['coef', 'intercept']
        train_session = OrtGradientForwardBackwardOptimizer(
            onx, inits, weight_name='weight')
        self.assertRaise(lambda: train_session.get_state(), AttributeError)
        train_session.fit(
            X_train, y_train, w_train,
            X_val=X_test, y_val=y_test, use_numpy=False)
        state_tensors = train_session.get_state()
        self.assertEqual(len(state_tensors), 2)
        r = repr(train_session)
        self.assertIn("OrtGradientForwardBackwardOptimizer(model_onnx=", r)
        self.assertIn("learning_rate='invscaling'", r)
        losses = train_session.train_losses_
        self.assertGreater(len(losses), 1)
        self.assertFalse(any(map(numpy.isnan, losses)))
        vlosses = train_session.validation_losses_
        self.assertGreater(len(vlosses), 1)
        self.assertFalse(any(map(numpy.isnan, losses)))

    @unittest.skipIf(TrainingSession is None, reason="not training")
    @ignore_warnings(ConvergenceWarning)
    def test_ort_gradient_optimizers_use_numpy_nn(self):
        from onnxcustom.training.optimizers_partial import OrtGradientForwardBackwardOptimizer
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=2, bias=2, random_state=0)
        X[:10, :] = 0
        X = X.astype(numpy.float32)
        y = (X.sum(axis=1) + y / 1000).astype(numpy.float32)
        X_train, _, y_train, __ = train_test_split(X, y)
        reg = MLPRegressor(
            hidden_layer_sizes=(3, 5), max_iter=5,
            solver='sgd', learning_rate_init=1e-4,
            n_iter_no_change=1000, batch_size=7)
        reg.fit(X_train, y_train)
        onx = to_onnx(reg, X_train, target_opset=opset)

        inits = list(sorted(['coefficient', 'intercepts', 'coefficient1',
                             'intercepts1', 'coefficient2', 'intercepts2']))
        train_session = OrtGradientForwardBackwardOptimizer(
            onx, inits, enable_logging=False)
        self.assertRaise(
            lambda: train_session.fit(X_train, y_train, use_numpy=True),
            ValueError)
        train_session = OrtGradientForwardBackwardOptimizer(
            onx, ["A"], enable_logging=False)
        self.assertRaise(
            lambda: train_session.fit(X_train, y_train, use_numpy=True),
            ValueError)

        onx = onnx_rename_weights(onx)
        set_model_props(onx, {'info': 'unit test'})
        train_session = OrtGradientForwardBackwardOptimizer(
            onx, enable_logging=False)
        self.assertRaise(lambda: train_session.get_state(), AttributeError)
        train_session.fit(X_train, y_train, use_numpy=True)
        state_tensors = train_session.get_state()
        self.assertEqual(len(state_tensors), 6)
        r = repr(train_session)
        self.assertIn("OrtGradientForwardBackwardOptimizer(model_onnx=", r)
        self.assertIn("learning_rate='invscaling'", r)
        losses = train_session.train_losses_
        self.assertGreater(len(losses), 1)
        self.assertFalse(any(map(numpy.isnan, losses)))

    @unittest.skipIf(TrainingSession is None, reason="not training")
    @ignore_warnings(ConvergenceWarning)
    def test_ort_gradient_optimizers_use_numpy_nn_w(self):
        from onnxcustom.training.optimizers_partial import OrtGradientForwardBackwardOptimizer
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=2, bias=2, random_state=0)
        X[:10, :] = 0
        X = X.astype(numpy.float32)
        y = (X.sum(axis=1) + y / 1000).astype(numpy.float32)
        w = (numpy.random.rand(y.shape[0]) + 1).astype(X.dtype)
        X_train, _, y_train, __, w_train, ___ = train_test_split(
            X, y, w)
        reg = MLPRegressor(
            hidden_layer_sizes=(3, 5), max_iter=5,
            solver='sgd', learning_rate_init=1e-4,
            n_iter_no_change=1000, batch_size=7)
        reg.fit(X_train, y_train)
        onx = to_onnx(reg, X_train, target_opset=opset)

        inits = list(sorted(['coefficient', 'intercepts', 'coefficient1',
                             'intercepts1', 'coefficient2', 'intercepts2']))
        train_session = OrtGradientForwardBackwardOptimizer(
            onx, inits, enable_logging=False, weight_name='weight')
        self.assertRaise(
            lambda: train_session.fit(
                X_train, y_train, w_train, use_numpy=True),
            ValueError)
        train_session = OrtGradientForwardBackwardOptimizer(
            onx, ["A"], enable_logging=False)
        self.assertRaise(
            lambda: train_session.fit(
                X_train, y_train, w_train, use_numpy=True),
            ValueError)

        onx = onnx_rename_weights(onx)
        set_model_props(onx, {'info': 'unit test'})
        train_session = OrtGradientForwardBackwardOptimizer(
            onx, enable_logging=False, weight_name='weight')
        self.assertRaise(lambda: train_session.get_state(), AttributeError)
        train_session.fit(X_train, y_train, w_train, use_numpy=True)
        state_tensors = train_session.get_state()
        self.assertEqual(len(state_tensors), 6)
        r = repr(train_session)
        self.assertIn("OrtGradientForwardBackwardOptimizer(model_onnx=", r)
        self.assertIn("learning_rate='invscaling'", r)
        losses = train_session.train_losses_
        self.assertGreater(len(losses), 1)
        self.assertFalse(any(map(numpy.isnan, losses)))

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_ort_gradient_optimizers_use_numpy_nesterov(self):
        from onnxcustom.training.optimizers_partial import OrtGradientForwardBackwardOptimizer
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=2, bias=2, random_state=0)
        X[:10, :] = 0
        X = X.astype(numpy.float32)
        y = (X.sum(axis=1) + y / 1000).astype(numpy.float32)
        X_train, _, y_train, __ = train_test_split(X, y)
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        reg.coef_ = reg.coef_.reshape((1, -1))
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'})
        # onx = onnx_rename_weights(onx)
        set_model_props(onx, {'info': 'unit test'})
        inits = ['coef', 'intercept']
        train_session = OrtGradientForwardBackwardOptimizer(
            onx, inits, enable_logging=False, learning_rate='Nesterov')
        self.assertRaise(lambda: train_session.get_state(), AttributeError)
        train_session.fit(X_train, y_train, use_numpy=True)
        self.assertIsInstance(train_session.learning_rate,
                              LearningRateSGDNesterov)
        state_tensors = train_session.get_state()
        self.assertEqual(len(state_tensors), 2)
        r = repr(train_session)
        self.assertIn("OrtGradientForwardBackwardOptimizer(model_onnx=", r)
        self.assertIn("learning_rate='invscaling'", r)
        losses = train_session.train_losses_
        self.assertGreater(len(losses), 1)
        self.assertFalse(any(map(numpy.isnan, losses)))

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_ort_gradient_optimizers_use_numpy_w_nesterov1(self):
        from onnxcustom.training.optimizers_partial import OrtGradientForwardBackwardOptimizer
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=2, bias=2, random_state=0)
        X[:10, :] = 0
        X = X.astype(numpy.float32)
        y = (X.sum(axis=1) + y / 1000).astype(numpy.float32)
        w = (numpy.random.rand(y.shape[0]) + 1).astype(X.dtype)
        X_train, _, y_train, __, w_train, ___ = train_test_split(
            X, y, w)
        reg = LinearRegression()
        reg.fit(X_train, y_train, w_train)
        reg.coef_ = reg.coef_.reshape((1, -1))
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'})
        # onx = onnx_rename_weights(onx)
        set_model_props(onx, {'info': 'unit test'})
        inits = ['coef', 'intercept']
        train_session = OrtGradientForwardBackwardOptimizer(
            onx, inits, enable_logging=False, weight_name='weight',
            learning_rate=LearningRateSGDNesterov(nesterov=False))
        self.assertRaise(lambda: train_session.get_state(), AttributeError)
        train_session.fit(X_train, y_train, w_train, use_numpy=True)
        self.assertIsInstance(train_session.learning_rate,
                              LearningRateSGDNesterov)
        state_tensors = train_session.get_state()
        self.assertEqual(len(state_tensors), 2)
        r = repr(train_session)
        self.assertIn("OrtGradientForwardBackwardOptimizer(model_onnx=", r)
        self.assertIn("learning_rate='invscaling'", r)
        losses = train_session.train_losses_
        self.assertGreater(len(losses), 1)
        self.assertFalse(any(map(numpy.isnan, losses)))

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_ort_gradient_optimizers_use_numpy_w_nesterov2(self):
        from onnxcustom.training.optimizers_partial import OrtGradientForwardBackwardOptimizer
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=2, bias=2, random_state=0)
        X[:10, :] = 0
        X = X.astype(numpy.float32)
        y = (X.sum(axis=1) + y / 1000).astype(numpy.float32)
        w = (numpy.random.rand(y.shape[0]) + 1).astype(X.dtype)
        X_train, _, y_train, __, w_train, ___ = train_test_split(
            X, y, w)
        reg = LinearRegression()
        reg.fit(X_train, y_train, w_train)
        reg.coef_ = reg.coef_.reshape((1, -1))
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'})
        # onx = onnx_rename_weights(onx)
        set_model_props(onx, {'info': 'unit test'})
        inits = ['coef', 'intercept']
        train_session = OrtGradientForwardBackwardOptimizer(
            onx, inits, enable_logging=False, weight_name='weight',
            learning_rate=LearningRateSGDNesterov(nesterov=True))
        self.assertRaise(lambda: train_session.get_state(), AttributeError)
        train_session.fit(X_train, y_train, w_train, use_numpy=True)
        self.assertIsInstance(train_session.learning_rate,
                              LearningRateSGDNesterov)
        state_tensors = train_session.get_state()
        self.assertEqual(len(state_tensors), 2)
        r = repr(train_session)
        self.assertIn("OrtGradientForwardBackwardOptimizer(model_onnx=", r)
        self.assertIn("learning_rate='invscaling'", r)
        losses = train_session.train_losses_
        self.assertGreater(len(losses), 1)
        self.assertFalse(any(map(numpy.isnan, losses)))

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_ort_gradient_optimizers_use_numpy_pickle_w_nesterov(self):
        from onnxcustom.training.optimizers_partial import OrtGradientForwardBackwardOptimizer
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=10, bias=2, random_state=0)
        X = X.astype(numpy.float32)
        y = y.astype(numpy.float32)
        w = (numpy.random.rand(y.shape[0]) + 1).astype(X.dtype)
        X_train, _, y_train, __, w_train, ___ = train_test_split(
            X, y, w)
        reg = LinearRegression()
        reg.fit(X_train, y_train, w_train)
        reg.coef_ = reg.coef_.reshape((1, -1))
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'})
        set_model_props(onx, {'info': 'unit test'})
        inits = ['coef', 'intercept']
        train_session0 = OrtGradientForwardBackwardOptimizer(
            onx, inits, learning_rate="Nesterov", weight_name='weight')
        self.assertIsInstance(train_session0.learning_rate,
                              LearningRateSGDNesterov)
        st = io.BytesIO()
        pickle.dump(train_session0, st)
        st2 = io.BytesIO(st.getvalue())
        train_session1 = pickle.load(st2)

        train_session1.fit(X_train, y_train, w_train, use_numpy=True)
        self.assertIsInstance(train_session1.learning_rate,
                              LearningRateSGDNesterov)

        st = io.BytesIO()
        pickle.dump(train_session1, st)
        st2 = io.BytesIO(st.getvalue())
        train_session = pickle.load(st2)
        state_tensors = train_session.get_state()
        self.assertEqual(len(state_tensors), 2)

        train_session.fit(X_train, y_train, w_train, use_numpy=True)
        state_tensors = train_session.get_state()
        self.assertEqual(len(state_tensors), 2)
        r = repr(train_session)
        self.assertIn("OrtGradientForwardBackwardOptimizer(model_onnx=", r)
        self.assertIn("learning_rate='invscaling'", r)
        losses = train_session.train_losses_
        self.assertGreater(len(losses), 1)
        self.assertFalse(any(map(numpy.isnan, losses)))

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_ort_gradient_optimizers_use_numpy_pickle_w_nesterov_rate(self):
        from onnxcustom.training.optimizers_partial import OrtGradientForwardBackwardOptimizer
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=10, bias=2, random_state=0)
        X = X.astype(numpy.float32)
        y = y.astype(numpy.float32)
        w = (numpy.random.rand(y.shape[0]) + 1).astype(X.dtype)
        X_train, _, y_train, __, w_train, ___ = train_test_split(
            X, y, w)
        reg = LinearRegression()
        reg.fit(X_train, y_train, w_train)
        reg.coef_ = reg.coef_.reshape((1, -1))
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'})
        set_model_props(onx, {'info': 'unit test'})
        inits = ['coef', 'intercept']
        train_session0 = OrtGradientForwardBackwardOptimizer(
            onx, inits, learning_rate="Nesterov", weight_name='weight',
            learning_loss=BaseLearningLoss.select(
                'elastic', l1_weight=0.1, l2_weight=0.9),
            learning_penalty=BaseLearningPenalty.select(
                'elastic', l1=0.1, l2=0.9))
        self.assertIsInstance(train_session0.learning_rate,
                              LearningRateSGDNesterov)
        self.assertIsInstance(train_session0.learning_loss,
                              ElasticLearningLoss)
        self.assertIsInstance(train_session0.learning_penalty,
                              ElasticLearningPenalty)
        st = io.BytesIO()
        pickle.dump(train_session0, st)
        st2 = io.BytesIO(st.getvalue())
        train_session1 = pickle.load(st2)

        train_session1.fit(X_train, y_train, w_train, use_numpy=True)
        self.assertIsInstance(train_session1.learning_rate,
                              LearningRateSGDNesterov)
        self.assertIsInstance(train_session1.learning_loss,
                              ElasticLearningLoss)
        self.assertIsInstance(train_session1.learning_penalty,
                              ElasticLearningPenalty)

        st = io.BytesIO()
        pickle.dump(train_session1, st)
        st2 = io.BytesIO(st.getvalue())
        train_session = pickle.load(st2)
        state_tensors = train_session.get_state()
        self.assertEqual(len(state_tensors), 2)

        train_session.fit(X_train, y_train, w_train, use_numpy=True)
        state_tensors = train_session.get_state()
        self.assertEqual(len(state_tensors), 2)
        r = repr(train_session)
        self.assertIn("OrtGradientForwardBackwardOptimizer(model_onnx=", r)
        self.assertIn("learning_rate='invscaling'", r)
        losses = train_session.train_losses_
        self.assertGreater(len(losses), 1)
        self.assertFalse(any(map(numpy.isnan, losses)))

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_ort_gradient_optimizers_nesterov_penalty_l2(self):
        from onnxcustom.training.optimizers_partial import OrtGradientForwardBackwardOptimizer
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=10, bias=2, random_state=0)
        X = X.astype(numpy.float32)
        y = y.astype(numpy.float32)
        w = (numpy.random.rand(y.shape[0]) + 1).astype(X.dtype)
        X_train, _, y_train, __, w_train, ___ = train_test_split(
            X, y, w)
        reg = LinearRegression()
        reg.fit(X_train, y_train, w_train)
        reg.coef_ = reg.coef_.reshape((1, -1))
        onx_model = to_onnx(reg, X_train, target_opset=opset,
                            black_op={'LinearRegressor'})
        set_model_props(onx_model, {'info': 'unit test'})
        inits = ['coef', 'intercept']

        train_session = OrtGradientForwardBackwardOptimizer(
            onx_model, inits,
            learning_rate=LearningRateSGDNesterov(
                1e-4, nesterov=True, momentum=0.85),
            learning_penalty=ElasticLearningPenalty(l1=0, l2=1e-4),
            warm_start=False, max_iter=100, batch_size=10)

        temp = get_temp_folder(
            __file__, "temp_ort_gradient_optimizers_nesterov_penalty_l2")

        saved = train_session.save_onnx_graph(temp)
        saved_bytes = train_session.save_onnx_graph(bytes)
        self.assertIsInstance(saved, dict)
        self.assertNotEmpty(saved)
        self.assertEqual(len(saved), len(saved_bytes))
        checked = []
        for k, v in saved_bytes.items():
            if k == "learning_penalty":
                for att, onxb in v.items():
                    if att in ('penalty_grad_onnx_', 'penalty_onnx_'):
                        onx = onnx.load(io.BytesIO(onxb))
                        for init in onx.graph.initializer:  # pylint: disable=E1101
                            vals = init.float_data
                            if len(vals) == 1 and vals[0] == 0:
                                checked.append((k, att))
        if len(checked) != 2:
            raise AssertionError("Unexpected parameter %r." % checked)
        train_session.fit(X, y)

        train_session = OrtGradientForwardBackwardOptimizer(
            onx_model, inits, weight_name='weight',
            learning_rate=LearningRateSGDNesterov(
                1e-4, nesterov=True, momentum=0.9),
            learning_penalty=ElasticLearningPenalty(l1=0, l2=1e-4),
            warm_start=False, max_iter=100, batch_size=10)
        temp = get_temp_folder(
            __file__, "temp_ort_gradient_optimizers_nesterov_penalty_l2_weight")
        train_session.save_onnx_graph(temp)
        train_session.fit(X, y, w)

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_ort_gradient_optimizers_nesterov_penalty_l1l2(self):
        from onnxcustom.training.optimizers_partial import OrtGradientForwardBackwardOptimizer
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=10, bias=2, random_state=0)
        X = X.astype(numpy.float32)
        y = y.astype(numpy.float32)
        w = (numpy.random.rand(y.shape[0]) + 1).astype(X.dtype)
        X_train, _, y_train, __, w_train, ___ = train_test_split(
            X, y, w)
        reg = LinearRegression()
        reg.fit(X_train, y_train, w_train)
        reg.coef_ = reg.coef_.reshape((1, -1))
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'})
        set_model_props(onx, {'info': 'unit test'})
        inits = ['coef', 'intercept']

        train_session = OrtGradientForwardBackwardOptimizer(
            onx, inits,
            learning_rate=LearningRateSGDNesterov(
                1e-4, nesterov=True, momentum=0.9),
            learning_penalty=ElasticLearningPenalty(l1=1e-3, l2=1e-4),
            warm_start=False, max_iter=100, batch_size=10)
        train_session.fit(X, y)

        train_session = OrtGradientForwardBackwardOptimizer(
            onx, inits, weight_name='weight',
            learning_rate=LearningRateSGDNesterov(
                1e-4, nesterov=True, momentum=0.9),
            learning_penalty=ElasticLearningPenalty(l1=1e-3, l2=1e-4),
            warm_start=False, max_iter=100, batch_size=10)
        train_session.fit(X, y, w)

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_ort_gradient_optimizers_nesterov_penalty_l1l2_no(self):
        from onnxcustom.training.optimizers_partial import OrtGradientForwardBackwardOptimizer
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=10, bias=2, random_state=0)
        X = X.astype(numpy.float32)
        y = y.astype(numpy.float32)
        w = (numpy.random.rand(y.shape[0]) + 1).astype(X.dtype)
        X_train, _, y_train, __, w_train, ___ = train_test_split(
            X, y, w)
        reg = LinearRegression()
        reg.fit(X_train, y_train, w_train)
        reg.coef_ = reg.coef_.reshape((1, -1))
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'})
        set_model_props(onx, {'info': 'unit test'})
        inits = ['coef', 'intercept']

        train_session = OrtGradientForwardBackwardOptimizer(
            onx, inits,
            learning_rate=LearningRateSGDNesterov(
                1e-4, nesterov=False, momentum=0.9),
            learning_penalty=ElasticLearningPenalty(l1=1e-3, l2=1e-4),
            warm_start=False, max_iter=100, batch_size=10)
        train_session.fit(X, y)

        train_session = OrtGradientForwardBackwardOptimizer(
            onx, inits, weight_name='weight',
            learning_rate=LearningRateSGDNesterov(
                1e-4, nesterov=False, momentum=0.9),
            learning_penalty=ElasticLearningPenalty(l1=1e-3, l2=1e-4),
            warm_start=False, max_iter=100, batch_size=10)
        train_session.fit(X, y, w)


if __name__ == "__main__":
    # import logging
    # logger = logging.getLogger('onnxcustom')
    # logger.setLevel(logging.DEBUG)
    # logging.basicConfig(level=logging.DEBUG)
    # cl = TestOptimizersForwardBackward()
    # cl.test_ort_gradient_optimizers_nesterov_penalty_l2()
    # stop
    unittest.main()
