"""
@brief      test log(time=8s)
"""

import unittest
import io
import pickle
from pyquickhelper.pycode import ExtTestCase
import numpy
from onnx.helper import set_model_props
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from mlprodict.onnx_conv import to_onnx
from onnxcustom import __max_supported_opset__ as opset
from onnxcustom.training.sgd_learning_rate import LearningRateSGDRegressor
try:
    from onnxruntime import TrainingSession
except ImportError:
    # onnxruntime not training
    TrainingSession = None


class TestOptimizers(ExtTestCase):

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_ort_gradient_optimizers_use_numpy(self):
        from onnxcustom.training.orttraining import add_loss_output
        from onnxcustom.training.optimizers import OrtGradientOptimizer
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
        onx_loss = add_loss_output(onx)
        inits = ['intercept', 'coef']
        train_session = OrtGradientOptimizer(
            onx_loss, inits, learning_rate=1e-3)
        self.assertRaise(lambda: train_session.get_state(), AttributeError)
        train_session.fit(X, y, use_numpy=True)
        state_tensors = train_session.get_state()
        self.assertEqual(len(state_tensors), 2)
        r = repr(train_session)
        self.assertIn("OrtGradientOptimizer(model_onnx=", r)
        self.assertIn("learning_rate='invscaling'", r)
        losses = train_session.train_losses_
        self.assertGreater(len(losses), 1)
        self.assertFalse(any(map(numpy.isnan, losses)))

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_ort_gradient_optimizers_use_numpy_pickle(self):
        from onnxcustom.training.orttraining import add_loss_output
        from onnxcustom.training.optimizers import OrtGradientOptimizer
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
        onx_loss = add_loss_output(onx)
        inits = ['intercept', 'coef']
        train_session0 = OrtGradientOptimizer(onx_loss, inits)

        st = io.BytesIO()
        pickle.dump(train_session0, st)
        st2 = io.BytesIO(st.getvalue())
        train_session1 = pickle.load(st2)

        train_session1.fit(X, y, use_numpy=True)

        st = io.BytesIO()
        pickle.dump(train_session1, st)
        st2 = io.BytesIO(st.getvalue())
        train_session = pickle.load(st2)
        state_tensors = train_session.get_state()
        self.assertEqual(len(state_tensors), 2)

        train_session.fit(X, y, use_numpy=True)
        state_tensors = train_session.get_state()
        self.assertEqual(len(state_tensors), 2)
        r = repr(train_session)
        self.assertIn("OrtGradientOptimizer(model_onnx=", r)
        self.assertIn("learning_rate='invscaling'", r)
        losses = train_session.train_losses_
        self.assertGreater(len(losses), 1)
        self.assertFalse(any(map(numpy.isnan, losses)))

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_ort_gradient_optimizers_use_ort(self):
        from onnxcustom.training.orttraining import add_loss_output
        from onnxcustom.training.optimizers import OrtGradientOptimizer
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
        onx_loss = add_loss_output(onx)
        inits = ['intercept', 'coef']
        train_session = OrtGradientOptimizer(onx_loss, inits)
        self.assertRaise(lambda: train_session.get_state(), AttributeError)
        train_session.fit(X, y, use_numpy=False)
        state_tensors = train_session.get_state()
        self.assertEqual(len(state_tensors), 2)
        r = repr(train_session)
        self.assertIn("OrtGradientOptimizer(model_onnx=", r)
        self.assertIn("learning_rate='invscaling'", r)
        losses = train_session.train_losses_
        self.assertGreater(len(losses), 1)
        self.assertFalse(any(map(numpy.isnan, losses)))

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_ort_gradient_optimizers_optimal_use_numpy(self):
        from onnxcustom.training.orttraining import add_loss_output
        from onnxcustom.training.optimizers import OrtGradientOptimizer
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
        onx_loss = add_loss_output(onx)
        inits = ['intercept', 'coef']
        train_session = OrtGradientOptimizer(
            onx_loss, inits, max_iter=10,
            learning_rate=LearningRateSGDRegressor(learning_rate='optimal'))
        self.assertRaise(lambda: train_session.get_state(), AttributeError)
        train_session.fit(X, y, use_numpy=True)
        state_tensors = train_session.get_state()
        self.assertEqual(len(state_tensors), 2)
        r = repr(train_session)
        self.assertIn("OrtGradientOptimizer(model_onnx=", r)
        self.assertIn("learning_rate='optimal'", r)
        losses = train_session.train_losses_
        self.assertGreater(len(losses), 1)
        self.assertFalse(any(map(numpy.isnan, losses)))

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_ort_gradient_optimizers_optimal_use_ort(self):
        from onnxcustom.training.orttraining import add_loss_output
        from onnxcustom.training.optimizers import OrtGradientOptimizer
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
        onx_loss = add_loss_output(onx)
        inits = ['intercept', 'coef']
        train_session = OrtGradientOptimizer(
            onx_loss, inits, max_iter=10,
            learning_rate=LearningRateSGDRegressor(learning_rate='optimal'))
        self.assertRaise(lambda: train_session.get_state(), AttributeError)
        train_session.fit(X, y, use_numpy=False)
        state_tensors = train_session.get_state()
        self.assertEqual(len(state_tensors), 2)
        r = repr(train_session)
        self.assertIn("OrtGradientOptimizer(model_onnx=", r)
        self.assertIn("learning_rate='optimal'", r)
        losses = train_session.train_losses_
        self.assertGreater(len(losses), 1)
        self.assertFalse(any(map(numpy.isnan, losses)))

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_ort_gradient_optimizers_evaluation_use_numpy(self):
        from onnxcustom.training.orttraining import add_loss_output
        from onnxcustom.training.optimizers import OrtGradientOptimizer
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=10, bias=2, random_state=0)
        X = X.astype(numpy.float32)
        y = y.astype(numpy.float32)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        reg.coef_ = reg.coef_.reshape((1, -1))
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'})
        onx_loss = add_loss_output(onx)
        inits = ['intercept', 'coef']
        train_session = OrtGradientOptimizer(onx_loss, inits)
        self.assertRaise(lambda: train_session.get_state(), AttributeError)
        train_session.fit(X, y, X_val=X_test, y_val=y_test, use_numpy=True)
        state_tensors = train_session.get_state()
        self.assertEqual(len(state_tensors), 2)
        r = repr(train_session)
        self.assertIn("OrtGradientOptimizer(model_onnx=", r)
        self.assertIn("learning_rate='invscaling'", r)
        losses = train_session.train_losses_
        self.assertGreater(len(losses), 1)
        self.assertFalse(any(map(numpy.isnan, losses)))
        vlosses = train_session.validation_losses_
        self.assertGreater(len(vlosses), 1)
        self.assertFalse(any(map(numpy.isnan, losses)))

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_ort_gradient_optimizers_evaluation_use_ort(self):
        from onnxcustom.training.orttraining import add_loss_output
        from onnxcustom.training.optimizers import OrtGradientOptimizer
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=10, bias=2, random_state=0)
        X = X.astype(numpy.float32)
        y = y.astype(numpy.float32)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        reg.coef_ = reg.coef_.reshape((1, -1))
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'})
        onx_loss = add_loss_output(onx)
        inits = ['intercept', 'coef']
        train_session = OrtGradientOptimizer(onx_loss, inits)
        self.assertRaise(lambda: train_session.get_state(), AttributeError)
        train_session.fit(X, y, X_val=X_test, y_val=y_test, use_numpy=False)
        state_tensors = train_session.get_state()
        self.assertEqual(len(state_tensors), 2)
        r = repr(train_session)
        self.assertIn("OrtGradientOptimizer(model_onnx=", r)
        self.assertIn("learning_rate='invscaling'", r)
        losses = train_session.train_losses_
        self.assertGreater(len(losses), 1)
        self.assertFalse(any(map(numpy.isnan, losses)))
        vlosses = train_session.validation_losses_
        self.assertGreater(len(vlosses), 1)
        self.assertFalse(any(map(numpy.isnan, losses)))


if __name__ == "__main__":
    unittest.main()
