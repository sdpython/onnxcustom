"""
@brief      test log(time=3s)
"""

import unittest
from pyquickhelper.pycode import ExtTestCase
import numpy
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from mlprodict.onnx_conv import to_onnx
from onnxcustom import __max_supported_opset__ as opset
try:
    from onnxruntime import TrainingSession
except ImportError:
    # onnxruntime not training
    TrainingSession = None


class TestOptimizers(ExtTestCase):

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_ort_gradient_optimizers(self):
        from onnxcustom.training.orttraining import add_loss_output
        from onnxcustom.training.optimizers import OrtGradientOptimizer
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=10, bias=2)
        X = X.astype(numpy.float32)
        y = y.astype(numpy.float32)
        X_train, _, y_train, __ = train_test_split(X, y)
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'})
        onx_loss = add_loss_output(onx)
        inits = ['intercept', 'coef']
        train_session = OrtGradientOptimizer(onx_loss, inits)
        self.assertRaise(lambda: train_session.get_state(), AttributeError)
        train_session.fit(X, y)
        state_tensors = train_session.get_state()
        self.assertEqual(len(state_tensors), 2)
        r = repr(train_session)
        self.assertIn("OrtGradientOptimizer(model_onnx=", r)
        self.assertIn("learning_rate='invscaling'", r)
        losses = train_session.train_losses_
        self.assertGreater(len(losses), 1)
        self.assertFalse(any(map(numpy.isnan, losses)))


if __name__ == "__main__":
    unittest.main()
