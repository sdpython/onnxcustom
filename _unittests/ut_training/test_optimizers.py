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
from onnxcustom.training.orttraining import add_loss_output
from onnxcustom.training.optimizers import OrtGradientOptimizer


class TestOptimizers(ExtTestCase):

    def test_ort_gradient_optimizers(self):
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


if __name__ == "__main__":
    unittest.main()
