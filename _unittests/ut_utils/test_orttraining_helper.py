"""
@brief      test log(time=5s)
"""

import unittest
from pyquickhelper.pycode import ExtTestCase
import numpy
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, log_loss
from mlprodict.onnx_conv import to_onnx
from mlprodict.onnxrt import OnnxInference
from mlprodict.plotting.text_plot import onnx_simple_text_plot
from onnxcustom import __max_supported_opset__ as opset
try:
    from onnxruntime import TrainingSession
except ImportError:
    # onnxruntime not training
    TrainingSession = None


class TestOrtTraining(ExtTestCase):

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_add_loss_output_reg(self):
        from onnxcustom.utils.orttraining_helper import add_loss_output
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
        onx_loss = add_loss_output(onx)
        oinf = OnnxInference(onx_loss)
        output = oinf.run({'X': X_test, 'label': y_test.reshape((-1, 1))})
        loss = output['loss']
        skl_loss = mean_squared_error(reg.predict(X_test), y_test)
        self.assertLess(numpy.abs(skl_loss - loss[0, 0]), 1e-5)

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_get_train_initializer(self):
        from onnxcustom.utils.orttraining_helper import get_train_initializer
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
        inits = get_train_initializer(onx)
        self.assertEqual({'intercept', 'coef'}, set(inits))

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_add_loss_output_cls(self):
        from onnxcustom.utils.orttraining_helper import add_loss_output
        X, y = make_classification(  # pylint: disable=W0632
            100, n_features=10)
        X = X.astype(numpy.float32)
        y = y.astype(numpy.int64)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        reg = LogisticRegression()
        reg.fit(X_train, y_train)
        reg.coef_ = reg.coef_.reshape((1, -1))
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearClassifier'},
                      options={'zipmap': False})
        onx_loss = add_loss_output(
            onx, 'log', output_index='probabilities')
        text = onnx_simple_text_plot(onx_loss)
        self.assertIn("Slice(probabilities", text)

        oinf = OnnxInference(onx_loss)
        output = oinf.run({'X': X_test, 'label': y_test.reshape((-1, 1))})
        loss = output['loss']
        skl_loss = log_loss(y_test, reg.predict(X_test))
        self.assertLess(numpy.abs(skl_loss - loss[0, 0]), 1e-5)


if __name__ == "__main__":
    unittest.main()
