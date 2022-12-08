# pylint: disable=E1101,W0212,E0611
"""
@brief      test log(time=4s)
"""
import unittest
import logging
import os
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
import numpy
import onnx
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from onnxruntime import InferenceSession
from onnxruntime.quantization.calibrate import CalibrationDataReader
from onnxruntime.quantization.quantize import quantize_static
try:
    from onnxruntime import TrainingSession
except ImportError:
    # onnxruntime not training
    TrainingSession = None
from mlprodict.onnx_conv import to_onnx
from onnxcustom import __max_supported_opset__ as opset
from onnxcustom.utils.onnx_helper import onnx_rename_weights
from onnxcustom.training.sgd_learning_loss import SquareLearningLoss
from onnxcustom.training.grad_helper import onnx_derivative


class DataReader(CalibrationDataReader):
    def __init__(self, input_name, data):
        self.input_name = input_name
        self.data = data
        self.pos = -1

    def get_next(self):
        if self.pos >= self.data.shape[0] - 1:
            return None
        self.pos += 1
        return {self.input_name: self.data[self.pos:self.pos + 1]}

    def rewind(self):
        self.pos = -1


class TestOrtTrainingForwardBackward(ExtTestCase):

    def setUp(self):
        logger = logging.getLogger('skl2onnx')
        logger.setLevel(logging.WARNING)
        logger = logging.getLogger('onnxcustom')
        logger.setLevel(logging.WARNING)
        logger = logging.getLogger('root')
        logger.setLevel(logging.ERROR)
        logging.basicConfig(level=logging.ERROR)

    def tearDown(self):
        logger = logging.getLogger('skl2onnx')
        logger.setLevel(logging.WARNING)
        logger = logging.getLogger('onnxcustom')
        logger.setLevel(logging.WARNING)
        logger = logging.getLogger('root')
        logger.setLevel(logging.WARNING)
        logging.basicConfig(level=logging.WARNING)

    @unittest.skipIf(TrainingSession is None, reason="no training available")
    def test_qat(self):
        from onnxcustom.training.optimizers_partial import OrtGradientForwardBackwardOptimizer
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=10, bias=2, random_state=0)
        X = X.astype(numpy.float32)
        y = y.astype(numpy.float32)
        X_train, _, y_train, __ = train_test_split(X, y)
        reg = MLPRegressor(hidden_layer_sizes=(3,), activation='logistic')
        reg.fit(X_train, y_train)
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'})
        temp = get_temp_folder(__file__, "temp_qat", clean=False)
        name = os.path.join(temp, "model.onnx")
        with open(name, "wb") as f:
            f.write(onx.SerializeToString())
        quan = os.path.join(temp, "model.qdq.onnx")
        logging.basicConfig(level=logging.ERROR)
        quantize_static(name, quan, DataReader('X', X))
        logging.basicConfig(level=logging.WARNING)
        with open(quan, "rb") as f:
            qdq_onx = onnx.load(f)

        sess = InferenceSession(onx.SerializeToString())
        sessq = InferenceSession(qdq_onx.SerializeToString())
        expected = sess.run(None, {'X': X})[0]
        gotq = sessq.run(None, {'X': X})[0]
        errorq = numpy.abs(expected - gotq).ravel().mean()

        qdq_onx2 = onnx_rename_weights(qdq_onx)
        with open(os.path.join(temp, "model.qdq2.onnx"), "wb") as f:
            f.write(qdq_onx2.SerializeToString())
        params = [init.name for init in qdq_onx2.graph.initializer]
        params = [name for name in params
                  if '_scale' in name or '_zero_point' in name]

        train_session = OrtGradientForwardBackwardOptimizer(
            qdq_onx2, params, learning_loss=SquareLearningLoss())
        try:
            onnx_derivative(qdq_onx2, weights=params)
        except RuntimeError:
            # QAT is not implemented yet in onnxruntime.
            return

        train_session.fit(X, expected)
        trained = train_session.get_trained_onnx()
        with open(os.path.join(temp, "model.qat.onnx"), "wb") as f:
            f.write(trained.SerializeToString())
        sessqq = InferenceSession(trained.SerializeToString())
        gotqq = sessqq.run(None, {'X': X})[0]
        errorqq = numpy.abs(expected - gotqq).ravel().mean()
        self.assertLess(errorqq, errorq)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
