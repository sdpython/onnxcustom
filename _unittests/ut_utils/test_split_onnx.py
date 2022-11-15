"""
@brief      test log(time=11s)
"""
import unittest
import numpy
from scipy.special import expit  # pylint: disable=E0611
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor, MLPClassifier
from onnxruntime.capi._pybind_state import (  # pylint: disable=E0611
    OrtValue as C_OrtValue)
try:
    from onnxruntime import TrainingSession
except ImportError:
    # onnxruntime not training
    TrainingSession = None
from mlprodict.onnx_conv import to_onnx
# from mlprodict.onnxrt import OnnxInference
from mlprodict.plotting.text_plot import onnx_simple_text_plot
from mlprodict.onnx_tools.onnx_manipulations import select_model_inputs_outputs
from onnxcustom import __max_supported_opset__ as opset
from onnxcustom.utils.onnx_split import split_onnx


class TestSplitOnnx(ExtTestCase):

    def test_split_onnx(self):
        X = numpy.arange(30).reshape((-1, 3)).astype(numpy.float32) / 100
        y = numpy.arange(X.shape[0]).astype(numpy.float32)
        y = y.reshape((-1, 1))
        reg = MLPRegressor(hidden_layer_sizes=(5,), max_iter=2,
                           activation='logistic',
                           momentum=0, nesterovs_momentum=False,
                           alpha=0)
        reg.fit(X, y.ravel())

        onx = to_onnx(reg, X, target_opset=opset)
        print(onnx_simple_text_plot(onx))
        split = split_onnx(onx, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
