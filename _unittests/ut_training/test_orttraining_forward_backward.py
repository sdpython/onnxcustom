# pylint: disable=E1101,W0212
"""
@brief      test log(time=3s)
"""

import unittest
import io
import pickle
from pyquickhelper.pycode import ExtTestCase
import numpy
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from onnxruntime import InferenceSession
from onnxruntime.capi._pybind_state import (  # pylint: disable=E0611
    OrtValue as C_OrtValue, OrtDevice, OrtMemType)
from onnxruntime.capi._pybind_state import (  # pylint: disable=E0611
    OrtValueVector)
try:
    from onnxruntime import TrainingSession
except ImportError:
    # onnxruntime not training
    TrainingSession = None
from mlprodict.onnx_conv import to_onnx
from onnxcustom import __max_supported_opset__ as opset


class TestOrtTrainingForwardBackward(ExtTestCase):

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_forward_no_training(self):
        from onnxcustom.training.ortgradient import OrtGradientForwardBackward
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=10, bias=2)
        X = X.astype(numpy.float32)
        y = y.astype(numpy.float32)
        X_train, X_test, y_train, _ = train_test_split(X, y)
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'})
        forback = OrtGradientForwardBackward(onx, debug=True)
        self.assertTrue(hasattr(forback, 'args_'))
        self.assertEqual(forback.args_._onx_inp, ['X', 'coef', 'intercept'])
        self.assertEqual(forback.args_._onx_out,
                         ['X_grad', 'coef_grad', 'intercept_grad'])
        self.assertEqual(forback.args_._weights_to_train,
                         ['coef', 'intercept'])
        self.assertEqual(forback.args_._grad_input_names,
                         ['X', 'coef', 'intercept'])
        self.assertEqual(forback.args_._input_names, ['X'])
        self.assertEqual(forback.args_._bw_fetches_names,
                         ['X_grad', 'coef_grad', 'intercept_grad'])
        self.assertEqual(forback.args_._output_names, ['variable'])

        expected = reg.predict(X_test)
        coef = reg.coef_.astype(numpy.float32).reshape((-1, ))
        intercept = numpy.array([reg.intercept_], dtype=numpy.float32)

        sess0 = InferenceSession(onx.SerializeToString())
        inames = [i.name for i in sess0.get_inputs()]  # pylint: disable=E1101
        self.assertEqual(inames, ['X'])
        got = sess0.run(None, {'X': X_test})
        self.assertEqualArray(expected, got[0].ravel(), decimal=4)

        sess_eval = forback.args_._sess_eval  # pylint: disable=E1101
        inames = [i.name for i in sess_eval.get_inputs()]
        self.assertEqual(inames, ['X', 'coef', 'intercept'])
        got = sess_eval.run(
            None, {'X': X_test, 'coef': coef, 'intercept': intercept})
        self.assertEqualArray(expected, got[0].ravel(), decimal=4)

        # OrtValue
        inputs = []
        device = OrtDevice(OrtDevice.cpu(), OrtMemType.DEFAULT, 0)
        for a in [X_test, coef, intercept]:
            inputs.append(C_OrtValue.ortvalue_from_numpy(a, device))
        got_ort = forback.forward(inputs)
        got = [v.numpy() for v in got_ort]
        self.assertEqual(len(got), 1)
        self.assertEqualArray(expected, got[0].ravel(), decimal=4)

        # OrtValueVector
        inputs = OrtValueVector()
        for a in [X_test, coef, intercept]:
            inputs.push_back(C_OrtValue.ortvalue_from_numpy(a, device))
        got = forback.forward(inputs)
        self.assertEqual(len(got), 1)
        self.assertEqualArray(expected, got[0].numpy().ravel(), decimal=4)

        # numpy
        inputs = [X_test, coef, intercept]
        got = forback.forward(inputs)
        self.assertEqual(len(got), 1)
        self.assertEqualArray(expected, got[0].numpy().ravel(), decimal=4)

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_forward_no_training_pickle(self):
        from onnxcustom.training.ortgradient import OrtGradientForwardBackward
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=10, bias=2)
        X = X.astype(numpy.float32)
        y = y.astype(numpy.float32)
        X_train, X_test, y_train, _ = train_test_split(X, y)
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'})
        forback0 = OrtGradientForwardBackward(onx, debug=True)
        st = io.BytesIO()
        pickle.dump(forback0, st)
        st2 = io.BytesIO(st.getvalue())
        forback = pickle.load(st2)

        self.assertTrue(hasattr(forback, 'args_'))
        self.assertEqual(forback.args_._onx_inp, ['X', 'coef', 'intercept'])
        self.assertEqual(forback.args_._onx_out,
                         ['X_grad', 'coef_grad', 'intercept_grad'])
        self.assertEqual(forback.args_._weights_to_train,
                         ['coef', 'intercept'])
        self.assertEqual(forback.args_._grad_input_names,
                         ['X', 'coef', 'intercept'])
        self.assertEqual(forback.args_._input_names, ['X'])
        self.assertEqual(forback.args_._bw_fetches_names,
                         ['X_grad', 'coef_grad', 'intercept_grad'])
        self.assertEqual(forback.args_._output_names, ['variable'])

        expected = reg.predict(X_test)
        coef = reg.coef_.astype(numpy.float32).reshape((-1, ))
        intercept = numpy.array([reg.intercept_], dtype=numpy.float32)

        sess0 = InferenceSession(onx.SerializeToString())
        inames = [i.name for i in sess0.get_inputs()]
        self.assertEqual(inames, ['X'])
        got = sess0.run(None, {'X': X_test})
        self.assertEqualArray(expected, got[0].ravel(), decimal=4)

        sess_eval = forback.args_._sess_eval  # pylint: disable=W0212
        inames = [i.name for i in sess_eval.get_inputs()]
        self.assertEqual(inames, ['X', 'coef', 'intercept'])
        got = sess_eval.run(
            None, {'X': X_test, 'coef': coef, 'intercept': intercept})
        self.assertEqualArray(expected, got[0].ravel(), decimal=4)

        # OrtValue
        inputs = []
        device = OrtDevice(OrtDevice.cpu(), OrtMemType.DEFAULT, 0)
        for a in [X_test, coef, intercept]:
            inputs.append(C_OrtValue.ortvalue_from_numpy(a, device))
        got_ort = forback.forward(inputs)
        got = [v.numpy() for v in got_ort]
        self.assertEqual(len(got), 1)
        self.assertEqualArray(expected, got[0].ravel(), decimal=4)

        # OrtValueVector
        inputs = OrtValueVector()
        for a in [X_test, coef, intercept]:
            inputs.push_back(C_OrtValue.ortvalue_from_numpy(a, device))
        got = forback.forward(inputs)
        self.assertEqual(len(got), 1)
        self.assertEqualArray(expected, got[0].numpy().ravel(), decimal=4)

        # numpy
        inputs = [X_test, coef, intercept]
        got = forback.forward(inputs)
        self.assertEqual(len(got), 1)
        self.assertEqualArray(expected, got[0].numpy().ravel(), decimal=4)


if __name__ == "__main__":
    unittest.main()
