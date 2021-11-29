# pylint: disable=E1101,W0212
"""
@brief      test log(time=3s)
"""
import unittest
import io
import pickle
import logging
import os
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
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

    @unittest.skipIf(TrainingSession is None, reason="no training")
    def forward_no_training(self):
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

        # starts testing
        forback = OrtGradientForwardBackward(
            onx, debug=True, enable_logging=True)
        self.assertTrue(hasattr(forback, 'cls_type_'))
        self.assertEqual(forback.cls_type_._onx_inp, ['X', 'coef', 'intercept'])
        self.assertEqual(forback.cls_type_._onx_out,
                         ['X_grad', 'coef_grad', 'intercept_grad'])
        self.assertEqual(forback.cls_type_._weights_to_train,
                         ['coef', 'intercept'])
        self.assertEqual(forback.cls_type_._grad_input_names,
                         ['X', 'coef', 'intercept'])
        self.assertEqual(forback.cls_type_._input_names, ['X'])
        self.assertEqual(forback.cls_type_._bw_fetches_names,
                         ['X_grad', 'coef_grad', 'intercept_grad'])
        self.assertEqual(forback.cls_type_._output_names, ['variable'])

        expected = reg.predict(X_test)
        coef = reg.coef_.astype(numpy.float32).reshape((-1, ))
        intercept = numpy.array([reg.intercept_], dtype=numpy.float32)

        sess0 = InferenceSession(onx.SerializeToString())
        inames = [i.name for i in sess0.get_inputs()]  # pylint: disable=E1101
        self.assertEqual(inames, ['X'])
        got = sess0.run(None, {'X': X_test})
        self.assertEqualArray(expected, got[0].ravel(), decimal=4)

        sess_eval = forback.cls_type_._sess_eval  # pylint: disable=E1101
        inames = [i.name for i in sess_eval.get_inputs()]
        self.assertEqual(inames, ['X', 'coef', 'intercept'])
        got = sess_eval.run(
            None, {'X': X_test, 'coef': coef, 'intercept': intercept})
        self.assertEqualArray(expected, got[0].ravel(), decimal=4)

        # OrtValue
        inst = forback.new_instance()
        device = OrtDevice(OrtDevice.cpu(), OrtMemType.DEFAULT, 0)

        # list of OrtValues
        inputs = []
        for a in [X_test, coef, intercept]:
            inputs.append(C_OrtValue.ortvalue_from_numpy(a, device))
        got_ort = inst.forward(inputs)
        got = [v.numpy() for v in got_ort]
        self.assertEqual(len(got), 1)
        self.assertEqualArray(expected, got[0].ravel(), decimal=4)

        # OrtValueVector
        inputs = OrtValueVector()
        for a in [X_test, coef, intercept]:
            inputs.push_back(C_OrtValue.ortvalue_from_numpy(a, device))
        got = inst.forward(inputs)
        self.assertEqual(len(got), 1)
        self.assertEqualArray(expected, got[0].numpy().ravel(), decimal=4)

        # numpy
        inputs = [X_test, coef, intercept]
        got = inst.forward(inputs)
        self.assertEqual(len(got), 1)
        self.assertEqualArray(expected, got[0].numpy().ravel(), decimal=4)

    @unittest.skipIf(TrainingSession is None, reason="no training")
    def test_forward_no_training(self):
        res, logs = self.assertLogging(
            self.forward_no_training, 'onnxcustom')
        self.assertEmpty(res)
        self.assertIn("[OrtGradientForwardBackward]", logs)
        self.assertIn("weights_to_train=['coef', 'intercept']", logs)

    @unittest.skipIf(TrainingSession is None, reason="no training")
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

        self.assertTrue(hasattr(forback, 'cls_type_'))
        self.assertEqual(forback.cls_type_._onx_inp, ['X', 'coef', 'intercept'])
        self.assertEqual(forback.cls_type_._onx_out,
                         ['X_grad', 'coef_grad', 'intercept_grad'])
        self.assertEqual(forback.cls_type_._weights_to_train,
                         ['coef', 'intercept'])
        self.assertEqual(forback.cls_type_._grad_input_names,
                         ['X', 'coef', 'intercept'])
        self.assertEqual(forback.cls_type_._input_names, ['X'])
        self.assertEqual(forback.cls_type_._bw_fetches_names,
                         ['X_grad', 'coef_grad', 'intercept_grad'])
        self.assertEqual(forback.cls_type_._output_names, ['variable'])

        expected = reg.predict(X_test)
        coef = reg.coef_.astype(numpy.float32).reshape((-1, ))
        intercept = numpy.array([reg.intercept_], dtype=numpy.float32)

        sess0 = InferenceSession(onx.SerializeToString())
        inames = [i.name for i in sess0.get_inputs()]
        self.assertEqual(inames, ['X'])
        got = sess0.run(None, {'X': X_test})
        self.assertEqualArray(expected, got[0].ravel(), decimal=4)

        sess_eval = forback.cls_type_._sess_eval  # pylint: disable=W0212
        inames = [i.name for i in sess_eval.get_inputs()]
        self.assertEqual(inames, ['X', 'coef', 'intercept'])
        got = sess_eval.run(
            None, {'X': X_test, 'coef': coef, 'intercept': intercept})
        self.assertEqualArray(expected, got[0].ravel(), decimal=4)

        # OrtValue
        inst = forback.new_instance()
        inputs = []
        device = OrtDevice(OrtDevice.cpu(), OrtMemType.DEFAULT, 0)
        for a in [X_test, coef, intercept]:
            inputs.append(C_OrtValue.ortvalue_from_numpy(a, device))
        got_ort = inst.forward(inputs)
        got = [v.numpy() for v in got_ort]
        self.assertEqual(len(got), 1)
        self.assertEqualArray(expected, got[0].ravel(), decimal=4)

        # OrtValueVector
        inputs = OrtValueVector()
        for a in [X_test, coef, intercept]:
            inputs.push_back(C_OrtValue.ortvalue_from_numpy(a, device))
        got = inst.forward(inputs)
        self.assertEqual(len(got), 1)
        self.assertEqualArray(expected, got[0].numpy().ravel(), decimal=4)

        # numpy
        inputs = [X_test, coef, intercept]
        got = inst.forward(inputs)
        self.assertEqual(len(got), 1)
        self.assertEqualArray(expected, got[0].numpy().ravel(), decimal=4)

    @unittest.skipIf(TrainingSession is None, reason="no training")
    def forward_training(self):
        from onnxcustom.training.ortgradient import OrtGradientForwardBackward
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=10, bias=2)
        X = X.astype(numpy.float32)
        y = y.astype(numpy.float32)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'})

        # starts testing
        forback = OrtGradientForwardBackward(
            onx, debug=True, enable_logging=True)
        temp = get_temp_folder(__file__, "temp_forward_training")
        with open(os.path.join(temp, "fw_train.onnx"), "wb") as f:
            f.write(forback.cls_type_._trained_onnx.SerializeToString())
        with open(os.path.join(temp, "fw_pre.onnx"), "wb") as f:
            gr = forback.cls_type_._optimized_pre_grad_model
            f.write(gr.SerializeToString())

        X_test = X_test[:1]
        expected = reg.predict(X_test)
        coef = reg.coef_.astype(numpy.float32).reshape((-1, ))
        intercept = numpy.array([reg.intercept_], dtype=numpy.float32)

        # OrtValue
        inst = forback.new_instance()
        device = OrtDevice(OrtDevice.cpu(), OrtMemType.DEFAULT, 0)

        # OrtValueVector
        inputs = OrtValueVector()
        for a in [X_test, coef, intercept]:
            inputs.push_back(C_OrtValue.ortvalue_from_numpy(a, device))
        got = inst.forward(inputs, training=True)
        self.assertEqual(len(got), 1)
        self.assertEqualArray(expected, got[0].numpy().ravel(), decimal=4)

        outputs = OrtValueVector()
        outputs.push_back(C_OrtValue.ortvalue_from_numpy(
            y_test.reshape((-1, 1)), device))
        got = inst.backward(outputs)
        self.assertEqual(len(got), 3)

        # list of OrtValues
        inputs = []
        for a in [X_test, coef, intercept]:
            inputs.append(C_OrtValue.ortvalue_from_numpy(a, device))
        got_ort = inst.forward(inputs, training=True)
        got = [v.numpy() for v in got_ort]
        self.assertEqual(len(got), 1)
        self.assertEqualArray(expected, got[0].ravel(), decimal=4)

        # numpy
        inputs = [X_test, coef, intercept]
        got = inst.forward(inputs, training=True)
        self.assertEqual(len(got), 1)
        self.assertEqualArray(expected, got[0].numpy().ravel(), decimal=4)

    @unittest.skipIf(TrainingSession is None, reason="no training")
    def test_forward_training(self):
        res, logs = self.assertLogging(
            self.forward_training, 'onnxcustom', level=logging.DEBUG, console=True)
        self.assertEmpty(res)
        self.assertIn("[OrtGradientForwardBackward]", logs)
        self.assertIn("weights_to_train=['coef', 'intercept']", logs)


if __name__ == "__main__":
    unittest.main()
