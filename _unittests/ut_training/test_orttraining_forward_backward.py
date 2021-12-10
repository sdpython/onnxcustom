# pylint: disable=E1101,W0212,E0611
"""
@brief      test log(time=9s)
"""
import unittest
import io
import pickle
import logging
import os
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
import numpy
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from onnxruntime import InferenceSession
try:
    from onnxruntime import TrainingSession
except ImportError:
    # onnxruntime not training
    TrainingSession = None
from mlprodict.onnx_conv import to_onnx
from mlprodict.onnx_tools.onnx_manipulations import select_model_inputs_outputs
from onnxcustom import __max_supported_opset__ as opset


class TestOrtTrainingForwardBackward(ExtTestCase):

    def forward_no_training(self):
        from onnxruntime.capi._pybind_state import (
            OrtValue as C_OrtValue, OrtDevice, OrtMemType)
        from onnxruntime.capi._pybind_state import (
            OrtValueVector)
        from onnxcustom.training.ortgradient import OrtGradientForwardBackward
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=10, bias=2)
        X = X.astype(numpy.float32)
        y = y.astype(numpy.float32)
        X_train, X_test, y_train, _ = train_test_split(X, y)
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        reg.coef_ = reg.coef_.reshape((1, -1))
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'})

        # starts testing
        self.assertRaise(
            lambda: OrtGradientForwardBackward(
                onx, debug=True, enable_logging=True, weights_to_train=[]),
            ValueError)
        self.assertRaise(
            lambda: OrtGradientForwardBackward(
                onx, debug=True, enable_logging=True, providers=['NONE']),
            ValueError)
        forback = OrtGradientForwardBackward(
            onx, debug=True, enable_logging=True)
        self.assertEqual(repr(forback), "OrtGradientForwardBackward(...)")
        self.assertTrue(hasattr(forback, 'cls_type_'))
        self.assertEqual(forback.cls_type_._onx_inp,
                         ['X', 'coef', 'intercept'])
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
        coef = reg.coef_.astype(numpy.float32).reshape((-1, 1))
        intercept = numpy.array([reg.intercept_], dtype=numpy.float32)

        sess0 = InferenceSession(onx.SerializeToString())
        inames = [i.name for i in sess0.get_inputs()]  # pylint: disable=E1101
        self.assertEqual(inames, ['X'])
        got = sess0.run(None, {'X': X_test})
        self.assertEqualArray(expected.ravel(), got[0].ravel(), decimal=4)

        sess_eval = forback.cls_type_._sess_eval  # pylint: disable=E1101
        inames = [i.name for i in sess_eval.get_inputs()]
        self.assertEqual(inames, ['X', 'coef', 'intercept'])
        got = sess_eval.run(
            None, {'X': X_test, 'coef': coef, 'intercept': intercept})
        self.assertEqualArray(expected.ravel(), got[0].ravel(), decimal=4)

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
        self.assertEqualArray(expected.ravel(), got[0].ravel(), decimal=4)

        # OrtValueVector
        inputs = OrtValueVector()
        for a in [X_test, coef, intercept]:
            inputs.push_back(C_OrtValue.ortvalue_from_numpy(a, device))
        got = inst.forward(inputs)
        self.assertEqual(len(got), 1)
        self.assertEqualArray(
            expected.ravel(), got[0].numpy().ravel(), decimal=4)

        # numpy
        inputs = [X_test, coef, intercept]
        got = inst.forward(inputs)
        self.assertEqual(len(got), 1)
        self.assertEqualArray(
            expected.ravel(), got[0].numpy().ravel(), decimal=4)

    @unittest.skipIf(TrainingSession is None, reason="no training")
    def test_forward_no_training(self):
        res, logs = self.assertLogging(
            self.forward_no_training, 'onnxcustom')
        self.assertEmpty(res)
        if len(logs) > 0:
            self.assertIn("[OrtGradientForwardBackward]", logs)
            self.assertIn("weights_to_train=['coef', 'intercept']", logs)

    @unittest.skipIf(TrainingSession is None, reason="no training")
    def test_forward_no_training_pickle(self):
        from onnxruntime.capi._pybind_state import (
            OrtValue as C_OrtValue, OrtDevice, OrtMemType)
        from onnxruntime.capi._pybind_state import (
            OrtValueVector)
        from onnxcustom.training.ortgradient import OrtGradientForwardBackward
        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=10, bias=2)
        X = X.astype(numpy.float32)
        y = y.astype(numpy.float32)
        X_train, X_test, y_train, _ = train_test_split(X, y)
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        reg.coef_ = reg.coef_.reshape((1, -1))
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'})
        forback0 = OrtGradientForwardBackward(onx, debug=True)
        st = io.BytesIO()
        pickle.dump(forback0, st)
        st2 = io.BytesIO(st.getvalue())
        forback = pickle.load(st2)

        self.assertTrue(hasattr(forback, 'cls_type_'))
        self.assertEqual(forback.cls_type_._onx_inp,
                         ['X', 'coef', 'intercept'])
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
        coef = reg.coef_.astype(numpy.float32).reshape((-1, 1))
        intercept = numpy.array([reg.intercept_], dtype=numpy.float32)

        sess0 = InferenceSession(onx.SerializeToString())
        inames = [i.name for i in sess0.get_inputs()]
        self.assertEqual(inames, ['X'])
        got = sess0.run(None, {'X': X_test})
        self.assertEqualArray(expected.ravel(), got[0].ravel(), decimal=4)

        sess_eval = forback.cls_type_._sess_eval  # pylint: disable=W0212
        inames = [i.name for i in sess_eval.get_inputs()]
        self.assertEqual(inames, ['X', 'coef', 'intercept'])
        got = sess_eval.run(
            None, {'X': X_test, 'coef': coef, 'intercept': intercept})
        self.assertEqualArray(expected.ravel(), got[0].ravel(), decimal=4)

        # OrtValue
        inst = forback.new_instance()
        inputs = []
        device = OrtDevice(OrtDevice.cpu(), OrtMemType.DEFAULT, 0)
        for a in [X_test, coef, intercept]:
            inputs.append(C_OrtValue.ortvalue_from_numpy(a, device))
        got_ort = inst.forward(inputs)
        got = [v.numpy() for v in got_ort]
        self.assertEqual(len(got), 1)
        self.assertEqualArray(
            expected.ravel(), got[0].ravel(), decimal=4)

        # OrtValueVector
        inputs = OrtValueVector()
        for a in [X_test, coef, intercept]:
            inputs.push_back(C_OrtValue.ortvalue_from_numpy(a, device))
        got = inst.forward(inputs)
        self.assertEqual(len(got), 1)
        self.assertEqualArray(
            expected.ravel(), got[0].numpy().ravel(), decimal=4)

        # numpy
        inputs = [X_test, coef, intercept]
        got = inst.forward(inputs)
        self.assertEqual(len(got), 1)
        self.assertEqualArray(
            expected.ravel(), got[0].numpy().ravel(), decimal=4)

    def forward_training(self, model, debug=False, n_classes=3, add_print=False):
        from onnxruntime.capi._pybind_state import (
            OrtValue as C_OrtValue, OrtDevice, OrtMemType)
        from onnxruntime.capi._pybind_state import (
            OrtValueVector)
        from onnxcustom.training.ortgradient import OrtGradientForwardBackward

        def to_proba(yt):
            mx = yt.max() + 1
            new_yt = numpy.zeros((yt.shape[0], mx), dtype=numpy.float32)
            for i, y in enumerate(yt):
                new_yt[i, y] = 1
            return new_yt

        if hasattr(model.__class__, 'predict_proba'):
            X, y = make_classification(  # pylint: disable=W0632
                100, n_features=10, n_classes=n_classes,
                n_informative=7)
            X = X.astype(numpy.float32)
            y = y.astype(numpy.int64)
        else:
            X, y = make_regression(  # pylint: disable=W0632
                100, n_features=10, bias=2)
            X = X.astype(numpy.float32)
            y = y.astype(numpy.float32)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        reg = model
        reg.fit(X_train, y_train)
        # needs if skl2onnx<1.10.4
        # reg.coef_ = reg.coef_.reshape((1, -1))
        # reg.intercept_ = reg.intercept_.reshape((-1, ))
        if hasattr(model.__class__, 'predict_proba'):
            onx = to_onnx(reg, X_train, target_opset=opset,
                          black_op={'LinearClassifier'},
                          options={'zipmap': False})
            onx = select_model_inputs_outputs(
                onx, outputs=[onx.graph.output[1].name])
        else:
            onx = to_onnx(reg, X_train, target_opset=opset,
                          black_op={'LinearRegressor'})

        # remove batch possibility
        #onx.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 0
        #onx.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "batch_size"
        #onx.graph.output[0].type.tensor_type.shape.dim[0].dim_value = 0
        #onx.graph.output[0].type.tensor_type.shape.dim[0].dim_param = "batch_size"
        sess = InferenceSession(onx.SerializeToString())
        sess.run(None, {'X': X_test[:1]})

        # starts testing
        forback = OrtGradientForwardBackward(
            onx, debug=True, enable_logging=True)
        if debug:
            n = model.__class__.__name__
            temp = get_temp_folder(__file__, "temp_forward_training_%s" % n)
            with open(os.path.join(temp, "model_%s.onnx" % n), "wb") as f:
                f.write(onx.SerializeToString())
            with open(os.path.join(temp, "fw_train_%s.onnx" % n), "wb") as f:
                f.write(forback.cls_type_._trained_onnx.SerializeToString())
            with open(os.path.join(temp, "fw_pre_%s.onnx" % n), "wb") as f:
                gr = forback.cls_type_._optimized_pre_grad_model
                f.write(gr.SerializeToString())

        if hasattr(model.__class__, 'predict_proba'):
            expected = reg.predict_proba(X_test)
            coef = reg.coef_.astype(numpy.float32).T
            intercept = reg.intercept_.astype(numpy.float32)
            # only one observation
            X_test1 = X_test[:1]
            y_test = to_proba(y_test).astype(numpy.float32)
            y_test1 = y_test[:1]
            expected1 = expected[:1]
        else:
            expected = reg.predict(X_test)
            coef = reg.coef_.astype(numpy.float32).reshape((-1, 1))
            intercept = numpy.array([reg.intercept_], dtype=numpy.float32)
            # only one observation
            X_test1 = X_test[:1]
            y_test1 = y_test[0].reshape((1, -1))
            expected1 = expected[:1]

        # OrtValue
        inst = forback.new_instance()
        device = OrtDevice(OrtDevice.cpu(), OrtMemType.DEFAULT, 0)

        # OrtValueVector
        if add_print:
            print("\n\n######################\nFORWARD")
        inputs = OrtValueVector()
        for a in [X_test1, coef, intercept]:
            inputs.push_back(C_OrtValue.ortvalue_from_numpy(a, device))
        got = inst.forward(inputs, training=True)
        self.assertEqual(len(got), 1)
        self.assertEqualArray(
            expected1.ravel(), got[0].numpy().ravel(), decimal=4)

        if add_print:
            print("\n\n######################\nBACKWARD")
        outputs = OrtValueVector()
        outputs.push_back(C_OrtValue.ortvalue_from_numpy(
            y_test1, device))
        got = inst.backward(outputs)
        self.assertEqual(len(got), 3)
        if add_print:
            print("\n######################\nEND\n")

        # OrtValueVectorN
        inputs = OrtValueVector()
        for a in [X_test, coef, intercept]:
            inputs.push_back(C_OrtValue.ortvalue_from_numpy(a, device))
        got = inst.forward(inputs, training=True)
        self.assertEqual(len(got), 1)
        self.assertEqualArray(
            expected.ravel(), got[0].numpy().ravel(), decimal=4)

        outputs = OrtValueVector()
        outputs.push_back(C_OrtValue.ortvalue_from_numpy(
            y_test.reshape((1, -1)), device))
        got = inst.backward(outputs)
        self.assertEqual(len(got), 3)

        # list of OrtValues
        inputs = []
        for a in [X_test, coef, intercept]:
            inputs.append(C_OrtValue.ortvalue_from_numpy(a, device))
        got_ort = inst.forward(inputs, training=True)
        got = [v.numpy() for v in got_ort]
        self.assertEqual(len(got), 1)
        self.assertEqualArray(expected.ravel(), got[0].ravel(), decimal=4)

        outputs = [C_OrtValue.ortvalue_from_numpy(
            y_test.reshape((1, -1)), device)]
        got = inst.backward(outputs)
        self.assertEqual(len(got), 3)

        # numpy
        inputs = [X_test, coef, intercept]
        got_ort = inst.forward(inputs, training=True)
        got = [v.numpy() for v in got_ort]
        self.assertEqual(len(got), 1)
        self.assertEqualArray(expected.ravel(), got[0].ravel(), decimal=4)

        outputs = [y_test.reshape((1, -1))]
        got = inst.backward(outputs)
        self.assertEqual(len(got), 3)

    @unittest.skipIf(TrainingSession is None, reason="no training")
    def test_forward_training_linreg(self):
        res, logs = self.assertLogging(
            lambda: self.forward_training(LinearRegression(), debug=True),
            'onnxcustom', level=logging.DEBUG)
        self.assertEmpty(res)
        if len(logs) > 0:
            self.assertIn("[OrtGradientForwardBackward]", logs)
            self.assertIn("weights_to_train=['coef', 'intercept']", logs)

    @unittest.skipIf(TrainingSession is None, reason="no training")
    def test_forward_training_logreg(self):
        res, logs = self.assertLogging(
            lambda: self.forward_training(
                LogisticRegression(), debug=False),
            'onnxcustom', level=logging.DEBUG)
        self.assertEmpty(res)
        if len(logs) > 0:
            self.assertIn("[OrtGradientForwardBackward]", logs)
            self.assertIn("weights_to_train=['coef', 'intercept']", logs)


if __name__ == "__main__":
    unittest.main()
