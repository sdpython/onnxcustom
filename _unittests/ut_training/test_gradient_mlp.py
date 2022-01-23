"""
@brief      test log(time=8s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from onnxruntime.capi._pybind_state import (  # pylint: disable=E0611
    OrtValue as C_OrtValue)
try:
    from onnxruntime import TrainingSession
except ImportError:
    # onnxruntime not training
    TrainingSession = None
from mlprodict.onnx_conv import to_onnx
# from mlprodict.onnxrt import OnnxInference
from onnxcustom import __max_supported_opset__ as opset
from onnxcustom.utils.onnx_helper import onnx_rename_weights


class TestGradientMlp(ExtTestCase):

    @unittest.skipIf(TrainingSession is None, reason="not training")
    @ignore_warnings(ConvergenceWarning)
    def test_gradient_mlpregressor(self):
        from onnxcustom.training.optimizers_partial import (
            OrtGradientForwardBackwardOptimizer)
        X = numpy.arange(30).reshape((-1, 3)).astype(numpy.float32) / 100
        y = numpy.arange(X.shape[0]).astype(numpy.float32)
        y = y.reshape((-1, 1))
        reg = MLPRegressor(hidden_layer_sizes=(5,), max_iter=2,
                           activation='logistic',
                           momentum=0, nesterovs_momentum=False,
                           alpha=0)
        reg.fit(X, y.ravel())
        onx = to_onnx(reg, X, target_opset=opset)

        xp = numpy.arange(2 * X.shape[1]).reshape((2, -1)).astype(
            numpy.float32) / 10
        yp = numpy.array([0.5, -0.5], dtype=numpy.float32).reshape((-1, 1))

        onx = to_onnx(reg, X, target_opset=opset)
        onx = onnx_rename_weights(onx)
        inits = ["I0_coefficient", 'I1_intercepts', 'I2_coefficient1',
                 'I3_intercepts1']

        train_session = OrtGradientForwardBackwardOptimizer(
            onx, inits, learning_rate=1e-5,
            warm_start=True, max_iter=2, batch_size=10)
        train_session.fit(X, y)
        state = train_session.get_state()
        state_np = [st.numpy() for st in state]

        # gradient scikit-learn

        coef_grads = state_np[::2]
        intercept_grads = state_np[1::2]
        layer_units = [3, 5, 1]
        activations = [xp] + [None] * (len(layer_units) - 1)
        deltas = [None] * (len(activations) - 1)

        skl_pred = reg.predict(xp)

        batch_loss, coef_grads, intercept_grads = reg._backprop(  # pylint: disable=W0212
            xp, yp, activations, deltas,
            coef_grads, intercept_grads)
        deltas = activations[-1] - yp

        # gradient onnxcustom

        ort_xp = C_OrtValue.ortvalue_from_numpy(xp, train_session.device)
        ort_yp = C_OrtValue.ortvalue_from_numpy(yp, train_session.device)
        ort_state = [ort_xp] + state
        prediction = train_session.train_function_.forward(
            ort_state, training=True)

        ort_pred = prediction[0].numpy()
        self.assertEqualArray(skl_pred.ravel(), ort_pred.ravel(), decimal=2)

        loss, loss_gradient = train_session.learning_loss.loss_gradient(
            train_session.device, ort_yp, prediction[0])

        gradient = train_session.train_function_.backward([loss_gradient])

        # comparison

        self.assertEqualArray(
            batch_loss, loss.numpy() / xp.shape[0], decimal=3)
        self.assertEqualArray(deltas, loss_gradient.numpy(), decimal=3)

        ort_grad = [g.numpy() / xp.shape[0] for g in gradient][1:]
        self.assertEqualArray(
            intercept_grads[1], ort_grad[3].ravel(), decimal=2)
        self.assertEqualArray(coef_grads[1], ort_grad[2], decimal=2)
        self.assertEqualArray(
            intercept_grads[0], ort_grad[1].ravel(), decimal=2)
        self.assertEqualArray(coef_grads[0], ort_grad[0], decimal=2)


if __name__ == "__main__":
    unittest.main()
