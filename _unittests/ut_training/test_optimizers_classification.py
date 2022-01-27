"""
@brief      test log(time=8s)
"""
import unittest
import logging
import numpy
from onnx import TensorProto
from onnx.helper import set_model_props
from pyquickhelper.pycode import (
    ExtTestCase, get_temp_folder, ignore_warnings)
from sklearn.exceptions import ConvergenceWarning
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from mlprodict.onnx_conv import to_onnx
from mlprodict.plotting.text_plot import onnx_simple_text_plot
from mlprodict.onnx_tools.onnx_manipulations import select_model_inputs_outputs
# from mlprodict.onnxrt import OnnxInference
from onnxcustom import __max_supported_opset__ as opset
from onnxcustom.utils.onnx_helper import onnx_rename_weights
try:
    from onnxruntime import TrainingSession
except ImportError:
    # onnxruntime not training
    TrainingSession = None
from onnxruntime.capi._pybind_state import (  # pylint: disable=E0611
    OrtDevice as C_OrtDevice, OrtValue as C_OrtValue)


class TestOptimizersClassification(ExtTestCase):

    @classmethod
    def setUpClass(cls):
        logger = logging.getLogger('skl2onnx')
        logger.setLevel(logging.WARNING)
        logger = logging.getLogger('onnxcustom')
        logger.setLevel(logging.WARNING)
        logging.basicConfig(level=logging.WARNING)

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_ort_gradient_optimizers_binary(self):
        self.wtest_ort_gradient_optimizers_binary(False)

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_ort_gradient_optimizers_binary_weighted(self):
        self.wtest_ort_gradient_optimizers_binary(True)

    def wtest_ort_gradient_optimizers_binary(self, use_weight=False):
        from onnxcustom.utils.orttraining_helper import add_loss_output
        from onnxcustom.training.optimizers import OrtGradientOptimizer
        X, y = make_classification(  # pylint: disable=W0632
            100, n_features=10, random_state=0)
        X = X.astype(numpy.float32)
        y = y.astype(numpy.int64)
        w = (numpy.random.rand(X.shape[0]) + 1).astype(numpy.float32)
        X_train, _, y_train, __, w_train, ___ = train_test_split(X, y, w)
        reg = SGDClassifier(loss='log')
        reg.fit(X_train, y_train)
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearClassifier'},
                      options={'zipmap': False})
        set_model_props(onx, {'info': 'unit test'})
        onx_loss = add_loss_output(
            onx, 'log', output_index=1,
            weight_name='weight' if use_weight else None)
        inits = ['intercept', 'coef']
        inputs = onx_loss.graph.input
        self.assertEqual(len(inputs), 3 if use_weight else 2)
        dt = inputs[1].type.tensor_type.elem_type
        self.assertEqual(TensorProto.INT64, dt)  # pylint: disable=E1101
        train_session = OrtGradientOptimizer(
            onx_loss, inits, learning_rate=1e-3)
        self.assertRaise(lambda: train_session.get_state(), AttributeError)
        if use_weight:
            train_session.fit(
                X_train, y_train.reshape((-1, 1)),
                w_train.reshape((-1, 1)), use_numpy=False)
        else:
            train_session.fit(
                X_train, y_train.reshape((-1, 1)), use_numpy=False)
        state_tensors = train_session.get_state()
        self.assertEqual(len(state_tensors), 2)
        r = repr(train_session)
        self.assertIn("OrtGradientOptimizer(model_onnx=", r)
        self.assertIn("learning_rate='invscaling'", r)
        losses = train_session.train_losses_
        self.assertGreater(len(losses), 1)
        self.assertFalse(any(map(numpy.isnan, losses)))

        # state
        state = train_session.get_state()
        self.assertIsInstance(state, dict)
        train_session.set_state(state)

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_ort_gradient_optimizers_fw_nesterov_binary(self):
        self.wtest_ort_gradient_optimizers_fw_nesterov_binary(False)

    @unittest.skipIf(TrainingSession is None, reason="not training")
    def test_ort_gradient_optimizers_fw_nesterov_binary_weight(self):
        self.wtest_ort_gradient_optimizers_fw_nesterov_binary(True)

    def wtest_ort_gradient_optimizers_fw_nesterov_binary(self, use_weight):
        from onnxcustom.training.optimizers_partial import (
            OrtGradientForwardBackwardOptimizer)
        from onnxcustom.training.sgd_learning_rate import (
            LearningRateSGDNesterov)
        from onnxcustom.training.sgd_learning_loss import NegLogLearningLoss
        X, y = make_classification(  # pylint: disable=W0632
            100, n_features=10, random_state=0)
        X = X.astype(numpy.float32)
        y = y.astype(numpy.int64)
        w = (numpy.random.rand(y.shape[0]) + 1).astype(numpy.float32)
        X_train, _, y_train, __, w_train, ___ = train_test_split(X, y, w)
        reg = SGDClassifier(loss='log')
        if use_weight:
            reg.fit(X_train, y_train,
                    sample_weight=w_train.astype(numpy.float64))
        else:
            reg.fit(X_train, y_train)
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'},
                      options={'zipmap': False,
                               'raw_scores': True})
        onx = select_model_inputs_outputs(onx, outputs=['score'])
        self.assertIn("output: name='score'",
                      onnx_simple_text_plot(onx))
        set_model_props(onx, {'info': 'unit test'})
        inits = ['coef', 'intercept']

        train_session = OrtGradientForwardBackwardOptimizer(
            onx, inits, weight_name='weight' if use_weight else None,
            learning_rate=LearningRateSGDNesterov(
                1e-4, nesterov=False, momentum=0.9),
            learning_loss=NegLogLearningLoss(),
            warm_start=False, max_iter=100, batch_size=10)
        self.assertIsInstance(train_session.learning_loss, NegLogLearningLoss)
        self.assertEqual(train_session.learning_loss.eps, 1e-5)
        y_train = y_train.reshape((-1, 1))
        if use_weight:
            train_session.fit(X_train, y_train, w_train.reshape((-1, 1)))
        else:
            train_session.fit(X_train, y_train)
        temp = get_temp_folder(
            __file__, "temp_ort_gradient_optimizers_fw_nesterov_binary")
        train_session.save_onnx_graph(temp)

        # state
        state = train_session.get_state()
        self.assertIsInstance(state, list)
        train_session.set_state(state)
        device = C_OrtDevice(
            C_OrtDevice.cpu(), C_OrtDevice.default_memory(), 0)
        for k in range(len(state)):
            state[k] = state[k].numpy()
        train_session.set_state(state)

    @unittest.skipIf(TrainingSession is None, reason="not training")
    @ignore_warnings(ConvergenceWarning)
    def test_ort_gradient_optimizers_fw_nesterov_binary_mlp(self):
        self.wtest_ort_gradient_optimizers_fw_nesterov_binary_mlp(False)

    @unittest.skipIf(TrainingSession is None, reason="not training")
    @ignore_warnings(ConvergenceWarning)
    def test_ort_gradient_optimizers_fw_nesterov_binary_mlp_weight(self):
        self.wtest_ort_gradient_optimizers_fw_nesterov_binary_mlp(True)

    def wtest_ort_gradient_optimizers_fw_nesterov_binary_mlp(
            self, use_weight=True):
        from onnxcustom.training.optimizers_partial import (
            OrtGradientForwardBackwardOptimizer)
        from onnxcustom.training.sgd_learning_rate import (
            LearningRateSGDNesterov)
        from onnxcustom.training.sgd_learning_loss import NegLogLearningLoss
        X, y = make_classification(  # pylint: disable=W0632
            100, n_features=10, random_state=0)
        X = X.astype(numpy.float32)
        y = y.astype(numpy.int64)
        w = (numpy.random.rand(y.shape[0]) + 1).astype(numpy.float32)
        X_train, _, y_train, __, w_train, ___ = train_test_split(X, y, w)
        reg = MLPClassifier(solver='sgd')
        reg.fit(X_train, y_train)
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'},
                      options={'zipmap': False})
        onx = select_model_inputs_outputs(
            onx, outputs=['out_activations_result'])
        self.assertIn("output: name='out_activations_result'",
                      onnx_simple_text_plot(onx))
        set_model_props(onx, {'info': 'unit test'})
        onx = onnx_rename_weights(onx)
        inits = ['I0_coefficient', 'I1_intercepts',
                 'I2_coefficient1', 'I3_intercepts1']

        train_session = OrtGradientForwardBackwardOptimizer(
            onx, inits, weight_name='weight' if use_weight else None,
            learning_rate=LearningRateSGDNesterov(
                1e-4, nesterov=False, momentum=0.9),
            learning_loss=NegLogLearningLoss(),
            warm_start=False, max_iter=100, batch_size=10)
        self.assertIsInstance(train_session.learning_loss, NegLogLearningLoss)
        self.assertEqual(train_session.learning_loss.eps, 1e-5)
        if use_weight:
            train_session.fit(X_train, y_train, w_train)
        else:
            train_session.fit(X_train, y_train)
        temp = get_temp_folder(
            __file__, "temp_ort_gradient_optimizers_fw_nesterov_binary_mlp%d" % use_weight)
        train_session.save_onnx_graph(temp)


if __name__ == "__main__":

    unittest.main()
