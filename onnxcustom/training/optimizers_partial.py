"""
@file
@brief Optimizer with :epkg:`onnxruntime-training` forward backward training.
"""
import logging
import numpy
from onnxruntime import InferenceSession, SessionOptions
from onnxruntime.capi._pybind_state import (  # pylint: disable=E0611
    OrtValue as C_OrtValue)
from ..utils.onnx_helper import get_onnx_opset, proto_type_to_dtype
from ..utils.onnxruntime_helper import (
    device_to_providers, numpy_to_ort_value)
from ..utils.onnx_function import function_onnx_graph
from ..utils.print_helper import str_ortvalue
from ..utils.orttraining_helper import get_train_initializer
from .ortgradient import OrtGradientForwardBackward
from .base_estimator import BaseEstimator
from .sgd_learning_loss import BaseLearningLoss
from .sgd_learning_penalty import BaseLearningPenalty
from .data_loader import OrtDataLoader
from .excs import ConvergenceError


class OrtGradientForwardBackwardOptimizer(BaseEstimator):
    """
    Implements a simple :epkg:`Stochastic Gradient Descent`
    with :epkg:`onnxruntime-training`. It leverages class
    @see class OrtGradientForwardBackward.

    :param model_onnx: ONNX graph used to train
    :param weights_to_train: names of initializers to be optimized,
        if None, function @see fn get_train_initialize returns
        the list of float iniitializer
    :param loss_output_name: name of the loss output
    :param max_iter: number of training iterations
    :param training_optimizer_name: optimizing algorithm
    :param batch_size: batch size (see class *DataLoader*)
    :param learning_rate: a name or a learning rate instance or a float,
        see module :mod:`onnxcustom.training.sgd_learning_rate`
    :param device: device as :epkg:`C_OrtDevice` or a string
        representing this device
    :param warm_start: when set to True, reuse the solution of the previous
        call to fit as initialization, otherwise, just erase the previous
        solution.
    :param learning_loss: loss function (see below)
    :param verbose: use :epkg:`tqdm` to display the training progress
    :param validation_every: validation with a test set every
        *validation_every* iterations
    :param enable_logging: enable logging (mostly for debugging puporse
        as it slows down the training)
    :param weight_name: if not None, the class assumes it is trained
        with training weight
    :param learning_penalty: weight penalty, None, or instance of
        @see cl BaseLearningPenalty

    *learning_rate* can be any instance of @see cl BaseLearningRate or
    a nick name in the following list as specified in
    :meth:`BaseLearningRate.select
    <onnxcustom.training.sgd_learning_loss.BaseLearningRate.select>`.

    *learning_loss* can be any instance of @see cl BaseLearningLoss or
    a nick name in the following list as specified in
    :meth:`BaseLearningLoss.select
    <onnxcustom.training.sgd_loss.BaseLearningLoss.select>`.
    """

    def __init__(self, model_onnx, weights_to_train=None,
                 loss_output_name='loss', max_iter=100,
                 training_optimizer_name='SGDOptimizer',
                 batch_size=10, learning_rate='SGD',
                 device='cpu', warm_start=False, verbose=0,
                 validation_every=0.1, learning_loss="square_error",
                 enable_logging=False, weight_name=None,
                 learning_penalty=None):
        if weights_to_train is None:
            weights_to_train = list(get_train_initializer(model_onnx))
        BaseEstimator.__init__(self, learning_rate, device)
        self.model_onnx = model_onnx
        self.batch_size = batch_size
        self.weights_to_train = weights_to_train
        self.loss_output_name = loss_output_name
        self.training_optimizer_name = training_optimizer_name
        self.verbose = verbose
        self.max_iter = max_iter
        self.warm_start = warm_start
        self.learning_loss = BaseLearningLoss.select(learning_loss)
        self.learning_penalty = BaseLearningPenalty.select(learning_penalty)
        self.enable_logging = enable_logging
        self.weight_name = weight_name
        if validation_every < 1:
            self.validation_every = int(self.max_iter * validation_every)
        else:
            self.validation_every = validation_every  # pragma: no cover
        self.build_onnx_function()

    @property
    def needs_grad(self):
        """
        Returns the True if the gradient update needs to retain
        past gradients.
        """
        return self.learning_rate.needs_grad

    def __getstate__(self):
        "Removes any non pickable attribute."
        state = BaseEstimator.__getstate__(self)
        for att in ['train_state_', 'train_grad_state_']:
            if hasattr(self, att):
                train_state = []
                for v in self.get_state():
                    if v is None:
                        train_state.append(v)
                    else:
                        train_state.append(v.numpy())
                state[att[:-1]] = train_state
        return state

    def __setstate__(self, state):
        "Restores any non pickable attribute."
        popped = {}
        for att in ['train_state', 'train_grad_state']:
            if att in state:
                popped[att] = state.pop(att)
        BaseEstimator.__setstate__(self, state)
        for k, v in popped.items():
            if k == 'train_state':
                self.set_state(v, check_trained=False, kind='weight')
            elif k == 'train_grad_state':
                self.set_state(v, check_trained=False, kind='grad')
            else:
                raise ValueError(  # pragma: no cover
                    "Unexpected key state %r." % k)
        self.build_onnx_function()
        return self

    def _get_att_state(self, kind):
        if kind == 'weight':
            return 'train_state_'
        if kind == 'grad':
            return 'train_grad_state_'
        raise ValueError(  # pragma: no cover
            "Unexpected kind=%r." % kind)

    def get_full_state(self, kind='weight'):
        """
        Returns the trained weights.
        """
        if isinstance(kind, list):
            return [self.get_full_state(kind=k) for k in kind]
        att = self._get_att_state(kind)
        if not hasattr(self, att):
            raise AttributeError(  # pragma: no cover
                "Method fit must be called before.")
        return getattr(self, att)

    def get_state(self, kind='weight'):
        """
        Returns the trained weights.
        """
        att = self._get_att_state(kind)
        if not hasattr(self, att):
            raise AttributeError("Method fit must be called before.")
        if getattr(self, att, None) is None:
            raise RuntimeError(  # pragma: no cover
                "No attribute %r available (None)." % att)
        if self.weights_to_train is None:
            raise RuntimeError(  # pragma: no cover
                "Unexpected self.weights_to_train (None).")
        value = getattr(self, att)
        n = len(value) - len(self.weights_to_train)
        return value[n:]

    def set_state(self, state, check_trained=True, kind='weight', zero=False):
        """
        Changes the trained weights.
        """
        if check_trained and not hasattr(self, 'train_session_'):
            raise AttributeError(  # pragma: no cover
                "Method fit must be called before.")
        state_ = []
        state_numpy_ = []
        for i, v in enumerate(state):
            if v is None:
                state_.append(None)
                state_numpy_.append(None)
            elif isinstance(v, numpy.ndarray):
                if zero:
                    v = numpy.zeros(v.shape, dtype=v.dtype)
                ortvalue = numpy_to_ort_value(v, self.device)
                state_.append(ortvalue)
                # The numpy container must be retained as the ortvalue
                # just borrows the pointer.
                state_numpy_.append(v)
            elif isinstance(v, C_OrtValue):
                if zero:
                    v = self.zero_sess_.run_with_ort_values(['Y'], {'X': v})
                state_.append(v)
                state_numpy_.append(None)
            else:
                raise TypeError(  # pragma: no cover
                    "Unexpected type %r for state %r." % (
                        type(v), i))
        att = self._get_att_state(kind)
        setattr(self, att, state_)
        setattr(self, att + "numpy_", state_numpy_)

    def build_onnx_function(self):
        """
        Creates ONNX graph and *InferenceSession* related to
        any operations applying on *OrtValue*.
        """
        opset = get_onnx_opset(self.model_onnx)
        so = SessionOptions()
        so.log_severity_level = 4

        # loss_grad
        self.learning_loss.build_onnx_function(
            opset, self.device, self.weight_name)

        # weight update
        self.learning_rate.build_onnx_function(opset, self.device)

        # penalty
        n = len(self.weights_to_train)
        self.learning_penalty.build_onnx_function(opset, self.device, n)

        # zero
        self.zero_onnx_ = function_onnx_graph("zero")
        self.zero_sess_ = InferenceSession(
            self.zero_onnx_.SerializeToString(), so,
            providers=device_to_providers(self.device))

        # logging
        if self.enable_logging:
            self._logger = logging.getLogger("onnxcustom")
        else:
            self._logger = None

    def fit(self, X, y, sample_weight=None,
            X_val=None, y_val=None, use_numpy=False):
        """
        Trains the model.

        :param X: features
        :param y: expected output
        :param sample_weight: training weight or None
        :param X_val: evaluation dataset
        :param y_val: evaluation dataset
        :param use_numpy: if True, slow iterator using numpy,
            otherwise, minimizes copy
        :return: self
        """
        if self.training_optimizer_name != 'SGDOptimizer':
            raise NotImplementedError(
                "Only the SGDOptimizer is implemented not %r."
                "" % self.training_optimizer_name)
        logger = self._logger

        session_function = self._create_training_session(
            self.model_onnx, self.weights_to_train,
            device=self.device)
        self.train_session_ = session_function[0]
        self.train_function_ = session_function[1]

        self.input_names_ = self.train_session_.cls_type_._grad_input_names
        self.output_names_ = self.train_session_.cls_type_._bw_fetches_names
        weights_to_train = self.train_session_.weights_to_train

        if logger is not None:
            logger.info(
                "[OrtGradientForwardBackwardOptimizer.fit] "
                "input_names=%r", self.input_names_)
            logger.info(
                "[OrtGradientForwardBackwardOptimizer.fit] "
                "output_names=%r", self.output_names_)
            logger.info(
                "[OrtGradientForwardBackwardOptimizer.fit] "
                "weights_to_train=%r", self.weights_to_train)
            logger.info(
                "[OrtGradientForwardBackwardOptimizer.fit] "
                "device=%r", self.device)
            if logger is not None:
                logger.info(
                    "[OrtGradientForwardBackwardOptimizer.fit] "
                    "warm_start=%r", self.warm_start)

        if not hasattr(self, 'state_'):
            self.set_state([
                self.train_session_.get_initializer(name, exc=False)
                for name in self.input_names_])
        if self.needs_grad and not hasattr(self, 'state_grad_'):
            self.set_state([
                self.train_session_.get_initializer(name, exc=False)
                for name in self.input_names_],
                kind='grad', zero=True)
        if not self.warm_start:
            state = self.get_full_state()
            if len(state) != len(self.input_names_):
                raise RuntimeError(  # pragma: no cover
                    "Length mismatch %r != %r." % (
                        len(state), len(self.input_names_)))
            new_state = []
            for iv, v in enumerate(state):
                if v is None:
                    new_state.append(v)
                else:
                    if not isinstance(v, C_OrtValue):
                        raise RuntimeError(  # pragma: no cover
                            "Unexpected type %r (state[%d])." % (
                                type(v), iv))
                    dtype = proto_type_to_dtype(
                        v.proto_type()
                        if hasattr(v, 'proto_type')
                        else v.data_type())
                    if len(v.shape()) > 0:
                        new_state.append(
                            numpy.random.randn(*v.shape()).astype(dtype))
                    else:
                        new_state.append(
                            numpy.random.randn(1).astype(dtype))
            self.set_state(new_state)
            if self.needs_grad:
                self.set_state(new_state, kind='grad', zero=True)

        data_loader = OrtDataLoader(
            X, y, sample_weight, batch_size=self.batch_size,
            device=self.device)
        if X_val is not None:
            data_loader_val = OrtDataLoader(
                X_val, y_val, batch_size=X_val.shape[0], device=self.device,
                random_iter=False)
        else:
            data_loader_val = None

        self.learning_rate.init_learning_rate()

        if self.verbose > 0:  # pragma: no cover
            from tqdm import tqdm  # pylint: disable=C0415
            loop = tqdm(range(self.max_iter))
        else:
            loop = range(self.max_iter)

        train_losses = []
        val_losses = []
        kinds = ['weight', 'grad'] if self.needs_grad else ['weight']
        for it in loop:
            loss = self._iteration(
                data_loader, self.get_full_state(kind=kinds),
                len(weights_to_train))
            lr = self.learning_rate.update_learning_rate(it).value
            if self.verbose > 1:  # pragma: no cover
                loop.set_description(
                    "loss=%1.3g lr=%1.3g" % (  # pylint: disable=E1101,E1307
                        loss, lr))  # pylint: disable=E1101,E1307
            train_losses.append(loss)
            if (data_loader_val is not None and
                    (it + 1) % self.validation_every == 0):
                val_losses.append(
                    self._evaluation(data_loader_val, self.get_full_state()))
        self.train_losses_ = train_losses
        self.validation_losses_ = (
            None if data_loader_val is None else val_losses)

        if logger is not None:
            logger.info(
                "[OrtGradientForwardBackwardOptimizer.fit] "
                "end loss=%r", self.train_losses_[-1])
        return self

    def _iteration(self, data_loader, states, n_weights):
        actual_losses = []
        bs = data_loader.batch_size
        logger = self._logger
        if len(states) == 1:
            state = states[0]
            grad = None
        else:
            state, grad = states

        if logger is not None:
            logger.debug(
                "[OrtGradientForwardBackwardOptimizer._iteration] "
                "iteration begin learning_rate=%r",
                self.learning_rate)

        for ib, ito in enumerate(data_loader.iter_ortvalue()):
            if len(ito) == 2:
                (ortx, orty) = ito
                ortw = None
            else:
                (ortx, orty, ortw) = ito
            state[0] = ortx

            if logger is not None:
                logger.debug(
                    "[OrtGradientForwardBackwardOptimizer._iteration] "
                    "batch %d", ib)

            prediction = self.train_function_.forward(states[0], training=True)
            loss, loss_gradient = self.learning_loss.loss_gradient(
                self.device, orty, prediction[0], weight=ortw)
            n = len(state) - n_weights
            loss = self.learning_penalty.penalty_loss(
                self.device, loss, *state[n:])
            cpu_loss = loss.numpy()
            if numpy.isinf(cpu_loss) or numpy.isnan(cpu_loss):
                raise ConvergenceError(
                    "Loss is nan, learning_rate=%r, "
                    "the gradient descent has failed "
                    "(past losses=%r)." % (
                        self.learning_rate,
                        [float(v) for v in (
                            actual_losses if len(actual_losses) < 5
                            else actual_losses[-5:])]))

            gradient = self.train_function_.backward([loss_gradient])

            if len(gradient) != len(state):
                raise RuntimeError(  # pragma: no cover
                    "gradient and state should have the same length but "
                    "%r != %r." % (len(gradient), len(state)))

            n = len(state) - n_weights
            for i in range(n, len(state)):
                self.learning_penalty.update_weights(self.device, state[i])
                self.learning_rate.update_weights(
                    self.device, state[i], gradient[i], bs,
                    None if grad is None else grad[i])

            if logger is not None:
                logger.debug(
                    "[OrtGradientForwardBackwardOptimizer._iteration] "
                    "loss=%g", cpu_loss)
                for i in range(n, len(state)):
                    logger.debug(
                        "[OrtGradientForwardBackwardOptimizer._iteration] "
                        "state[%i]=%s", i, str_ortvalue(state[i]))

            actual_losses.append(cpu_loss / bs)

        if logger is not None:
            logger.debug(
                "[OrtGradientForwardBackwardOptimizer._iteration] "
                "iteration end")

        return numpy.array(actual_losses).mean()

    def _evaluation(self, data_loader, state):
        bs = data_loader.batch_size
        logger = self._logger
        actual_losses = []
        for ib, (ortx, orty) in enumerate(data_loader.iter_ortvalue()):
            state[0] = ortx

            if logger is not None:
                logger.debug(
                    "[OrtGradientForwardBackwardOptimizer._evaluation] "
                    "batch %d", ib)

            prediction = self.train_function_.forward(state, training=False)
            loss, _ = self.learning_loss.loss_gradient(
                self.device, orty, prediction[0])
            cpu_loss = loss.numpy()
            if numpy.isinf(cpu_loss) or numpy.isnan(cpu_loss):
                raise ConvergenceError(
                    "Loss is nan, "
                    "the evaluation has failed "
                    "(past losses=%r)." %
                    [float(v) for v in (
                        actual_losses if len(actual_losses) < 5
                        else actual_losses[-5:])])
            actual_losses.append(cpu_loss / bs)

        return numpy.array(actual_losses).sum() / len(data_loader)

    def _create_training_session(
            self, model_onnx, weights_to_train, device):

        forback = OrtGradientForwardBackward(
            model_onnx, weights_to_train=weights_to_train,
            debug=False, enable_logging=False,
            providers=device_to_providers(device))
        inst = forback.new_instance()
        return (forback, inst)
