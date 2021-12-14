"""
@file
@brief Optimizer with :epkg:`onnxruntime-training` forward backward training.
"""
import logging
import numpy
from onnxruntime import InferenceSession, OrtValue
from onnxruntime.capi._pybind_state import (  # pylint: disable=E0611
    OrtValue as C_OrtValue)
from ..utils.onnx_helper import get_onnx_opset, proto_type_to_dtype
from ..utils.onnxruntime_helper import device_to_provider
from ..utils.onnx_function import function_onnx_graph
from ..utils.print_helper import str_ortvalue
from ..utils.onnx_orttraining import get_train_initializer
from .ortgradient import OrtGradientForwardBackward
from .optimizers import BaseEstimator
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
    :param device: `'cpu'` or `'cuda'`
    :param device_index: device index
    :param warm_start: when set to True, reuse the solution of the previous
        call to fit as initialization, otherwise, just erase the previous
        solution.
    :param loss_function: loss function (see below)
    :param verbose: use :epkg:`tqdm` to display the training progress
    :param validation_every: validation with a test set every
        *validation_every* iterations
    :param enable_logging: enable logging (mostly for debugging puporse
        as it slows down the training)

    *loss_function* can be:
    * `square_error`: mean square error, used for regression
    """

    def __init__(self, model_onnx, weights_to_train=None,
                 loss_output_name='loss', max_iter=100,
                 training_optimizer_name='SGDOptimizer',
                 batch_size=10, learning_rate='SGDRegressor',
                 device='cpu', device_index=0,
                 warm_start=False, verbose=0, validation_every=0.1,
                 loss_function="square_error",
                 enable_logging=False):
        if weights_to_train is None:
            weights_to_train = list(get_train_initializer(model_onnx))
        BaseEstimator.__init__(self, learning_rate)
        self.model_onnx = model_onnx
        self.batch_size = batch_size
        self.weights_to_train = weights_to_train
        self.loss_output_name = loss_output_name
        self.training_optimizer_name = training_optimizer_name
        self.verbose = verbose
        self.max_iter = max_iter
        self.device = device
        self.device_index = device_index
        self.warm_start = warm_start
        self.loss_function = loss_function
        self.enable_logging = enable_logging
        if validation_every < 1:
            self.validation_every = int(self.max_iter * validation_every)
        else:
            self.validation_every = validation_every  # pragma: no cover
        self._build_loss_function()

    def __getstate__(self):
        "Removes any non pickable attribute."
        atts = [k for k in self.__dict__ if not k.endswith('_')]
        state = {att: getattr(self, att) for att in atts}
        if hasattr(self, 'train_state_'):
            train_state = []
            for v in self.get_state():
                if v is None:
                    train_state.append(v)
                else:
                    train_state.append(v.numpy())
            state['train_state'] = train_state
        return state

    def __setstate__(self, state):
        "Restores any non pickable attribute."
        if 'train_state' in state:
            train_state = state.pop('train_state')
        else:
            train_state = None
        for att, v in state.items():
            setattr(self, att, v)
        if train_state is not None:
            self.set_state(train_state, check_trained=False)
        self._build_loss_function()
        return self

    def get_full_state(self):
        """
        Returns the trained weights.
        """
        if not hasattr(self, 'train_state_'):
            raise AttributeError(  # pragma: no cover
                "Method fit must be called before.")
        return self.train_state_

    def get_state(self):
        """
        Returns the trained weights.
        """
        if not hasattr(self, 'train_state_'):
            raise AttributeError("Method fit must be called before.")
        if self.train_state_ is None:
            raise RuntimeError("No train_state_ available (None).")
        if self.weights_to_train is None:
            raise RuntimeError("Unexpected self.weights_to_train (None).")
        n = len(self.train_state_) - len(self.weights_to_train)
        return self.train_state_[n:]

    def set_state(self, state, check_trained=True):
        """
        Changes the trained weights.
        """
        if check_trained and not hasattr(self, 'train_session_'):
            raise AttributeError(  # pragma: no cover
                "Method fit must be called before.")
        self.train_state_ = []
        self.train_state_numpy_ = []
        for i, v in enumerate(state):
            if v is None:
                self.train_state_.append(None)
                self.train_state_numpy_.append(None)
            elif isinstance(v, numpy.ndarray):
                ortvalue = OrtValue.ortvalue_from_numpy(
                    v, self.device, self.device_index)._ortvalue
                self.train_state_.append(ortvalue)
                # The numpy container must be retained as the ortvalue
                # just borrows the pointer.
                self.train_state_numpy_.append(v)
            elif isinstance(v, C_OrtValue):
                self.train_state_.append(v)
                self.train_state_numpy_.append(None)
            else:
                raise TypeError(  # pragma: no cover
                    "Unexpected type %r for state %r." % (
                        type(v), i))

    def _build_loss_function(self):
        opset = get_onnx_opset(self.model_onnx)
        self.loss_grad_onnx_ = function_onnx_graph(
            "grad_loss_" + self.loss_function, target_opset=opset)
        self.loss_grad_sess_ = InferenceSession(
            self.loss_grad_onnx_.SerializeToString())
        self.loss_grad_sess_bind_ = self.loss_grad_sess_.io_binding()

        self.axpy_onnx_ = function_onnx_graph("axpy")
        self.axpy_sess_ = InferenceSession(self.axpy_onnx_.SerializeToString())
        self.axpy_sess_bind_ = self.axpy_sess_.io_binding()

        if self.enable_logging:
            self._logger = logging.getLogger("onnxcustom")
        else:
            self._logger = None

    def fit(self, X, y, X_val=None, y_val=None, use_numpy=False):
        """
        Trains the model.

        :param X: features
        :param y: expected output
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

        data_loader = OrtDataLoader(
            X, y, batch_size=self.batch_size, device=self.device)
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
        lr = self.learning_rate.value
        for it in loop:
            loss = self._iteration(data_loader, lr, self.get_full_state(),
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

    def _bind_input_ortvalue(self, name, bind, c_ortvalue):
        """
        Binds :epkg:`C_OrtValue` to the structure used by
        :epkg:`InferenceSession` to run inference.

        :param name: str
        :param bind: python structure
        :param c_ortvalue: C structure for OrtValue (:epkg:`C_OrtValue`),
            it can be also a numpy array
        """
        if isinstance(c_ortvalue, C_OrtValue):
            # does not work
            # bind._iobinding.bind_ortvalue_input(name, c_ortvalue)
            if hasattr(c_ortvalue, 'proto_type'):
                dtype = proto_type_to_dtype(c_ortvalue.proto_type())
            else:
                dtype = proto_type_to_dtype(c_ortvalue.data_type())
            bind.bind_input(
                name=name, device_type=self.device,
                device_id=self.device_index,
                element_type=dtype,
                shape=c_ortvalue.shape(),
                buffer_ptr=c_ortvalue.data_ptr())
        elif isinstance(c_ortvalue, numpy.ndarray):
            bind.bind_input(
                name, device_type=self.device,
                device_id=self.device_index,
                element_type=c_ortvalue.dtype,
                shape=c_ortvalue.shape,
                buffer_ptr=c_ortvalue.__array_interface__['data'][0])
        else:
            raise TypeError(  # pragma: no cover
                "Unable to bind type %r for name %r." % (
                    type(c_ortvalue), name))

    def _bind_output_ortvalue(self, name, bind, c_ortvalue):
        """
        Binds :epkg:`C_OrtValue` to the structure used by
        :epkg:`InferenceSession` to run inference.

        :param name: str
        :param bind: python structure
        :param c_ortvalue: C structure for OrtValue (:epkg:`C_OrtValue`)

        This method can be used for inplace computation.
        """
        if isinstance(c_ortvalue, C_OrtValue):
            # does not work
            # bind._iobinding.bind_ortvalue_input(name, c_ortvalue)
            if hasattr(c_ortvalue, 'proto_type'):
                dtype = proto_type_to_dtype(c_ortvalue.proto_type())
            else:
                dtype = proto_type_to_dtype(c_ortvalue.data_type())
            bind.bind_output(
                name=name, device_type=self.device,
                device_id=self.device_index,
                element_type=dtype,
                shape=c_ortvalue.shape(),
                buffer_ptr=c_ortvalue.data_ptr())
        else:
            raise TypeError(  # pragma: no cover
                "Unable to bind type %r for name %r." % (
                    type(c_ortvalue), name))

    def _loss_gradient(self, expected, predicted):
        """
        Returns the loss and the gradient as OrtValue.
        """
        self._bind_input_ortvalue("X1", self.loss_grad_sess_bind_, expected)
        self._bind_input_ortvalue("X2", self.loss_grad_sess_bind_, predicted)
        self.loss_grad_sess_bind_.bind_output('Y')
        self.loss_grad_sess_bind_.bind_output('Z')
        self.loss_grad_sess_.run_with_iobinding(self.loss_grad_sess_bind_)
        loss, grad = self.loss_grad_sess_bind_._iobinding.get_outputs()
        return loss, grad

    def _update_weights(self, statei, gradienti, alpha):
        self._bind_input_ortvalue("X1", self.axpy_sess_bind_, gradienti)
        self._bind_input_ortvalue("X2", self.axpy_sess_bind_, statei)
        alpha_alive = numpy.array([alpha], dtype=numpy.float32)
        self._bind_input_ortvalue(
            "alpha", self.axpy_sess_bind_, alpha_alive)
        self._bind_output_ortvalue('Y', self.axpy_sess_bind_, statei)
        self.axpy_sess_.run_with_iobinding(self.axpy_sess_bind_)
        return self.axpy_sess_bind_._iobinding.get_outputs()[0]

    def _iteration(self, data_loader, learning_rate, state, n_weights):
        actual_losses = []
        bs = data_loader.batch_size
        logger = self._logger

        if logger is not None:
            logger.debug(
                "[OrtGradientForwardBackwardOptimizer._iteration] "
                "iteration begin learning_rate=%f", learning_rate)

        for ib, (ortx, orty) in enumerate(data_loader.iter_ortvalue()):
            state[0] = ortx

            if logger is not None:
                logger.debug(
                    "[OrtGradientForwardBackwardOptimizer._iteration] "
                    "batch %d", ib)

            prediction = self.train_function_.forward(state, training=True)
            loss, loss_gradient = self._loss_gradient(orty, prediction[0])
            cpu_loss = loss.numpy()
            if numpy.isinf(cpu_loss) or numpy.isnan(cpu_loss):
                raise ConvergenceError(
                    "Loss is nan, learning_rate=%r, "
                    "the gradient descent has failed "
                    "(past losses=%r)." % (
                        learning_rate,
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
                self._update_weights(
                    state[i], gradient[i], -learning_rate / bs)

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
            loss, _ = self._loss_gradient(orty, prediction[0])
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
            providers=[device_to_provider(device)])
        inst = forback.new_instance()
        return (forback, inst)
