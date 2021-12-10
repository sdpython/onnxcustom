"""
@file
@brief Optimizer with :epkg:`onnxruntime-training`.
"""
import inspect
import numpy
from onnxruntime import (  # pylint: disable=E0611
    OrtValue as PyOrtValue, TrainingParameters,
    SessionOptions, TrainingSession)
from onnxruntime.capi._pybind_state import (  # pylint: disable=E0611
    OrtValue as C_OrtValue)
from ..utils.onnx_helper import proto_type_to_dtype
from .data_loader import OrtDataLoader
from .sgd_learning_rate import BaseLearningRate
from .excs import ConvergenceError, EvaluationError


class BaseEstimator:
    """
    Base class for optimizers.
    Implements common methods such `__repr__`.

    :param learning_rate: learning rate class,
        see module :mod:`onnxcustom.training.sgd_learning_rate`
    """

    def __init__(self, learning_rate):
        self.learning_rate = BaseLearningRate.select(learning_rate)

    @classmethod
    def _get_param_names(cls):
        "Extracts all parameters to serialize."
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        init_signature = inspect.signature(init)
        parameters = [
            p for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD]
        return [(p.name, p.default) for p in parameters]

    def __repr__(self):
        "Usual."
        param = self._get_param_names()
        ps = []
        for k, v in param:
            if k not in self.__dict__:
                continue  # pragma: no cover
            ov = getattr(self, k)
            if isinstance(ov, BaseLearningRate):
                ps.append("%s=%s" % (k, repr(ov)))
            elif v is not inspect._empty or ov != v:
                ro = repr(ov)
                if len(ro) > 50 or "\n" in ro:
                    ro = ro[:10].replace("\n", " ") + "..."
                    ps.append("%s=%r" % (k, ro))
                else:
                    ps.append("%s=%r" % (k, ov))
        return "%s(%s)" % (self.__class__.__name__, ", ".join(ps))


class OrtGradientOptimizer(BaseEstimator):
    """
    Implements a simple :epkg:`Stochastic Gradient Descent`
    with :epkg:`onnxruntime-training`.

    :param model_onnx: ONNX graph used to train
    :param weights_to_train: names of initializers to be optimized
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
    :param verbose: use :epkg:`tqdm` to display the training progress
    :param validation_every: validation with a test set every
        *validation_every* iterations

    Once initialized, the class creates the attribute
    `train_session_` which holds an instance of :ref:`l-ort-training-session`.

    See example :ref:`l-orttraining-nn-gpu`.
    """

    def __init__(self, model_onnx, weights_to_train, loss_output_name='loss',
                 max_iter=100, training_optimizer_name='SGDOptimizer',
                 batch_size=10, learning_rate='SGDRegressor',
                 device='cpu', device_index=0,
                 warm_start=False, verbose=0, validation_every=0.1):
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
        if validation_every < 1:
            self.validation_every = int(self.max_iter * validation_every)
        else:
            self.validation_every = validation_every  # pragma: no cover

    def __getstate__(self):
        "Removes any non pickable attribute."
        atts = [k for k in self.__dict__ if not k.endswith('_')]
        if hasattr(self, 'trained_coef_'):
            atts.append('trained_coef_')
        return {att: getattr(self, att) for att in atts}

    def __setstate__(self, state):
        "Restores any non pickable attribute."
        for att, v in state.items():
            setattr(self, att, v)
        return self

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
        self.train_session_ = self._create_training_session(
            self.model_onnx, self.weights_to_train,
            loss_output_name=self.loss_output_name,
            training_optimizer_name=self.training_optimizer_name,
            device=self.device)

        if not self.warm_start:
            state = self.get_state()
            new_state = {}
            for k, v in state.items():
                if len(v.shape) > 0:
                    new_state[k] = numpy.random.randn(*v.shape).astype(v.dtype)
                else:
                    f = numpy.random.randn(1)
                    f = f.astype(v.dtype)
                    new_state[k] = f
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
        self.input_names_ = [i.name for i in self.train_session_.get_inputs()]
        self.output_names_ = [
            o.name for o in self.train_session_.get_outputs()]
        self.loss_index_ = self.output_names_.index(self.loss_output_name)

        bind = self.train_session_.io_binding()

        if self.verbose > 0:  # pragma: no cover
            from tqdm import tqdm  # pylint: disable=C0415
            loop = tqdm(range(self.max_iter))
        else:
            loop = range(self.max_iter)

        train_losses = []
        val_losses = []
        lr = self.learning_rate.value
        for it in loop:
            lr_alive = numpy.array([lr / self.batch_size], dtype=numpy.float32)
            ort_lr = PyOrtValue.ortvalue_from_numpy(
                lr_alive, self.device, self.device_index)._ortvalue
            loss = self._iteration(data_loader, ort_lr,
                                   bind, use_numpy=use_numpy)
            lr = self.learning_rate.update_learning_rate(it).value
            if self.verbose > 1:  # pragma: no cover
                loop.set_description(
                    "loss=%1.3g lr=%1.3g "  # pylint: disable=E1307
                    "lrn=%1.3g" % (
                        loss, lr, lr_alive[0]))
            train_losses.append(loss)
            if (data_loader_val is not None and
                    (it + 1) % self.validation_every == 0):
                val_losses.append(self._evaluation(data_loader_val, bind))
        self.train_losses_ = train_losses
        self.validation_losses_ = (
            None if data_loader_val is None else val_losses)
        self.trained_coef_ = self.train_session_.get_state()
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
            dtype = proto_type_to_dtype(
                c_ortvalue.proto_type() if hasattr(c_ortvalue, 'proto_type')
                else c_ortvalue.data_type())
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

    def _iteration(self, data_loader, ort_lr, bind, use_numpy):
        actual_losses = []

        bind.bind_output('loss')

        if use_numpy:
            # onnxruntime does not copy the data, so the numpy
            # array must remain alive all along the iteration
            lr_alive = ort_lr.numpy()
            self._bind_input_ortvalue(
                self.input_names_[2], bind, lr_alive)

            # Slow iterations.
            for data, target in data_loader.iter_numpy():

                self._bind_input_ortvalue(
                    self.input_names_[0], bind, data)
                self._bind_input_ortvalue(
                    self.input_names_[1], bind, target)

                self.train_session_.run_with_iobinding(bind)
                outputs = bind.copy_outputs_to_cpu()
                if numpy.isinf(outputs[0]) or numpy.isnan(outputs[0]):
                    raise ConvergenceError(
                        "Loss is nan, learning_rate=%r, "
                        "the gradient descent has failed "
                        "(past losses=%r)." % (
                            ort_lr.numpy(),
                            [float(v[0]) for v in (
                                actual_losses if len(actual_losses) < 5
                                else actual_losses[-5:])]))
                actual_losses.append(outputs[0] / data.shape[0])
        else:
            self._bind_input_ortvalue(self.input_names_[2], bind, ort_lr)

            # Fast iterations
            # Slow iterations.
            for batch_size in data_loader.iter_bind(bind, self.input_names_):
                self.train_session_.run_with_iobinding(bind)
                # We copy the predicted output as well which is not needed.
                outputs = bind.copy_outputs_to_cpu()
                if numpy.isinf(outputs[0]) or numpy.isnan(outputs[0]):
                    raise ConvergenceError(
                        "Loss is nan or infinite, learning_rate=%r, "
                        "the gradient descent has failed "
                        "(past losses=%r)." % (
                            ort_lr.numpy(),
                            [float(v[0]) for v in (
                                actual_losses if len(actual_losses) < 5
                                else actual_losses[-5:])]))
                actual_losses.append(outputs[0] / batch_size)

        return numpy.array(actual_losses).mean()

    def _evaluation(self, data_loader, bind):
        lr_alive = numpy.array([0], dtype=numpy.float32)
        self._bind_input_ortvalue(self.input_names_[2], bind, lr_alive)
        bind.bind_output('loss')

        actual_losses = []
        for batch_size in data_loader.iter_bind(bind, self.input_names_):
            self.train_session_.run_with_iobinding(bind)
            outputs = bind.copy_outputs_to_cpu()
            if numpy.isinf(outputs[0]) or numpy.isnan(outputs[0]):
                raise EvaluationError(
                    "Loss is nan or infinite (%r), "
                    "evaluation has failed." % outputs[0])
            actual_losses.append(outputs[0] / batch_size)
        return numpy.array(actual_losses).sum() / len(data_loader)

    def _create_training_session(
            self, training_onnx, weights_to_train,
            loss_output_name='loss',
            training_optimizer_name='SGDOptimizer',
            device='cpu'):
        if training_optimizer_name != 'SGDOptimizer':
            raise NotImplementedError(
                "Only the SGDOptimizer is implemented not %r."
                "" % training_optimizer_name)
        ort_parameters = TrainingParameters()
        ort_parameters.loss_output_name = loss_output_name
        ort_parameters.use_mixed_precision = False
        # ort_parameters.world_rank = -1
        # ort_parameters.world_size = 1
        # ort_parameters.gradient_accumulation_steps = 1
        # ort_parameters.allreduce_post_accumulation = False
        # ort_parameters.deepspeed_zero_stage = 0
        # ort_parameters.enable_grad_norm_clip = False
        # ort_parameters.set_gradients_as_graph_outputs = False
        # ort_parameters.use_memory_efficient_gradient = False
        # ort_parameters.enable_adasum = False

        output_types = {}
        for output in training_onnx.graph.output:
            output_types[output.name] = output.type.tensor_type

        ort_parameters.weights_to_train = set(weights_to_train)
        ort_parameters.training_optimizer_name = training_optimizer_name
        # ort_parameters.lr_params_feed_name = lr_params_feed_name

        ort_parameters.optimizer_attributes_map = {
            name: {} for name in weights_to_train}
        ort_parameters.optimizer_int_attributes_map = {
            name: {} for name in weights_to_train}

        session_options = SessionOptions()
        # session_options.use_deterministic_compute = True

        lower_device = device.lower()
        if lower_device == 'cpu':
            provider = ['CPUExecutionProvider']
        elif (lower_device.startswith("cuda") or
                lower_device == 'gpu'):  # pragma: no cover
            provider = ['CUDAExecutionProvider']
        else:
            raise ValueError(  # pragma: no cover
                "Unexpected device %r." % device)

        session = TrainingSession(
            training_onnx.SerializeToString(), ort_parameters, session_options,
            providers=provider)

        return session

    def get_state(self):
        """
        Returns the trained weights.
        """
        if not hasattr(self, 'train_session_'):
            if hasattr(self, 'trained_coef_'):
                return self.trained_coef_
            raise AttributeError("Method fit must be called before.")
        return self.train_session_.get_state()

    def set_state(self, state):
        """
        Changes the trained weights.
        """
        if not hasattr(self, 'train_session_'):
            raise AttributeError(  # pragma: no cover
                "Method fit must be called before.")
        return self.train_session_.load_state(state)
