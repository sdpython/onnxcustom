"""
@file
@brief Helper for :epkg:`onnxruntime-training`.
"""
import inspect
import numpy
from onnxruntime import (  # pylint: disable=E0611
    OrtValue, TrainingParameters,
    SessionOptions, TrainingSession)
from .data_loader import OrtDataLoader


class BaseEstimator:
    """
    Base class for optimizers.
    Implements common methods such `__repr__`.
    """

    @classmethod
    def _get_param_names(cls):
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        init_signature = inspect.signature(init)
        parameters = [
            p for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD]
        return [(p.name, p.default) for p in parameters]

    def __repr__(self):
        param = self._get_param_names()
        ps = []
        for k, v in param:
            if k not in self.__dict__:
                continue
            ov = getattr(self, k)
            if v is not inspect._empty or ov != v:
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
    :param eta0: initial learning rate for the `'constant'`, `'invscaling'`
        or `'adaptive'` schedules.
    :param alpha: constant that multiplies the regularization term,
        the higher the value, the stronger the regularization.
        Also used to compute the learning rate when set to *learning_rate*
        is set to `'optimal'`.
    :param power_t: exponent for inverse scaling learning rate
    :param learning_rate: learning rate schedule:
        * `'constant'`: `eta = eta0`
        * `'optimal'`: `eta = 1.0 / (alpha * (t + t0))` where *t0* is chosen
            by a heuristic proposed by Leon Bottou.
        * `'invscaling'`: `eta = eta0 / pow(t, power_t)`
    :param device: `'cpu'` or `'cuda'`
    :param device_idx: device index
    :param warm_start: when set to True, reuse the solution of the previous
        call to fit as initialization, otherwise, just erase the previous
        solution.
    :param verbose: use :epkg:`tqdm` to display the training progress

    Once initialized, the class creates the attribute
    `session_` which holds an instance of `onnxruntime.TrainingSession`.

    See example :ref:`l-orttraining-nn-gpu`.
    """

    def __init__(self, model_onnx, weights_to_train, loss_output_name='loss',
                 max_iter=100, training_optimizer_name='SGDOptimizer',
                 batch_size=10, eta0=0.01, alpha=0.0001, power_t=0.25,
                 learning_rate='invscaling', device='cpu', device_idx=0,
                 warm_start=False, verbose=0, validation_every=0.1):
        # See https://scikit-learn.org/stable/modules/generated/
        # sklearn.linear_model.SGDRegressor.html
        self.model_onnx = model_onnx
        self.batch_size = batch_size
        self.weights_to_train = weights_to_train
        self.loss_output_name = loss_output_name
        self.training_optimizer_name = training_optimizer_name
        self.verbose = verbose
        self.max_iter = max_iter
        self.eta0 = eta0
        self.alpha = alpha
        self.power_t = power_t
        self.learning_rate = learning_rate.lower()
        self.device = device
        self.device_idx = device_idx
        self.warm_start = warm_start
        if validation_every < 1:
            self.validation_every = int(self.max_iter * validation_every)
        else:
            self.validation_every = validation_every

    def _init_learning_rate(self):
        self.eta0_ = self.eta0
        if self.learning_rate == "optimal":
            typw = numpy.sqrt(1.0 / numpy.sqrt(self.alpha))
            self.eta0_ = typw / max(1.0, (1 + typw) * 2)
            self.optimal_init_ = 1.0 / (self.eta0_ * self.alpha)
        else:
            self.eta0_ = self.eta0
        return self.eta0_

    def _update_learning_rate(self, t, eta):
        if self.learning_rate == "optimal":
            eta = 1.0 / (self.alpha * (self.optimal_init_ + t))
        elif self.learning_rate == "invscaling":
            eta = self.eta0_ / numpy.power(t + 1, self.power_t)
        return eta

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Trains the model.
        :param X: features
        :param y: expected output
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
                X_val, y_val, batch_size=X_val.shape[0], device=self.device)
        else:
            data_loader_val = None
        lr = self._init_learning_rate()
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
        for it in loop:
            bind_lr = OrtValue.ortvalue_from_numpy(
                numpy.array([lr / self.batch_size], dtype=numpy.float32),
                self.device, self.device_idx)
            loss = self._iteration(data_loader, bind_lr, bind)
            lr = self._update_learning_rate(it, lr)
            if self.verbose > 1:  # pragma: no cover
                loop.set_description(
                    "loss=%1.3g lr=%1.3g" % (  # pylint: disable=E1101,E1307
                        loss, lr))  # pylint: disable=E1101,E1307
            train_losses.append(loss)
            if (data_loader_val is not None and
                    (it + 1) % self.validation_every == 0):
                val_losses.append(self._evaluation(data_loader_val, bind))
        self.train_losses_ = train_losses
        self.validation_losses_ = (
            None if data_loader_val is None else val_losses)
        self.trained_coef_ = self.train_session_.get_state()
        return self

    def _iteration(self, data_loader, learning_rate, bind):
        actual_losses = []
        for data, target in data_loader:

            bind.bind_input(
                name=self.input_names_[0],
                device_type=self.device,
                device_id=self.device_idx,
                element_type=numpy.float32,
                shape=data.shape(),
                buffer_ptr=data.data_ptr())

            bind.bind_input(
                name=self.input_names_[1],
                device_type=self.device,
                device_id=self.device_idx,
                element_type=numpy.float32,
                shape=target.shape(),
                buffer_ptr=target.data_ptr())

            bind.bind_input(
                name=self.input_names_[2],
                device_type=learning_rate.device_name(), device_id=0,
                element_type=numpy.float32, shape=learning_rate.shape(),
                buffer_ptr=learning_rate.data_ptr())

            bind.bind_output('loss')

            self.train_session_.run_with_iobinding(bind)
            outputs = bind.copy_outputs_to_cpu()
            actual_losses.append(outputs[0] / data.shape()[0])
        return numpy.array(actual_losses).mean()

    def _evaluation(self, data_loader, bind):
        learning_rate = OrtValue.ortvalue_from_numpy(
            numpy.array([1], dtype=numpy.float32),
            self.device, self.device_idx)
        actual_losses = []
        for data, target in data_loader:

            bind.bind_input(
                name=self.input_names_[0],
                device_type=self.device,
                device_id=self.device_idx,
                element_type=numpy.float32,
                shape=data.shape(),
                buffer_ptr=data.data_ptr())

            bind.bind_input(
                name=self.input_names_[1],
                device_type=self.device,
                device_id=self.device_idx,
                element_type=numpy.float32,
                shape=target.shape(),
                buffer_ptr=target.data_ptr())

            bind.bind_input(
                name=self.input_names_[2],
                device_type=learning_rate.device_name(), device_id=0,
                element_type=numpy.float32, shape=learning_rate.shape(),
                buffer_ptr=learning_rate.data_ptr())

            bind.bind_output('loss')

            self.train_session_.run_with_iobinding(bind)
            outputs = bind.copy_outputs_to_cpu()
            actual_losses.append(outputs[0])
        return numpy.array(actual_losses).sum() / len(data_loader)

    def _create_training_session(
            self, training_onnx, weights_to_train,
            loss_output_name='loss',
            training_optimizer_name='SGDOptimizer',
            device='cpu'):
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
        session_options.use_deterministic_compute = True

        if device == 'cpu':
            provider = ['CPUExecutionProvider']
        elif device.startswith("cuda"):
            provider = ['CUDAExecutionProvider']
        else:
            raise ValueError("Unexpected device %r." % device)

        session = TrainingSession(
            training_onnx.SerializeToString(), ort_parameters, session_options,
            providers=provider)

        return session

    def get_state(self):
        """
        Returns the trained weights.
        """
        if not hasattr(self, 'train_session_'):
            raise AttributeError("Method fit must be called before.")
        return self.train_session_.get_state()

    def set_state(self, state):
        """
        Changes the trained weights.
        """
        if not hasattr(self, 'train_session_'):
            raise AttributeError("Method fit must be called before.")
        return self.train_session_.load_state(state)
