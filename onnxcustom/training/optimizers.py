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
from .sgd_learning_rate import BaseLearningRate


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
    :param learning_rate: a name or a learning rate instance,
        see module :mod:`onnxcustom.training.sgd_learning_rate`
    :param device: `'cpu'` or `'cuda'`
    :param device_idx: device index
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
                 device='cpu', device_idx=0,
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
        self.device_idx = device_idx
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
                X_val, y_val, batch_size=X_val.shape[0], device=self.device)
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
            bind_lr = OrtValue.ortvalue_from_numpy(
                numpy.array([lr / self.batch_size], dtype=numpy.float32),
                self.device, self.device_idx)
            loss = self._iteration(data_loader, bind_lr,
                                   bind, use_numpy=use_numpy)
            lr = self.learning_rate.update_learning_rate(it).value
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

    def _iteration(self, data_loader, learning_rate, bind, use_numpy):
        actual_losses = []

        bind.bind_input(
            name=self.input_names_[2],
            device_type=learning_rate.device_name(), device_id=self.device_idx,
            element_type=numpy.float32, shape=learning_rate.shape(),
            buffer_ptr=learning_rate.data_ptr())
        bind.bind_output('loss')

        if use_numpy:
            # Slow iterations.
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

                self.train_session_.run_with_iobinding(bind)
                outputs = bind.copy_outputs_to_cpu()
                actual_losses.append(outputs[0] / data.shape()[0])
        else:
            # Fast iterations
            # Slow iterations.
            for batch_size in data_loader.iter_bind(bind, self.input_names_):
                self.train_session_.run_with_iobinding(bind)
                # We copy the predicted output as well which is not needed.
                outputs = bind.copy_outputs_to_cpu()
                actual_losses.append(outputs[0] / batch_size)

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
        session_options.use_deterministic_compute = True

        if device == 'cpu':
            provider = ['CPUExecutionProvider']
        elif device.startswith("cuda"):  # pragma: no cover
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
