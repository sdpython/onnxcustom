"""
@file
@brief Helper for :epkg:`onnxruntime-training`.
"""
import inspect
import numpy
from .ortgradient import OrtGradientForwardBackward
from .optimizers import BaseEstimator


class OrtGradientForwardBackwardOptimizer(BaseEstimator):
    """
    Implements a simple :epkg:`Stochastic Gradient Descent`
    with :epkg:`onnxruntime-training`. It leverages class
    @see class OrtGradientForwardBackward.

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
    :param enable_logging: enable logging (mostly for debugging puporse
        as it slows down the training)
    """

    def __init__(self, model_onnx, weights_to_train, loss_output_name='loss',
                 max_iter=100, training_optimizer_name='SGDOptimizer',
                 batch_size=10, learning_rate='SGDRegressor',
                 device='cpu', device_idx=0,
                 warm_start=False, verbose=0, validation_every=0.1,
                 enable_logging=False):
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
        self.enable_logging = enable_logging
        if validation_every < 1:
            self.validation_every = int(self.max_iter * validation_every)
        else:
            self.validation_every = validation_every  # pragma: no cover

    def __getstate__(self):
        "Removes any non pickable attribute."
        atts = [k for k in self.__dict__ if not k.endswith('_')]
        return {att: getattr(self, att) for att in atts}

    def __setstate__(self, state):
        "Restores any non pickable attribute."
        for att, v in state.items():
            setattr(self, att, v)
        return self

    def get_state(self):
        """
        Returns the trained weights.
        """
        if not hasattr(self, 'train_state_'):
            raise AttributeError("Method fit must be called before.")
        return self.train_state_

    def set_state(self, state):
        """
        Changes the trained weights.
        """
        if not hasattr(self, 'train_session_'):
            raise AttributeError(  # pragma: no cover
                "Method fit must be called before.")
        self.train_state_ = state

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
        
        self.train_session_ = self._create_training_session(
            self.model_onnx, self.weights_to_train,
            device=self.device)

        self.input_names_ = self.train_session_[0].cls_type_._grad_input_names
        self.output_names_ = self.train_session_[0].cls_type_._bw_fetches_names

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
        elif not hasattr(self, 'state_'):
            self.state = [self.train_session_[0].get_ortvalue(name)
                          for name in self.input_names_]

        data_loader = OrtDataLoader(
            X, y, batch_size=self.batch_size, device=self.device)
        if X_val is not None:
            data_loader_val = OrtDataLoader(
                X_val, y_val, batch_size=X_val.shape[0], device=self.device)
        else:
            data_loader_val = None

        self.learning_rate.init_learning_rate()
        self.loss_index_ = self.output_names_.index(self.loss_output_name)

        if self.verbose > 0:  # pragma: no cover
            from tqdm import tqdm  # pylint: disable=C0415
            loop = tqdm(range(self.max_iter))
        else:
            loop = range(self.max_iter)

        train_losses = []
        val_losses = []
        lr = self.learning_rate.value
        for it in loop:
            loss = self._iteration(data_loader, lr, state)
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

    def _iteration(self, data_loader, learning_rate, state):
        actual_losses = []

        for ortx, orty in data_loader:
            state[0] = ortx
            prediction = self.train_session_[1].forward(state)            
            loss, gradient = self._gradient(prediction, orty)
            gradient = self.train_session_[1].backward([error])
            
            if len(gradient) != len(state):
                raise RuntimeError(
                    "gradient and state should have the same length but "
                    "%r != %r." % (len(gradient), len(state)))
            
            n = len(state) - len(self._weights_to_train)
            for i in range(n, len(state)):
                self._update_weights(state[i], gradient[i], learning_rate)
            
            actual_losses.append(loss / batch_size)

        return numpy.array(actual_losses).mean()

    def _evaluation(self, data_loader, bind):
        raise NotImplementedError()

    def _create_training_session(
            self, model_onnx, weights_to_train, device='cpu'):

        forback = OrtGradientForwardBackward(
            model_onnx, weights_to_train=weights_to_train,
            debug=False, enable_logging=self.enable_logging,
            providers=[device])
        inst = forback.new_instance()
        return (forback, inst)
