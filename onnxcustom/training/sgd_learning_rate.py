# pylint: disable=W0105
"""
@file
@brief Helper for :epkg:`onnxruntime-training`.
"""
import numpy
from onnxruntime import SessionOptions, InferenceSession
from onnxruntime.capi._pybind_state import (  # pylint: disable=E0611
    OrtValue as C_OrtValue)
from ..utils.onnx_function import function_onnx_graph
from ..utils.onnxruntime_helper import device_to_providers
from .base_onnx_function import BaseLearningOnnx


class BaseLearningRate(BaseLearningOnnx):
    """
    Class handling the learning rate update after every
    iteration of a gradient. Two methods need to be overwritten
    `init_learning_rate` and `update_learning_rate`. The first one
    starts the loop, the second returns the next one.
    """

    def __init__(self):
        BaseLearningOnnx.__init__(self)

    def init_learning_rate(self):
        """
        Initializes the learning rate at the beginning of the training.
        :return: self
        """
        raise NotImplementedError(
            "This method must be overwritten.")

    def update_learning_rate(self, t):
        """
        Updates the learning rate at the end of an iteration.
        :param t: iteration number
        :return: self
        """
        raise NotImplementedError(
            "This method must be overwritten.")

    @property
    def value(self):
        "Returns the current learning rate."
        raise NotImplementedError(
            "This method must be overwritten.")

    def __repr_extended__(self):
        return ', value=%r' % self.value

    @property
    def needs_grad(self):
        """
        Returns the True if the gradient update needs to retain
        past gradients.
        """
        raise NotImplementedError(
            "This method must be overwritten.")

    def update_weights(self, device, statei, gradienti, batch_size,
                       velocity=None):
        """
        Updates weights based on the algorithm this class
        is setting up.

        :param device: device
        :param statei: current weight
        :param gradienti: gradient
        :param batch_size: batch_size
        :param velocity: same shape as the gradient
        """
        raise NotImplementedError(
            "This method must be overwritten.")

    def loop(self, n=1000):
        """
        Loops over learning rate values, *n* to be precise.
        :param n: number of requested iterations
        :return: iterator
        """
        self.init_learning_rate()
        for i in range(n):
            yield self.value
            self.update_learning_rate(i + 1)

    @staticmethod
    def select(class_name, **kwargs):
        """
        Returns an instance of a given initialized with
        *kwargs*.
        :param class_name: an instance of @see cl BaseLearningRate
            or a string among the following class names (see below),
            it can also be a float and in that case, class
            @see cl LearningRateSGD is used
        :return: instance of @see cl BaseLearningRate

        Possible values for *class_name*:
        * `'SGD'` or `'LearningRateSGD'`: see @see cl LearningRateSGD
        """
        if isinstance(class_name, BaseLearningRate):
            return class_name
        if isinstance(class_name, float):
            return LearningRateSGD(class_name)
        cls = {LearningRateSGD: ['SGD'],
               LearningRateSGDNesterov: ['SGDNesterov', 'Nesterov']}
        for cl, aliases in cls.items():
            if class_name == cl.__class__.__name__ or class_name in aliases:
                return cl(**kwargs)
        raise ValueError(  # pragma: no cover
            "Unexpected class name %r. It should be one of %r." % (
                class_name, list(map(lambda c: c.__name__, cls))))


class LearningRateSGD(BaseLearningRate):
    """
    Implements the learning the same way as
    :class:`sklearn.linear_model.SGDRegressor`.

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
            by a heuristic proposed by Leon Bottou, this number is multiplied
            by a constant C to make the first number equal to *eta0*
        * `'invscaling'`: `eta = eta0 / pow(t, power_t)`

    Created attributes:
    * `eta0_`: initial eta0
    * `optimal_init_`: use when `learning_rate=='optimal'`
    * `value_`: value to be returned by property `value`
    """

    def __init__(self, eta0=0.01, alpha=0.0001, power_t=0.25,
                 learning_rate='invscaling'):
        BaseLearningRate.__init__(self)
        if learning_rate not in ('invscaling', 'optimal', 'constant'):
            raise ValueError(
                "Unxepected value for learning_rate=%r." % learning_rate)
        self.eta0 = eta0
        self.alpha = alpha
        self.power_t = power_t
        self.learning_rate = learning_rate.lower()
        self.value_ = None

    @property
    def value(self):
        "Returns the current learning rate."
        if self.value_ is None:
            raise RuntimeError(  # pragma: no cover
                "Method init_learning_rate was never called.")
        return self.value_

    @property
    def needs_grad(self):
        """
        Returns the True if the gradient update needs to retain
        past gradients.
        """
        return False

    def init_learning_rate(self):
        """
        Updates the learning rate at the end of an iteration.
        :return: self
        """
        self.eta0_ = self.eta0
        if self.learning_rate == "optimal":
            typw = numpy.sqrt(1.0 / numpy.sqrt(self.alpha))
            eta0 = typw / max(1.0, (1 + typw) * 2)
            self.optimal_init_ = 1.0 / (eta0 * self.alpha)
            eta = 1. / (self.alpha * self.optimal_init_)
            self.optimal_fact_ = self.eta0 / eta
            self.eta0_ = self.eta0
        else:
            self.eta0_ = self.eta0
        self.value_ = self.eta0_
        return self

    def update_learning_rate(self, t):
        """
        Updates the learning rate at the end of an iteration.
        :param t: iteration number
        :return: self
        """
        eta = self.value_
        if self.learning_rate == "optimal":
            eta = self.optimal_fact_ / (self.alpha * (self.optimal_init_ + t))
        elif self.learning_rate == "invscaling":
            eta = self.eta0_ / numpy.power(t + 1, self.power_t)
        self.value_ = eta
        return self

    def build_onnx_function(self, opset, device):
        so = SessionOptions()
        so.log_severity_level = 4

        self.axpy_onnx_ = function_onnx_graph("axpy")
        self.axpy_sess_ = InferenceSession(
            self.axpy_onnx_.SerializeToString(), so,
            providers=device_to_providers(device))
        self.axpy_sess_bind_ = self.axpy_sess_.io_binding()._iobinding

    def update_weights(self, device, statei, gradienti, batch_size,
                       velocity=None):
        if velocity is not None:
            raise RuntimeError(  # pragma: no cover
                "Velocity must be None for this way of updating weights.")
        self._bind_input_ortvalue(
            "X1", self.axpy_sess_bind_, gradienti, device)
        self._bind_input_ortvalue("X2", self.axpy_sess_bind_, statei, device)
        alpha = - self.value / batch_size  # pylint: disable=E1130
        alpha_alive = numpy.array([alpha], dtype=numpy.float32)
        ort_alpha_alive = C_OrtValue.ortvalue_from_numpy(
            alpha_alive, device)
        self._bind_input_ortvalue(
            "alpha", self.axpy_sess_bind_, ort_alpha_alive, device)
        self._bind_output_ortvalue('Y', self.axpy_sess_bind_, statei)
        self.axpy_sess_._sess.run_with_iobinding(self.axpy_sess_bind_, None)
        loss = self.axpy_sess_bind_.get_outputs()[0]
        return loss


class LearningRateSGDNesterov(LearningRateSGD):
    """
    Implements the learning the same way as
    :class:`sklearn.linear_model.SGDRegressor`.

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
            by a heuristic proposed by Leon Bottou, this number is multiplied
            by a constant C to make the first number equal to *eta0*
        * `'invscaling'`: `eta = eta0 / pow(t, power_t)`
    :param momentum: float, default=0.9
        Value of momentum used, must be larger than or equal to 0.
    :param nesterov: bool, default=True
        Whether to use nesterov's momentum or not. Use nesterov's if True
        Not using nesterov is equivalent to class @see cl LearningRateSGD.

    Created attributes:
    * `eta0_`: initial eta0
    * `optimal_init_`: use when `learning_rate=='optimal'`
    * `value_`: value to be returned by property `value`

    ::

        updates = [
            self.momentum * velocity - self.learning_rate * grad
            for velocity, grad in zip(self.velocities, grads)]
        self.velocities = updates

        if self.nesterov:
            updates_nesterov = [
                self.momentum * velocity - self.learning_rate * grad
                for velocity, grad in zip(self.velocities, grads)]
            return updates, updates_nesterov    --> new gradient and velocities
        else:
            return updates                      --> new gradient
    """

    def __init__(self, eta0=0.01, alpha=0.0001, power_t=0.25,
                 learning_rate='invscaling', momentum=0.9, nesterov=True):
        LearningRateSGD.__init__(
            self, eta0=eta0, alpha=alpha, power_t=power_t,
            learning_rate=learning_rate)
        self.momentum = momentum
        self.nesterov = nesterov

    @property
    def needs_grad(self):
        """
        Returns the True if the gradient update needs to retain
        past gradients.
        """
        return True

    def init_learning_rate(self):
        """
        Updates the learning rate at the end of an iteration.
        :return: self
        """
        return LearningRateSGD.init_learning_rate(self)

    def update_learning_rate(self, t):
        """
        Updates the learning rate at the end of an iteration.
        :param t: iteration number
        :return: self
        """
        return LearningRateSGD.update_learning_rate(self, t)

    def build_onnx_function(self, opset, device):
        so = SessionOptions()
        so.log_severity_level = 4

        # axpyw
        if self.nesterov:
            self.axpyw_onnx_ = function_onnx_graph("axpyw2")
        else:
            self.axpyw_onnx_ = function_onnx_graph("axpyw")
        self.axpyw_sess_ = InferenceSession(
            self.axpyw_onnx_.SerializeToString(), so,
            providers=device_to_providers(device))
        self.axpyw_sess_bind_ = self.axpyw_sess_.io_binding()._iobinding

    def update_weights(self, device, statei, gradienti, batch_size,
                       velocity=None):
        if velocity is None:
            raise RuntimeError(  # pragma: no cover
                "Velocity must not be None for this way of updating weights.")
        self._bind_input_ortvalue(
            "X1", self.axpyw_sess_bind_, gradienti, device)
        self._bind_input_ortvalue("X2", self.axpyw_sess_bind_, statei, device)
        self._bind_input_ortvalue("G", self.axpyw_sess_bind_, velocity, device)
        alpha = - self.value / batch_size  # pylint: disable=E1130
        alpha_alive = numpy.array([alpha], dtype=numpy.float32)
        beta = self.momentum
        beta_alive = numpy.array([beta], dtype=numpy.float32)
        ort_alpha_alive = C_OrtValue.ortvalue_from_numpy(
            alpha_alive, device)
        ort_beta_alive = C_OrtValue.ortvalue_from_numpy(
            beta_alive, device)
        self._bind_input_ortvalue(
            "alpha", self.axpyw_sess_bind_, ort_alpha_alive, device)
        self._bind_input_ortvalue(
            "beta", self.axpyw_sess_bind_, ort_beta_alive, device)
        self._bind_output_ortvalue('Y', self.axpyw_sess_bind_, statei)
        self._bind_output_ortvalue('Z', self.axpyw_sess_bind_, velocity)
        self.axpyw_sess_._sess.run_with_iobinding(self.axpyw_sess_bind_, None)
        return self.axpyw_sess_bind_.get_outputs()  # loss, velocity
