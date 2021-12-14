"""
@file
@brief Helper for :epkg:`onnxruntime-training`.
"""
import inspect
import numpy


class BaseLearningRate:
    """
    Class handling the learning rate update after every
    iteration of a gradient. Two methods need to be overwritten
    `init_learning_rate` and `update_learning_rate`. The first one
    starts the loop, the second returns the next one.
    """

    def __init__(self):
        pass

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
            @see cl LearningRateSGDRegressor is used
        :return: instance of @see cl BaseLearningRate

        Possible values for *class_name*:
        * `'LearningRateSGDRegressor'`: see @see cl LearningRateSGDRegressor
        """
        if isinstance(class_name, BaseLearningRate):
            return class_name
        if isinstance(class_name, float):
            return LearningRateSGDRegressor(class_name)
        cls = {LearningRateSGDRegressor: ['SGDRegressor']}
        for cl, aliases in cls.items():
            if class_name == cl.__class__.__name__ or class_name in aliases:
                return cl(**kwargs)
        raise ValueError(  # pragma: no cover
            "Unexpected class name %r. It should be one of %r." % (
                class_name, list(map(lambda c: c.__name__, cls))))

    @classmethod
    def _get_param_names(cls):
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        init_signature = inspect.signature(init)
        parameters = [
            p for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD]
        return [(p.name, p.default) for p in parameters]

    def __repr__(self):
        """
        Usual
        """
        param = self._get_param_names()
        ps = []
        for k, v in param:
            if k not in self.__dict__:
                continue  # pragma: no cover
            ov = getattr(self, k)
            if v is not inspect._empty or ov != v:
                ro = repr(ov)
                ps.append("%s=%s" % (k, ro))
        return "%s(%s)" % (self.__class__.__name__, ", ".join(ps))


class LearningRateSGDRegressor(BaseLearningRate):
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

    @property
    def value(self):
        "Returns the current learning rate."
        return self.value_
