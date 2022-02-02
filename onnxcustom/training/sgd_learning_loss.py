# pylint: disable=W0105
"""
@file
@brief Helper for :epkg:`onnxruntime-training`.
"""
from onnxruntime import SessionOptions, InferenceSession, RunOptions
from ..utils.onnx_function import function_onnx_graph
from ..utils.onnxruntime_helper import device_to_providers
from ..utils.onnx_rewriter import unreduced_onnx_loss
from ._base_onnx_function import BaseLearningOnnx


class BaseLearningLoss(BaseLearningOnnx):
    """
    Class handling the loss for class
    @see cl OrtGradientForwardBackwardOptimizer.
    All classes inheriting from this one creates one ONNX function,
    returning the loss and the gradient of the loss against the
    outputs. Method `loss_gradient` is the main method, it computes
    the loss and the gradient defiend by one ONNX graph and
    executed by an instance of :epkg:`InferenceSession`.
    """

    def __init__(self):
        BaseLearningOnnx.__init__(self)
        self.ro_ = RunOptions()

    def build_onnx_score_function(self, opset, device, weight_name):
        """
        Assuming the loss function was created. This
        one takes the onnx graph and generate the onnx graph
        for the method `loss_score`.
        """
        if not hasattr(self, 'loss_grad_onnx_'):
            raise RuntimeError(
                "Missing attribute 'loss_grad_onnx_'. "
                "Method 'build_onnx_function' should be called first.")

        # score
        so = SessionOptions()
        so.log_severity_level = 4
        self.loss_score_onnx_ = unreduced_onnx_loss(
            self.loss_grad_onnx_, 'Y')  # pylint: disable=E1101
        self.loss_score_sess_ = InferenceSession(
            self.loss_score_onnx_.SerializeToString(), so,
            providers=device_to_providers(device))
        self.loss_score_sess_bind_ = (
            self.loss_score_sess_.io_binding()._iobinding)

    def _call_iobinding(self, sess, bind):
        sess.run_with_iobinding(bind, self.ro_)

    def loss_gradient(  # pylint: disable=E1101
            self, device, expected, predicted, weight=None):
        """
        Returns the loss and the gradient as OrtValue.

        :param device: device where the training takes place
        :param expected: expected value
        :param predicted: predicted value
        :param weight: optional, training weights
            (same dimension as expected and predicted tensors)
        :return: loss and gradient
        """
        if (not hasattr(self, "loss_grad_sess_") or
                not hasattr(self, "loss_grad_sess_bind_")):
            raise RuntimeError(  # pragma: no cover
                "Attributes 'loss_grad_sess_bind_' or 'loss_grad_sess_' is "
                "missing. Method 'build_onnx_function' has not been called.")
        bind = self.loss_grad_sess_bind_
        if weight is not None:
            self._bind_input_ortvalue(
                "weight", bind, weight, device, cache=True)
        else:
            self.clear_binding_inputs("weight", bind, cache=True)
        self._bind_input_ortvalue("X1", bind, expected, device, cache=True)
        self._bind_input_ortvalue("X2", bind, predicted, device, cache=True)
        self.loss_grad_sess_bind_.bind_output('Y', device)
        self.loss_grad_sess_bind_.bind_output('Z', device)
        self._call_iobinding(self.loss_grad_sess_._sess, bind)
        loss, grad = bind.get_outputs()
        return loss, grad

    def loss_scores(  # pylint: disable=E1101
            self, device, expected, predicted, weight=None):
        """
        Returns the weighted loss (or score)
        for every observation as OrtValue.

        :param device: device where the training takes place
        :param expected: expected value
        :param predicted: predicted value
        :param weight: optional, training weights
            (same dimension as expected and predicted tensors)
        :return: a score for every observation
        """
        if (not hasattr(self, "loss_score_sess_") or
                not hasattr(self, "loss_score_sess_bind_")):
            raise RuntimeError(  # pragma: no cover
                "Attributes 'loss_score_sess_bind_' or 'loss_score_sess_' is "
                "missing. Method 'build_onnx_function' has not been called.")
        bind = self.loss_score_sess_bind_
        if weight is not None:
            self._bind_input_ortvalue(
                "weight", bind, weight, device, cache=True)
        else:
            self.clear_binding_inputs("weight", bind, cache=True)
        self._bind_input_ortvalue("X1", bind, expected, device, cache=True)
        self._bind_input_ortvalue("X2", bind, predicted, device, cache=True)
        self.loss_score_sess_bind_.bind_output('Y', device)
        self._call_iobinding(self.loss_score_sess_._sess, bind)
        score = bind.get_outputs()
        return score[0]

    @staticmethod
    def select(class_name, **kwargs):
        """
        Returns an instance of a given initialized with
        *kwargs*.
        :param class_name: an instance of @see cl BaseLearningLoss
            or a string among the following class names (see below)
        :return: instance of @see cl BaseLearningLoss

        Possible values for *class_name*:
        * `'square_error'`: see @see cl SquareLearningLoss
        * `'absolute_error'`: see @see cl AbsoluteLearningLoss
        * `'elastic_error'`: see @see cl ElasticLearningLoss
        """
        if isinstance(class_name, BaseLearningLoss):
            return class_name
        cls = {SquareLearningLoss: ['square_error', 'square'],
               AbsoluteLearningLoss: ['absolute_error', 'absolute'],
               ElasticLearningLoss: ['elastic_error', 'elastic'],
               NegLogLearningLoss: ['log', 'neglog', 'logloss']}
        for cl, aliases in cls.items():
            if class_name == cl.__class__.__name__ or class_name in aliases:
                return cl(**kwargs)
        raise ValueError(  # pragma: no cover
            "Unexpected class name %r. It should be one of %r." % (
                class_name, list(map(lambda c: c.__name__, cls))))


class SquareLearningLoss(BaseLearningLoss):
    """
    Implements a square loss :math:`(Y - Z)^2`
    where *Y* is the output and *Z* the expected output.
    See @see fn _onnx_grad_loss_square_error for the ONNX
    implementation.
    """

    def __init__(self):
        BaseLearningLoss.__init__(self)

    def build_onnx_function(self, opset, device, weight_name):
        so = SessionOptions()
        so.log_severity_level = 4

        # loss_grad
        self.loss_grad_onnx_ = function_onnx_graph(
            "grad_loss_square_error", target_opset=opset,
            weight_name=weight_name, multiply=1)
        self.loss_grad_sess_ = InferenceSession(
            self.loss_grad_onnx_.SerializeToString(), so,
            providers=device_to_providers(device))
        self.loss_grad_sess_bind_ = (
            self.loss_grad_sess_.io_binding()._iobinding)

        # score
        self.build_onnx_score_function(opset, device, weight_name)


class AbsoluteLearningLoss(BaseLearningLoss):
    """
    Implements a square loss :math:`|Y - Z|`
    where *Y* is the output and *Z* the expected output.
    See @see fn _onnx_grad_loss_absolute_error for the ONNX
    implementation.
    """

    def __init__(self):
        BaseLearningLoss.__init__(self)

    def build_onnx_function(self, opset, device, weight_name):
        so = SessionOptions()
        so.log_severity_level = 4

        # loss_grad
        self.loss_grad_onnx_ = function_onnx_graph(
            "grad_loss_absolute_error", target_opset=opset,
            weight_name=weight_name)
        self.loss_grad_sess_ = InferenceSession(
            self.loss_grad_onnx_.SerializeToString(), so,
            providers=device_to_providers(device))
        self.loss_grad_sess_bind_ = (
            self.loss_grad_sess_.io_binding()._iobinding)

        # score
        self.build_onnx_score_function(opset, device, weight_name)


class ElasticLearningLoss(BaseLearningLoss):
    """
    Implements a square loss
    :math:`(Y - Z)^2 \\alpha + |Y - Z| * \\beta`
    where *Y* is the output and *Z* the expected output,
    :math:`\\alpha` is *l2_weight* and :math:`\\beta`
    is *l1_weight*.

    :param l1_weight: weight of L1 norm
    :param l2_weight: weight of L2 norm

    See @see fn _onnx_grad_loss_elastic_error for the ONNX
    implementation.
    """

    def __init__(self, l1_weight=0.5, l2_weight=0.5):
        BaseLearningLoss.__init__(self)
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight

    def build_onnx_function(self, opset, device, weight_name):
        so = SessionOptions()
        so.log_severity_level = 4

        # loss_grad
        self.loss_grad_onnx_ = function_onnx_graph(
            "grad_loss_elastic_error", target_opset=opset,
            weight_name=weight_name, l1_weight=self.l1_weight,
            l2_weight=self.l2_weight)
        self.loss_grad_sess_ = InferenceSession(
            self.loss_grad_onnx_.SerializeToString(), so,
            providers=device_to_providers(device))
        self.loss_grad_sess_bind_ = (
            self.loss_grad_sess_.io_binding()._iobinding)

        # score
        self.build_onnx_score_function(opset, device, weight_name)


class NegLogLearningLoss(BaseLearningLoss):
    """
    Implements a negative log loss
    `'log(yt, yp) = -(1-yt)\\log(1-yp) - yt\\log(yp)`,
    this only works for a binary classification where *yp* is the
    predicted probability, *yt* is the expected probability.
    *yt* is expected to be binary, *yp* is a matrix with two
    columns, the sum on every line is 1.
    However, this loss is usually applied after a function softmax
    and the gradient is directly computed from the loss to the
    raw score before they are processed through the softmax function
    (see class `Log
    <https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/
    linear_model/_sgd_fast.pyx#L236>`_).

    :param eps: clipping value for probabilities,
        avoids computing `log(0)`
    :param probability_function: function to convert
        raw scores into probabilities, default value is `sigmoid`
        for a logistic regression
    """

    def __init__(self, eps=1e-5, probability_function='sigmoid'):
        BaseLearningLoss.__init__(self)
        self.eps = eps
        self.probability_function = probability_function

    def build_onnx_function(self, opset, device, weight_name):
        so = SessionOptions()
        so.log_severity_level = 4

        # loss_grad
        fct_name = "grad_%s_neg_log_loss_error" % self.probability_function
        self.loss_grad_onnx_ = function_onnx_graph(
            fct_name, target_opset=opset,
            weight_name=weight_name, eps=self.eps)
        self.loss_grad_sess_ = InferenceSession(
            self.loss_grad_onnx_.SerializeToString(), so,
            providers=device_to_providers(device))
        self.loss_grad_sess_bind_ = (
            self.loss_grad_sess_.io_binding()._iobinding)

        # score
        self.build_onnx_score_function(opset, device, weight_name)
