# pylint: disable=W0105
"""
@file
@brief Helper for :epkg:`onnxruntime-training`.
"""
from onnxruntime import SessionOptions, InferenceSession
from ..utils.onnx_function import function_onnx_graph
from ..utils.onnxruntime_helper import device_to_providers
from .base_onnx_function import BaseLearningOnnx


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
                "Attributes 'loss_grad_sess_bind_' or 'loss_grad_sess_' "
                "is missing. Method 'build_onnx_function' has not been called.")
        if weight is not None:
            self._bind_input_ortvalue(
                "weight", self.loss_grad_sess_bind_, weight, device)
        else:
            self.loss_grad_sess_bind_.clear_binding_inputs()
        self._bind_input_ortvalue(
            "X1", self.loss_grad_sess_bind_, expected, device)
        self._bind_input_ortvalue(
            "X2", self.loss_grad_sess_bind_, predicted, device)
        self.loss_grad_sess_bind_.bind_output('Y', device)
        self.loss_grad_sess_bind_.bind_output('Z', device)
        self.loss_grad_sess_._sess.run_with_iobinding(
            self.loss_grad_sess_bind_, None)
        loss, grad = self.loss_grad_sess_bind_.get_outputs()
        return loss, grad

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
        """
        if isinstance(class_name, BaseLearningLoss):
            return class_name
        cls = {SquareLearningLoss: ['square_error'],
               AbsoluteLearningLoss: ['absolute_error']}
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
    """

    def __init__(self):
        BaseLearningLoss.__init__(self)

    def build_onnx_function(self, opset, device, weight_name):
        so = SessionOptions()
        so.log_severity_level = 4

        # loss_grad
        self.loss_grad_onnx_ = function_onnx_graph(
            "grad_loss_square_error", target_opset=opset,
            weight_name=weight_name)
        self.loss_grad_sess_ = InferenceSession(
            self.loss_grad_onnx_.SerializeToString(), so,
            providers=device_to_providers(device))
        self.loss_grad_sess_bind_ = (
            self.loss_grad_sess_.io_binding()._iobinding)


class AbsoluteLearningLoss(BaseLearningLoss):
    """
    Implements a square loss :math:`|Y - Z|`
    where *Y* is the output and *Z* the expected output.
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
