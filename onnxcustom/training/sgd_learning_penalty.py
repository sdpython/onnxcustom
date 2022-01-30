# pylint: disable=W0105
"""
@file
@brief Helper for :epkg:`onnxruntime-training`.
"""
from onnxruntime import SessionOptions, InferenceSession, RunOptions
from ..utils.onnx_function import function_onnx_graph
from ..utils.onnxruntime_helper import device_to_providers
from ._base_onnx_function import BaseLearningOnnx


class BaseLearningPenalty(BaseLearningOnnx):
    """
    Class handling the penalty on the coefficients for class
    @see cl OrtGradientForwardBackwardOptimizer.
    """

    def __init__(self):
        BaseLearningOnnx.__init__(self)
        self.ro_ = RunOptions()

    def _call_iobinding(self, sess, bind):
        sess.run_with_iobinding(bind, self.ro_)

    @staticmethod
    def select(class_name, **kwargs):
        """
        Returns an instance of a given initialized with
        *kwargs*.
        :param class_name: an instance of @see cl BaseLearningPenalty
            or a string among the following class names (see below)
        :return: instance of @see cl BaseLearningPenalty

        Possible values for *class_name*:
        * None or `'penalty'`: see @see cl L1L2PenaltyLearning
        """
        if isinstance(class_name, BaseLearningPenalty):
            return class_name
        cls = {NoLearningPenalty: [None, ''],
               ElasticLearningPenalty: ['elastic', 'l1l2']}
        for cl, aliases in cls.items():
            if class_name == cl.__class__.__name__ or class_name in aliases:
                return cl(**kwargs)
        raise ValueError(  # pragma: no cover
            "Unexpected class name %r. It should be one of %r." % (
                class_name, list(map(lambda c: c.__name__, cls))))

    def penalty_loss(self, device, loss, *weights):
        """
        Returns the received loss. Updates the loss inplace.

        :param device: device where the training takes place
        :param loss: loss without penalty
        :param weights: any weights to be penalized
        :return: loss
        """
        raise NotImplementedError(
            "penalty_loss must be overwritten.")

    def update_weights(self, device, statei):
        """
        Returns the received loss. Updates the weight inplace.

        :param device: device where the training takes place
        :param statei: loss without penalty
        :return: weight
        """
        raise NotImplementedError(
            "update_weights must be overwritten.")


class NoLearningPenalty(BaseLearningPenalty):
    """
    No regularization.
    """

    def __init__(self):
        BaseLearningPenalty.__init__(self)

    def build_onnx_function(self, opset, device, n_tensors):
        # Nothing to do.
        pass

    def penalty_loss(self, device, loss, *weights):
        """
        Returns the received loss. Updates the loss inplace.

        :param device: device where the training takes place
        :param loss: loss without penalty
        :param weights: any weights to be penalized
        :return: loss
        """
        return loss

    def update_weights(self, n_bind, device, statei):
        """
        Returns the received loss. Updates the weight inplace.

        :param device: device where the training takes place
        :param statei: loss without penalty
        :return: weight
        """
        return statei


class ElasticLearningPenalty(BaseLearningPenalty):
    """
    Implements a L1 or L2 regularization on weights.
    """

    def __init__(self, l1=0.5, l2=0.5):
        BaseLearningPenalty.__init__(self)
        self.l1 = l1
        self.l2 = l2

    def build_onnx_function(self, opset, device, n_tensors):
        so = SessionOptions()
        so.log_severity_level = 4

        # loss_grad
        self.penalty_onnx_ = function_onnx_graph(
            "n_penalty_elastic_error", target_opset=opset, n_tensors=n_tensors,
            loss_shape=None, l1_weight=self.l1, l2_weight=self.l2)
        self.penalty_sess_ = InferenceSession(
            self.penalty_onnx_.SerializeToString(), so,
            providers=device_to_providers(device))
        self.penalty_sess_bind_ = (
            self.penalty_sess_.io_binding()._iobinding)
        self.names_ = [i.name for i in self.penalty_onnx_.graph.input]

        # weight updates
        self.penalty_grad_onnx_ = function_onnx_graph(
            "update_penalty_elastic_error", target_opset=opset,
            l1=self.l1, l2=self.l2)
        self.penalty_grad_sess_ = InferenceSession(
            self.penalty_grad_onnx_.SerializeToString(), so,
            providers=device_to_providers(device))
        self.penalty_grad_sess_binds_ = [
            self.penalty_grad_sess_.io_binding()._iobinding
            for n in range(n_tensors)]

    def penalty_loss(self, device, *inputs):
        """
        Computes the penalty associated to every
        weights and adds them up to the loss.

        :param device: device where the training takes place
        :param inputs: loss without penalty and weights
        :return: loss + penatlies
        """
        if (not hasattr(self, "penalty_onnx_") or
                not hasattr(self, "penalty_sess_bind_")):
            raise RuntimeError(  # pragma: no cover
                "Attributes 'penalty_sess_bind_' or 'penalty_onnx_' is "
                "missing. Method 'build_onnx_function' has not been called.")
        if len(self.names_) != len(inputs):
            raise RuntimeError(  # pragma: no cover
                "Mismatched number of inputs: %d != %d." % (
                    len(self.names_), len(inputs)))

        for name, inp in zip(self.names_, inputs):
            self._bind_input_ortvalue(
                name, self.penalty_sess_bind_, inp, device, cache=True)
        self._bind_output_ortvalue(
            'Y', self.penalty_sess_bind_, inputs[0], cache=True)
        self._call_iobinding(self.penalty_sess_._sess, self.penalty_sess_bind_)
        return self.penalty_sess_bind_.get_outputs()[0]

    def update_weights(self, n_bind, device, statei):
        if (not hasattr(self, "penalty_grad_onnx_") or
                not hasattr(self, "penalty_grad_sess_binds_")):
            raise RuntimeError(  # pragma: no cover
                "Attributes 'penalty_grad_sess_binds_' or "
                "'penalty_grad_onnx_' is missing. Method "
                "'build_onnx_function' has not been called.")
        bind = self.penalty_grad_sess_binds_[n_bind]
        self._bind_input_ortvalue("X", bind, statei, device, cache=True)
        self._bind_output_ortvalue('Y', bind, statei, cache=True)
        self._call_iobinding(self.penalty_grad_sess_._sess, bind)
        return bind.get_outputs()[0]  # X
