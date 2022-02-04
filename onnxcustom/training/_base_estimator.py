"""
@file
@brief Optimizer with :epkg:`onnxruntime-training`.
"""
import inspect
from onnxruntime.capi._pybind_state import (  # pylint: disable=E0611
    OrtDevice as C_OrtDevice)
from ..utils.onnxruntime_helper import (
    get_ort_device, ort_device_to_string)
from ..utils.onnx_helper import replace_initializers_into_onnx
from ._base import BaseOnnxClass
from ._base_onnx_function import BaseLearningOnnx
from .sgd_learning_rate import BaseLearningRate


class BaseEstimator(BaseOnnxClass):
    """
    Base class for optimizers.
    Implements common methods such `__repr__`.

    :param model_onnx: onnx graph to train
    :param learning_rate: learning rate class,
        see module :mod:`onnxcustom.training.sgd_learning_rate`
    :param device: device as :epkg:`C_OrtDevice` or a string
        representing this device
    """

    def __init__(self, model_onnx, learning_rate, device):
        self.model_onnx = model_onnx
        self.learning_rate = BaseLearningRate.select(learning_rate)
        self.device = get_ort_device(device)

    @classmethod
    def _get_param_names(cls):
        "Extracts all parameters to serialize."
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        init_signature = inspect.signature(init)
        parameters = [
            p for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD]
        return [(p.name, p.default) for p in parameters]

    def get_params(self, deep=False):
        """
        Returns the list of parameters.
        Parameter *deep* is unused.
        """
        ps = set(p[0] for p in self._get_param_names())
        res = {att: getattr(self, att)
               for att in dir(self)
               if not att.endswith('_') and att in ps}
        if 'device' in res and not isinstance(res['device'], str):
            res['device'] = ort_device_to_string(res['device'])
        return res

    def set_params(self, **params):
        """
        Returns the list of parameters.
        Parameter *deep* is unused.
        """
        for k, v in params.items():
            if k == 'device' and isinstance(v, str):
                v = get_ort_device(v)
            setattr(self, k, v)
        self.build_onnx_function()  # pylint: disable=E1101
        return self

    def __repr__(self):
        "Usual."
        param = self._get_param_names()
        ps = []
        for k, v in param:
            if k not in self.__dict__:
                continue  # pragma: no cover
            ov = getattr(self, k)
            if isinstance(ov, BaseLearningOnnx):
                ps.append("%s=%s" % (k, repr(ov)))
            elif isinstance(ov, C_OrtDevice):
                ps.append("%s=%r" % (k, ort_device_to_string(ov)))
            elif v is not inspect._empty or ov != v:
                ro = repr(ov)
                if len(ro) > 50 or "\n" in ro:
                    ro = ro[:10].replace("\n", " ") + "..."
                    ps.append("%s=%r" % (k, ro))
                else:
                    ps.append("%s=%r" % (k, ov))
        return "%s(%s)" % (self.__class__.__name__, ", ".join(ps))

    def __getstate__(self):
        "Removes any non pickable attribute."
        atts = [k for k in self.__dict__ if not k.endswith('_')]
        if hasattr(self, 'trained_coef_'):
            atts.append('trained_coef_')
        state = {att: getattr(self, att) for att in atts}
        state['device'] = ort_device_to_string(state['device'])
        return state

    def __setstate__(self, state):
        "Restores any non pickable attribute."
        for att, v in state.items():
            setattr(self, att, v)
        self.device = get_ort_device(self.device)
        return self

    def get_trained_onnx(self):
        """
        Returns the trained onnx graph, the initial graph
        modified by replacing the initializers with the trained
        weights.

        :return: onnx graph
        """
        raise NotImplementedError(  # pragma: no cover
            "The method needs to be overloaded.")

    def _get_trained_onnx(self, state, model=None):
        """
        Returns the trained onnx graph, the initial graph
        modified by replacing the initializers with the trained
        weights.

        :param state: trained weights
        :param model: replace the weights in another graph
            than the training graph
        :return: onnx graph
        """
        if model is None:
            return replace_initializers_into_onnx(
                self.model_onnx, state)
        return replace_initializers_into_onnx(model, state)
