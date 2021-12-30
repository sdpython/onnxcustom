"""
@file
@brief Optimizer with :epkg:`onnxruntime-training`.
"""
import inspect
from onnxruntime.capi._pybind_state import (  # pylint: disable=E0611
    OrtDevice as C_OrtDevice)
from ..utils.onnxruntime_helper import (
    get_ort_device, ort_device_to_string)
from .sgd_learning_rate import BaseLearningRate


class BaseEstimator:
    """
    Base class for optimizers.
    Implements common methods such `__repr__`.

    :param learning_rate: learning rate class,
        see module :mod:`onnxcustom.training.sgd_learning_rate`
    :param device: device as :epkg:`C_OrtDevice` or a string
        representing this device
    """

    def __init__(self, learning_rate, device):
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
