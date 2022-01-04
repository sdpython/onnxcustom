# pylint: disable=W0105
"""
@file
@brief Helper for :epkg:`onnxruntime-training`.
"""
import inspect
from io import BytesIO
import numpy
import onnx
from onnxruntime import SessionOptions, InferenceSession
from onnxruntime.capi._pybind_state import (  # pylint: disable=E0611
    OrtValue as C_OrtValue)
from ..utils.onnxruntime_helper import ort_device_to_string
from .excs import ProviderError


class BaseLearningOnnx:
    """
    Class handling ONNX function to manipulate OrtValue.
    Base class for @see cl BaseLearningRate and
    @see cl BaseLearningLoss.
    """

    def __init__(self):
        self.cache_in_ = {}
        self.cache_out_ = {}

    def __getstate__(self):
        """
        Overwrites getstate to get rid of InferenceSession.
        """
        atts = [k for k in self.__dict__ if not k.endswith('_')]
        state = {k: getattr(self, k) for k in atts}
        onx = [k for k in self.__dict__ if k.endswith('_onnx_')]
        for o in onx:
            state[o] = getattr(self, o).SerializeToString()
        onx = [k for k in self.__dict__ if k.endswith('_sess_')]
        bind = [k for k in self.__dict__ if k.endswith('_bind_')]
        for k in bind:
            state[k] = True
        binds = [k for k in self.__dict__ if k.endswith('_binds_')]
        for k in binds:
            state[k] = len(getattr(self, k))
        for o in onx:
            state[o] = getattr(self, o).get_providers()
        return state

    def __setstate__(self, state):
        """
        Overwrites getstate to get rid of InferenceSession.
        """
        for k, v in state.items():
            if not k.endswith('_onnx_') and not k.endswith('_sess_'):
                setattr(self, k, v)

        so = SessionOptions()
        so.log_severity_level = 4
        for k, v in state.items():
            if k.endswith('_onnx_'):
                setattr(self, k, onnx.load(BytesIO(v)))
                k2 = k.replace("onnx", "sess")
                prov = state[k2]
                setattr(self, k2, InferenceSession(
                    getattr(self, k).SerializeToString(), so,
                    providers=prov))
        for k, v in state.items():
            if k.endswith('_bind_'):
                k2 = k[:-5]
                setattr(self, k, getattr(self, k2).io_binding()._iobinding)
            elif k.endswith('_binds_'):
                k2 = k[:-6]
                n = v
                setattr(self, k, [
                    getattr(self, k2).io_binding()._iobinding
                    for i in range(n)])
        self.cache_in_ = {}
        self.cache_out_ = {}
        return self

    def __repr_extended__(self):
        return ''

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
        return "%s(%s)%s" % (
            self.__class__.__name__, ", ".join(ps), self.__repr_extended__())

    def build_onnx_function(self, opset, device, *args):
        """
        This class updates the weights.
        It assumes it can do operator on *OrtValue*.
        This can be done through ONNX graph.
        This function creates :epkg:`InferenceSession`
        which do that.

        :param opset: opset to use
        :param device: :epkg:`C_OrtDevice`
        :param args: additional arguments
        """
        raise NotImplementedError(
            "This method must be overwritten.")

    def clear_binding_inputs(self, name, bind, cache=False):
        """
        Clears binding and empty cache.
        """
        def _cache_in(name, bind):
            key = name, id(bind)
            if key in self.cache_in_ and self.cache_in_[key] == 0:
                return True
            self.cache_in_[key] = 0
            return False

        if cache and _cache_in(name, bind):
            return
        bind.clear_binding_inputs()

    def _bind_input_ortvalue(self, name, bind, c_ortvalue, device, cache=False):
        """
        Binds :epkg:`C_OrtValue` to the structure used by
        :epkg:`InferenceSession` to run inference.

        :param name: str
        :param bind: python structure
        :param c_ortvalue: C structure for OrtValue (:epkg:`C_OrtValue`),
            it can be also a numpy array
        :param device: device
        :param cache: avoids binding again the data pointer did not change,
            only works when c_ortvalue is of :epkg:`C_OrtValue`
        """
        def _cache_in(name, bind, c_ortvalue):
            key = name, id(bind)
            ptr = self.cache_in_.get(key, 0)
            ptr2 = c_ortvalue.data_ptr()
            if ptr == ptr2:
                return True
            self.cache_in_[key] = ptr2
            return False

        def _cache_np(name, bind, c_ortvalue):
            key = name, id(bind)
            ptr = self.cache_in_.get(key, 0)
            ptr2 = c_ortvalue.__array_interface__['data'][0]
            if ptr == ptr2:
                return True
            self.cache_in_[key] = ptr2
            return False

        def do_bind(name, bind, c_ortvalue):
            bind.bind_ortvalue_input(name, c_ortvalue)

        if isinstance(c_ortvalue, C_OrtValue):
            if cache and _cache_in(name, bind, c_ortvalue):
                return
            do_bind(name, bind, c_ortvalue)
        elif isinstance(c_ortvalue, numpy.ndarray):
            if self.device_type() != device.cpu():  # pylint: disable=E1101
                raise ProviderError(
                    "device=%s is not CPU." % ort_device_to_string(
                        device))
            if cache and _cache_np(name, bind, c_ortvalue):
                return
            bind.bind_input(
                name, device, c_ortvalue.dtype, c_ortvalue.shape,
                c_ortvalue.__array_interface__['data'][0])
        else:
            raise TypeError(  # pragma: no cover
                "Unable to bind type %r for name %r." % (
                    type(c_ortvalue), name))

    def _bind_output_ortvalue(self, name, bind, c_ortvalue, cache=False):
        """
        Binds :epkg:`C_OrtValue` to the structure used by
        :epkg:`InferenceSession` to run inference.

        :param name: str
        :param bind: python structure
        :param c_ortvalue: C structure for OrtValue (:epkg:`C_OrtValue`)
        :param cache: avoids binding again the data pointer did not change,
            only works when c_ortvalue is of :epkg:`C_OrtValue`

        This method can be used for inplace computation.
        """
        def _cache_out(name, bind, c_ortvalue):
            key = name, id(bind)
            ptr = self.cache_in_.get(key, 0)
            ptr2 = c_ortvalue.data_ptr()
            if ptr == ptr2:
                return True
            self.cache_in_[key] = ptr2
            return False

        def do_bind(name, bind, c_ortvalue):
            bind.bind_ortvalue_output(name, c_ortvalue)

        if isinstance(c_ortvalue, C_OrtValue):
            if cache and _cache_out(name, bind, c_ortvalue):
                return
            do_bind(name, bind, c_ortvalue)
        else:
            raise TypeError(  # pragma: no cover
                "Unable to bind type %r for name %r." % (
                    type(c_ortvalue), name))

    @classmethod
    def _get_param_names(cls):
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        init_signature = inspect.signature(init)
        parameters = [
            p for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD]
        return [(p.name, p.default) for p in parameters]
