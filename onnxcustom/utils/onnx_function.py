# pylint: disable=C0415,E0611
"""
@file
@brief Onnx helper.
"""
import numpy
from .onnx_helper import dtype_to_var_type


def get_supported_functions():
    """
    Returns the list of supported function by @see fn function_onnx_graph.
    """
    glo = globals()
    res = {}
    for k, v in glo.items():
        if k.startswith('_onnx_'):
            res[k[6:]] = v.__doc__
    return res


def function_onnx_graph(name, target_opset=None, dtype=numpy.float32):
    """
    Returns the ONNX graph corresponding to a function.

    :param name: name
    :param target_opset: opset version
    :param dtype: computation type
    :return: ONNX graph

    A wrong name will raise an exception giving the whole of
    supported function. One example with function `square_error`:

    .. gdot::
        :script: DOT-SECTION

        from mlprodict.onnxrt import OnnxInference
        from onnxcustom.utils.onnx_function import function_onnx_graph

        model_onnx = function_onnx_graph('square_error')
        oinf = OnnxInference(model_onnx, inplace=False)

        print("DOT-SECTION", oinf.to_dot())

    An example on how to use it:

    .. runpython::
        :showcode:

        import numpy
        from onnxruntime import InferenceSession
        from onnxcustom.utils.onnx_function import function_onnx_graph

        model_onnx = function_onnx_graph('square_error')
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {
            'X1': numpy.array([[0, 1]], dtype=numpy.float32).T,
            'X2': numpy.array([[1, 2]], dtype=numpy.float32).T})
        print(res[0])

    List of supported functions:

    .. runpython::
        :showcode:

        from onnxcustom.utils.onnx_function import get_supported_functions
        print("\\n".join(sorted(get_supported_functions())))
    """
    glo = globals()
    full_name = "_onnx_" + name
    if full_name in glo:
        return glo[full_name](target_opset=target_opset, dtype=dtype)
    raise ValueError(
        "Unable to find function %r in %r." % (
            full_name, ", ".join(list(sorted(
                k for k in glo if k.startswith('_onnx_'))))))


def _onnx_square_error(target_opset=None, dtype=numpy.float32):
    """
    Returns the ONNX graph for function
    :math:`Y = f(X1, X2) = \\lVert X1 - X2 \\rVert ^2`.

    .. gdot::
        :script: DOT-SECTION

        from mlprodict.onnxrt import OnnxInference
        from onnxcustom.utils.onnx_function import function_onnx_graph

        model_onnx = function_onnx_graph('square_error')
        oinf = OnnxInference(model_onnx, inplace=False)

        print("DOT-SECTION", oinf.to_dot())
    """
    from skl2onnx.algebra.onnx_ops import OnnxSub, OnnxReduceSumSquare
    diff = OnnxSub('X1', 'X2', op_version=target_opset)
    res = OnnxReduceSumSquare(diff, op_version=target_opset,
                              keepdims=0, output_names=['Y'])
    var_type = dtype_to_var_type(dtype)
    varsx = [('X1', var_type([None, None])), ('X2', var_type([None, None]))]
    onx = res.to_onnx(varsx, outputs=[('Y', var_type())],
                      target_opset=target_opset)
    return onx


def _onnx_grad_square_error(target_opset=None, dtype=numpy.float32):
    """
    Returns the ONNX graph for the gradient of function
    :math:`Y = f(X1, X2) = \\lVert X1 - X2 \\rVert ^2`.

    .. gdot::
        :script: DOT-SECTION

        from mlprodict.onnxrt import OnnxInference
        from onnxcustom.utils.onnx_function import function_onnx_graph

        model_onnx = function_onnx_graph('grad_square_error')
        oinf = OnnxInference(model_onnx, inplace=False)

        print("DOT-SECTION", oinf.to_dot())
    """
    from skl2onnx.algebra.onnx_ops import OnnxSub, OnnxMul
    diff = OnnxSub('X1', 'X2', op_version=target_opset)
    res = OnnxMul(diff, numpy.array([-2], dtype=dtype),
                  op_version=target_opset, output_names=['Y'])
    var_type = dtype_to_var_type(dtype)
    varsx = [('X1', var_type([None, None])), ('X2', var_type([None, None]))]
    onx = res.to_onnx(varsx, outputs=[('Y', var_type())],
                      target_opset=target_opset)
    return onx


def _onnx_axpy(target_opset=None, dtype=numpy.float32):
    """
    Returns the ONNX graph for function
    :math:`Y = f(X1, X2, \\alpha) = \\alpha X1 + X2`.

    .. gdot::
        :script: DOT-SECTION

        from mlprodict.onnxrt import OnnxInference
        from onnxcustom.utils.onnx_function import function_onnx_graph

        model_onnx = function_onnx_graph('avxpy')
        oinf = OnnxInference(model_onnx, inplace=False)

        print("DOT-SECTION", oinf.to_dot())
    """
    from skl2onnx.algebra.onnx_ops import OnnxAdd, OnnxMul
    res = OnnxAdd(OnnxMul('X1', 'alpha', op_version=target_opset),
                  'X2', op_version=target_opset, output_names=['Y'])
    var_type = dtype_to_var_type(dtype)
    varsx = [('X1', var_type()), ('X2', var_type()),
             ('alpha', var_type([1]))]
    onx = res.to_onnx(varsx, outputs=[('Y', var_type())],
                      target_opset=target_opset)
    return onx
