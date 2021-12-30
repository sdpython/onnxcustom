# pylint: disable=C0415,E0611
"""
@file
@brief Onnx helper.
"""
import numpy
from .onnx_helper import dtype_to_var_type, add_initializer


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


def function_onnx_graph(name, target_opset=None, dtype=numpy.float32,
                        weight_name=None):
    """
    Returns the ONNX graph corresponding to a function.

    :param name: name
    :param target_opset: opset version
    :param dtype: computation type
    :param weight_name: weight name if any
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
        if weight_name is None:
            return glo[full_name](target_opset=target_opset, dtype=dtype)
        return glo[full_name](target_opset=target_opset, dtype=dtype,
                              weight_name=weight_name)
    raise ValueError(
        "Unable to find function %r in %r." % (
            full_name, list(sorted(
                k for k in glo if k.startswith('_onnx_')))))


def _onnx_square_error(target_opset=None, dtype=numpy.float32,
                       weight_name=None):
    """
    Returns the ONNX graph for function
    :math:`Y = f(X1, X2) = \\lVert X1 - X2 \\rVert ^2` or
    :math:`Y = f(X1, X2) = \\lVert X1 - X2 \\rVert ^2 w` if
    *weight_name* is not None

    .. gdot::
        :script: DOT-SECTION

        from mlprodict.onnxrt import OnnxInference
        from onnxcustom.utils.onnx_function import function_onnx_graph

        model_onnx = function_onnx_graph('square_error')
        oinf = OnnxInference(model_onnx, inplace=False)

        print("DOT-SECTION", oinf.to_dot())
    """
    from skl2onnx.algebra.onnx_ops import (
        OnnxSub, OnnxReduceSumSquare, OnnxReshape,
        OnnxReduceSum, OnnxMul)
    diff = OnnxSub('X1', 'X2', op_version=target_opset)
    if weight_name is None:
        res = OnnxReduceSumSquare(diff, op_version=target_opset,
                                  keepdims=0, output_names=['Y'])
    else:
        mul = OnnxMul(
            OnnxMul(diff, diff, op_version=target_opset),
            OnnxReshape(weight_name,
                        numpy.array([-1, 1], dtype=numpy.int64),
                        op_version=target_opset),
            op_version=target_opset)
        res = OnnxReduceSum(mul, op_version=target_opset,
                            keepdims=0, output_names=['Y'])
    var_type = dtype_to_var_type(dtype)
    varsx = [('X1', var_type([None, None])),
             ('X2', var_type([None, None]))]
    if weight_name is not None:
        varsx.append((weight_name, var_type([None])))
    onx = res.to_onnx(varsx, outputs=[('Y', var_type())],
                      target_opset=target_opset)
    if weight_name is not None:
        onx = add_initializer(
            onx, weight_name, numpy.array([1], dtype=dtype))
    return onx


def _onnx_grad_square_error(target_opset=None, dtype=numpy.float32,
                            weight_name=None):
    """
    Returns the ONNX graph for the gradient of function
    :math:`Y = f(X1, X2) = \\lVert X1 - X2 \\rVert ^2` or
    :math:`Y = f(X1, X2) = \\lVert X1 - X2 \\rVert ^2 w` if
    *weight_name* is not None

    .. gdot::
        :script: DOT-SECTION

        from mlprodict.onnxrt import OnnxInference
        from onnxcustom.utils.onnx_function import function_onnx_graph

        model_onnx = function_onnx_graph('grad_square_error')
        oinf = OnnxInference(model_onnx, inplace=False)

        print("DOT-SECTION", oinf.to_dot())
    """
    from skl2onnx.algebra.onnx_ops import OnnxSub, OnnxMul, OnnxReshape
    diff = OnnxSub('X1', 'X2', op_version=target_opset)
    if weight_name is None:
        res = OnnxMul(diff, numpy.array([-2], dtype=dtype),
                      op_version=target_opset, output_names=['Y'])
    else:
        res = OnnxMul(
            OnnxMul(diff, numpy.array([-2], dtype=dtype),
                    op_version=target_opset),
            OnnxReshape(weight_name,
                        numpy.array([-1, 1], dtype=numpy.int64),
                        op_version=target_opset),
            op_version=target_opset, output_names=['Y'])
    var_type = dtype_to_var_type(dtype)
    varsx = [('X1', var_type([None, None])), ('X2', var_type([None, None]))]
    if weight_name is not None:
        varsx.append((weight_name, var_type([None])))
    onx = res.to_onnx(varsx, outputs=[('Y', var_type())],
                      target_opset=target_opset)
    if weight_name is not None:
        onx = add_initializer(
            onx, weight_name, numpy.array([1], dtype=dtype))
    return onx


def _onnx_copy(target_opset=None, dtype=numpy.float32):
    """
    Returns the ONNX graph for function
    :math:`Y = X`.

    .. gdot::
        :script: DOT-SECTION

        from mlprodict.onnxrt import OnnxInference
        from onnxcustom.utils.onnx_function import function_onnx_graph

        model_onnx = function_onnx_graph('copy')
        oinf = OnnxInference(model_onnx, inplace=False)

        print("DOT-SECTION", oinf.to_dot())
    """
    from skl2onnx.algebra.onnx_ops import OnnxIdentity
    res = OnnxIdentity('X', op_version=target_opset,
                       output_names=['Y'])
    var_type = dtype_to_var_type(dtype)
    varsx = [('X', var_type())]
    onx = res.to_onnx(varsx, outputs=[('Y', var_type())],
                      target_opset=target_opset)
    return onx


def _onnx_zero(target_opset=None, dtype=numpy.float32):
    """
    Returns the ONNX graph for function
    :math:`Y = X * 0`.

    .. gdot::
        :script: DOT-SECTION

        from mlprodict.onnxrt import OnnxInference
        from onnxcustom.utils.onnx_function import function_onnx_graph

        model_onnx = function_onnx_graph('zero')
        oinf = OnnxInference(model_onnx, inplace=False)

        print("DOT-SECTION", oinf.to_dot())
    """
    from skl2onnx.algebra.onnx_ops import OnnxMul
    res = OnnxMul('X', numpy.array([0], dtype=dtype),
                  op_version=target_opset,
                  output_names=['Y'])
    var_type = dtype_to_var_type(dtype)
    varsx = [('X', var_type())]
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

        model_onnx = function_onnx_graph('axpy')
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


def _onnx_grad_loss_square_error(target_opset=None, dtype=numpy.float32,
                                 weight_name=None):
    """
    Returns the ONNX graph for function
    :math:`Y = f(X1, X2) = \\lVert X1 - X2 \\rVert ^2` or
    :math:`Y = f(X1, X2) = \\lVert X1 - X2 \\rVert ^2 w` if
    *weight_name* is not None and its gradient.

    .. gdot::
        :script: DOT-SECTION

        from mlprodict.onnxrt import OnnxInference
        from onnxcustom.utils.onnx_function import function_onnx_graph

        model_onnx = function_onnx_graph('square_error')
        oinf = OnnxInference(model_onnx, inplace=False)

        print("DOT-SECTION", oinf.to_dot())
    """
    from skl2onnx.algebra.onnx_ops import (
        OnnxSub, OnnxReduceSumSquare, OnnxMul,
        OnnxReduceSum, OnnxReshape)
    diff = OnnxSub('X1', 'X2', op_version=target_opset)
    if weight_name is None:
        res = OnnxReduceSumSquare(diff, op_version=target_opset,
                                  keepdims=0, output_names=['Y'])
        res2 = OnnxMul(diff, numpy.array([-2], dtype=dtype),
                       op_version=target_opset, output_names=['Z'])
    else:
        resh = OnnxReshape(weight_name,
                           numpy.array([-1, 1], dtype=numpy.int64),
                           op_version=target_opset)
        mul = OnnxMul(
            OnnxMul(diff, diff, op_version=target_opset),
            resh, op_version=target_opset)
        res = OnnxReduceSum(mul, op_version=target_opset,
                            keepdims=0, output_names=['Y'])
        res2 = OnnxMul(
            OnnxMul(diff, numpy.array([-2], dtype=dtype),
                    op_version=target_opset),
            resh, op_version=target_opset, output_names=['Z'])

    var_type = dtype_to_var_type(dtype)
    varsx = [('X1', var_type([None, None])),
             ('X2', var_type([None, None]))]
    if weight_name is not None:
        varsx.append((weight_name, var_type([None])))
    onx = res.to_onnx(
        varsx, outputs=[('Y', var_type()), ('Z', var_type())],
        target_opset=target_opset, other_outputs=[res2])
    if weight_name is not None:
        onx = add_initializer(
            onx, weight_name, numpy.array([1], dtype=dtype))
    return onx


def _onnx_linear_regression(target_opset=None, dtype=numpy.float32):
    """
    Returns the ONNX graph for function
    :math:`Y = f(X, A, B) = A X + B`.

    .. gdot::
        :script: DOT-SECTION

        from mlprodict.onnxrt import OnnxInference
        from onnxcustom.utils.onnx_function import function_onnx_graph

        model_onnx = function_onnx_graph('linear_regression')
        oinf = OnnxInference(model_onnx, inplace=False)

        print("DOT-SECTION", oinf.to_dot())
    """
    from skl2onnx.algebra.onnx_ops import (
        OnnxMatMul, OnnxAdd)
    res = OnnxAdd(
        OnnxMatMul('X', 'A', op_version=target_opset),
        'B', op_version=target_opset, output_names=['Y'])

    var_type = dtype_to_var_type(dtype)
    varsx = [('X', var_type([None, None])),
             ('A', var_type([None, None])),
             ('B', var_type([None, None]))]
    onx = res.to_onnx(
        varsx, outputs=[('Y', var_type())],
        target_opset=target_opset, other_outputs=[res])
    return onx
