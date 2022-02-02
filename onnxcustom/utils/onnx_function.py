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
                        weight_name=None, **kwargs):
    """
    Returns the ONNX graph corresponding to a function.

    :param name: name
    :param target_opset: opset version
    :param dtype: computation type
    :param weight_name: weight name if any
    :param kwargs: additional parameters
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
            return glo[full_name](target_opset=target_opset,
                                  dtype=dtype, **kwargs)
        return glo[full_name](target_opset=target_opset, dtype=dtype,
                              weight_name=weight_name, **kwargs)
    raise ValueError(
        "Unable to find function %r in %r." % (
            full_name, list(sorted(
                k for k in glo if k.startswith('_onnx_')))))


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


def _onnx_axpyw(target_opset=None, dtype=numpy.float32):
    """
    Returns the ONNX graph for function
    :math:`Y, Z = f(X1, X2, G, \\alpha, \\beta) = (Y, Z)`
    where :math:`Z = \\beta G + \\alpha X1` and
    :math:`Y = Z + X2`.

    .. gdot::
        :script: DOT-SECTION

        from mlprodict.onnxrt import OnnxInference
        from onnxcustom.utils.onnx_function import function_onnx_graph

        model_onnx = function_onnx_graph('axpy')
        oinf = OnnxInference(model_onnx, inplace=False)

        print("DOT-SECTION", oinf.to_dot())
    """
    from skl2onnx.algebra.onnx_ops import OnnxAdd, OnnxMul
    s1 = OnnxMul('X1', 'alpha', op_version=target_opset)
    s2 = OnnxMul('G', 'beta', op_version=target_opset)
    Z = OnnxAdd(s1, s2, op_version=target_opset,
                output_names=['Z'])
    Y = OnnxAdd(Z, 'X2', op_version=target_opset, output_names=['Y'])
    var_type = dtype_to_var_type(dtype)
    varsx = [('X1', var_type()), ('X2', var_type()),
             ('G', var_type()),
             ('alpha', var_type([1])), ('beta', var_type([1]))]
    onx = Y.to_onnx(
        varsx, outputs=[('Y', var_type()), ('Z', var_type())],
        target_opset=target_opset, other_outputs=[Z])
    return onx


def _onnx_axpyw2(target_opset=None, dtype=numpy.float32):
    """
    Returns the ONNX graph for function
    :math:`Y, Z = f(X1, X2, G, \\alpha, \\beta) = (Y, Z)`
    where :math:`Z = \\beta G + \\alpha X1` and
    :math:`Y = \\beta * Z + \\alpha X1 + X2`.

    .. gdot::
        :script: DOT-SECTION

        from mlprodict.onnxrt import OnnxInference
        from onnxcustom.utils.onnx_function import function_onnx_graph

        model_onnx = function_onnx_graph('axpy')
        oinf = OnnxInference(model_onnx, inplace=False)

        print("DOT-SECTION", oinf.to_dot())
    """
    from skl2onnx.algebra.onnx_ops import OnnxAdd, OnnxMul
    s1 = OnnxMul('X1', 'alpha', op_version=target_opset)
    s2 = OnnxMul('G', 'beta', op_version=target_opset)
    Z = OnnxAdd(s1, s2, op_version=target_opset,
                output_names=['Z'])
    s2_2 = OnnxMul(Z, 'beta', op_version=target_opset)
    s2_3 = OnnxAdd(s1, s2_2, op_version=target_opset)
    Y = OnnxAdd(s2_3, 'X2', op_version=target_opset, output_names=['Y'])
    var_type = dtype_to_var_type(dtype)
    varsx = [('X1', var_type()), ('X2', var_type()),
             ('G', var_type()),
             ('alpha', var_type([1])), ('beta', var_type([1]))]
    onx = Y.to_onnx(
        varsx, outputs=[('Y', var_type()), ('Z', var_type())],
        target_opset=target_opset, other_outputs=[Z])
    return onx


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
        res = OnnxReduceSumSquare(diff, op_version=target_opset)
    else:
        mul = OnnxMul(
            OnnxMul(diff, diff, op_version=target_opset),
            OnnxReshape(weight_name,
                        numpy.array([-1, 1], dtype=numpy.int64),
                        op_version=target_opset),
            op_version=target_opset)
        res = OnnxReduceSum(mul, op_version=target_opset)
    res = OnnxReshape(res, numpy.array([-1], numpy.int64),
                      op_version=target_opset,
                      output_names=['Y'])
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


def _onnx_grad_loss_square_error(target_opset=None, dtype=numpy.float32,
                                 weight_name=None, multiply=2):
    """
    Returns the ONNX graph for function
    :math:`Y = f(X1, X2) = \\lVert (X1 - X2) \\rVert ^2` or
    :math:`Y = f(X1, X2) = \\lVert (\\sqrt{w}(X1 - X2) \\rVert ^2 w` if
    *weight_name* is not None and its gradient.

    .. gdot::
        :script: DOT-SECTION

        from mlprodict.onnxrt import OnnxInference
        from onnxcustom.utils.onnx_function import function_onnx_graph

        model_onnx = function_onnx_graph('grad_loss_square_error')
        oinf = OnnxInference(model_onnx, inplace=False)

        print("DOT-SECTION", oinf.to_dot())
    """
    from skl2onnx.algebra.onnx_ops import (
        OnnxSub, OnnxReduceSumSquare, OnnxMul,
        OnnxReduceSum, OnnxReshape)
    diff = OnnxSub('X1', 'X2', op_version=target_opset)
    if weight_name is None:
        res = OnnxMul(OnnxReduceSumSquare(diff, op_version=target_opset),
                      numpy.array([multiply * 0.5], dtype=numpy.float32),
                      op_version=target_opset)
        res2 = OnnxMul(diff, numpy.array([-multiply], dtype=dtype),
                       op_version=target_opset, output_names=['Z'])
    else:
        resh = OnnxReshape(weight_name,
                           numpy.array([-1, 1], dtype=numpy.int64),
                           op_version=target_opset)
        mul = OnnxMul(
            OnnxMul(diff, diff, op_version=target_opset),
            resh, op_version=target_opset)
        res = OnnxMul(OnnxReduceSum(mul, op_version=target_opset),
                      numpy.array([multiply * 0.5], dtype=numpy.float32),
                      op_version=target_opset)

        res2 = OnnxMul(
            OnnxMul(diff, numpy.array([-multiply], dtype=dtype),
                    op_version=target_opset),
            resh, op_version=target_opset, output_names=['Z'])

    res = OnnxReshape(res, numpy.array([-1], numpy.int64),
                      op_version=target_opset,
                      output_names=['Y'])

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


def _onnx_grad_loss_absolute_error(target_opset=None, dtype=numpy.float32,
                                   weight_name=None):
    """
    Returns the ONNX graph for function
    :math:`Y = f(X1, X2) = \\lVert X1 - X2 \\rVert` or
    :math:`Y = f(X1, X2) = \\lVert (X1 - X2)w \\rVert` if
    *weight_name* is not None and its gradient.

    .. gdot::
        :script: DOT-SECTION

        from mlprodict.onnxrt import OnnxInference
        from onnxcustom.utils.onnx_function import function_onnx_graph

        model_onnx = function_onnx_graph('grad_loss_absolute_error')
        oinf = OnnxInference(model_onnx, inplace=False)

        print("DOT-SECTION", oinf.to_dot())
    """
    from skl2onnx.algebra.onnx_ops import (
        OnnxSub, OnnxMul, OnnxReduceSum, OnnxReshape,
        OnnxSign, OnnxAbs)
    diff = OnnxSub('X1', 'X2', op_version=target_opset)
    abs_diff = OnnxAbs(diff, op_version=target_opset)
    if weight_name is None:
        res = OnnxReduceSum(abs_diff, op_version=target_opset)
        res2 = OnnxSign(diff, op_version=target_opset,
                        output_names=['Z'])
    else:
        resh = OnnxReshape(weight_name,
                           numpy.array([-1, 1], dtype=numpy.int64),
                           op_version=target_opset)
        mul = OnnxMul(abs_diff, resh, op_version=target_opset)
        res = OnnxReduceSum(mul, op_version=target_opset)
        res2 = OnnxMul(
            OnnxSign(diff, op_version=target_opset),
            resh, op_version=target_opset, output_names=['Z'])

    res = OnnxReshape(res, numpy.array([-1], numpy.int64),
                      op_version=target_opset,
                      output_names=['Y'])
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


def _onnx_grad_loss_elastic_error(target_opset=None, dtype=numpy.float32,
                                  weight_name=None,
                                  l1_weight=0.01, l2_weight=0.01):
    """
    Returns the ONNX graph for function
    :math:`Y = f(X1, X2) = \\beta \\lVert X1 - X2 \\rVert +
    \\alpha \\lVert X1 - X2 \\rVert^2` or
    :math:`Y = f(X1, X2) = \\beta \\lVert w(X1 - X2) \\rVert +
    \\alpha \\lVert (\\sqrt{w})(X1 - X2) \\rVert^2` if
    *weight_name* is not None and its gradient.
    *l1_weight* is :math:`\\beta` and
    *l2_weight* is :math:`\\alpha`.

    .. gdot::
        :script: DOT-SECTION

        from mlprodict.onnxrt import OnnxInference
        from onnxcustom.utils.onnx_function import function_onnx_graph

        model_onnx = function_onnx_graph('grad_loss_elastic_error')
        oinf = OnnxInference(model_onnx, inplace=False)

        print("DOT-SECTION", oinf.to_dot())
    """
    from skl2onnx.algebra.onnx_ops import (
        OnnxSub, OnnxMul, OnnxAdd, OnnxIdentity,
        OnnxReduceSum, OnnxReshape, OnnxSign, OnnxAbs)
    diff = OnnxSub('X1', 'X2', op_version=target_opset)
    abs_diff = OnnxAbs(diff, op_version=target_opset)

    # loss
    abs_diff_l1 = OnnxMul(
        abs_diff, numpy.array([l1_weight], dtype=dtype),
        op_version=target_opset)
    diff_l2 = OnnxMul(
        OnnxMul(diff, diff, op_version=target_opset),
        numpy.array([l2_weight], dtype=dtype),
        op_version=target_opset)
    score = OnnxAdd(abs_diff_l1, diff_l2, op_version=target_opset)

    # gradient
    grad_l1 = OnnxMul(
        OnnxSign(diff, op_version=target_opset),
        numpy.array([l1_weight], dtype=dtype),
        op_version=target_opset)
    grad_l2 = OnnxMul(
        diff, numpy.array([l2_weight * -2], dtype=dtype),
        op_version=target_opset)
    grad = OnnxAdd(grad_l1, grad_l2, op_version=target_opset)

    if weight_name is None:
        res = OnnxReduceSum(score, op_version=target_opset)
        res2 = OnnxIdentity(grad, op_version=target_opset,
                            output_names=['Z'])
    else:
        resh = OnnxReshape(weight_name,
                           numpy.array([-1, 1], dtype=numpy.int64),
                           op_version=target_opset)
        res = OnnxReduceSum(
            OnnxMul(score, resh, op_version=target_opset),
            op_version=target_opset)
        res2 = OnnxMul(grad, resh, op_version=target_opset,
                       output_names=['Z'])

    res = OnnxReshape(res, numpy.array([-1], numpy.int64),
                      op_version=target_opset,
                      output_names=['Y'])

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


def _onnx_grad_penalty_elastic_error(target_opset=None, dtype=numpy.float32,
                                     l1_weight=0.01, l2_weight=0.01):
    """
    Returns the ONNX graph for function
    :math:`Y = f(W) = \\beta \\lVert W \\rVert +
    \\alpha \\lVert W \\rVert^2`
    *l1_weight* is :math:`\\beta` and
    *l2_weight* is :math:`\\alpha`.

    .. gdot::
        :script: DOT-SECTION

        from mlprodict.onnxrt import OnnxInference
        from onnxcustom.utils.onnx_function import function_onnx_graph

        model_onnx = function_onnx_graph('grad_penalty_elastic_error')
        oinf = OnnxInference(model_onnx, inplace=False)

        print("DOT-SECTION", oinf.to_dot())
    """
    from skl2onnx.algebra.onnx_ops import (
        OnnxMul, OnnxAdd, OnnxReduceSumSquare,
        OnnxReduceSum, OnnxSign, OnnxAbs, OnnxReshape)
    diff = 'X'
    abs_diff = OnnxAbs(diff, op_version=target_opset)
    res_l1 = OnnxReduceSum(abs_diff, op_version=target_opset)
    res2_l1 = OnnxSign(diff, op_version=target_opset)
    res_l2 = OnnxReduceSumSquare(diff, op_version=target_opset)
    res2_l2 = diff

    res = OnnxAdd(
        OnnxMul(res_l1, numpy.array([l1_weight], dtype=dtype),
                op_version=target_opset),
        OnnxMul(res_l2, numpy.array([l2_weight], dtype=dtype),
                op_version=target_opset),
        op_version=target_opset)
    res = OnnxReshape(res, numpy.array([-1], numpy.int64),
                      op_version=target_opset,
                      output_names=['Y'])

    res2 = OnnxAdd(
        OnnxMul(res2_l1, numpy.array([l1_weight], dtype=dtype),
                op_version=target_opset),
        OnnxMul(res2_l2, numpy.array([l2_weight * (2)], dtype=dtype),
                op_version=target_opset),
        op_version=target_opset, output_names=['Z'])

    var_type = dtype_to_var_type(dtype)
    varsx = [('X', var_type([None, None]))]
    onx = res.to_onnx(
        varsx, outputs=[('Y', var_type([None])), ('Z', var_type())],
        target_opset=target_opset, other_outputs=[res2])
    return onx


def _onnx_n_penalty_elastic_error(target_opset=None, dtype=numpy.float32,
                                  weight_name=None,
                                  l1_weight=0.01, l2_weight=0.01, n_tensors=1,
                                  loss_shape=(1, 1)):
    """
    Returns the ONNX graph for function
    :math:`Y = f(W) = \\beta \\lVert W \\rVert +
    \\alpha \\lVert W \\rVert^2`
    *l1_weight* is :math:`\\beta` and
    *l2_weight* is :math:`\\alpha`.
    It does that for *n_tensors* and adds all of the results
    to an input loss.

    .. gdot::
        :script: DOT-SECTION

        from mlprodict.onnxrt import OnnxInference
        from onnxcustom.utils.onnx_function import function_onnx_graph

        model_onnx = function_onnx_graph(
            'n_penalty_elastic_error', n_tensors=2)
        oinf = OnnxInference(model_onnx, inplace=False)

        print("DOT-SECTION", oinf.to_dot())
    """
    from skl2onnx.algebra.onnx_ops import (
        OnnxMul, OnnxAdd, OnnxReduceSumSquare,
        OnnxReduceSum, OnnxAbs, OnnxReshape)

    if n_tensors <= 0:
        raise ValueError(  # pragma: no cover
            "This function is useless if the number of tensors is null.")

    var_type = dtype_to_var_type(dtype)
    varsx = [('loss', var_type(loss_shape))]
    names = ['loss']
    for n in range(n_tensors):
        name = 'W%d' % n
        abs_diff = OnnxAbs(name, op_version=target_opset)
        res_l1 = OnnxReduceSum(abs_diff, op_version=target_opset)
        # res2_l1 = OnnxSign(diff, op_version=target_opset)
        res_l2 = OnnxReduceSumSquare(name, op_version=target_opset)
        # res2_l2 = diff
        res = OnnxAdd(
            OnnxMul(res_l1, numpy.array([l1_weight], dtype=dtype),
                    op_version=target_opset),
            OnnxMul(res_l2, numpy.array([l2_weight], dtype=dtype),
                    op_version=target_opset),
            op_version=target_opset)
        names.append(res)
        varsx.append(('W%d' % n, var_type()))

    if len(names) == 2:
        res = OnnxAdd(*names, op_version=target_opset)
    else:
        res = OnnxAdd(names[1], names[2], op_version=target_opset)
        for i in range(3, len(names)):
            res = OnnxAdd(res, names[i], op_version=target_opset)
        res = OnnxAdd(names[0], res, op_version=target_opset)

    res = OnnxReshape(res, numpy.array([-1], numpy.int64),
                      op_version=target_opset,
                      output_names=['Y'])
    onx = res.to_onnx(
        varsx, outputs=[('Y', var_type([None]))],
        target_opset=target_opset)
    return onx


def _onnx_update_penalty_elastic_error(target_opset=None, dtype=numpy.float32,
                                       l1=1e-4, l2=1e-4):
    """
    Returns the ONNX graph for function
    :math:`Y = f(W) = W - 2 \\beta W - \\alpha sign(W)`
    *l1* is :math:`\\beta` and
    *l2* is :math:`\\alpha`.

    .. gdot::
        :script: DOT-SECTION

        from mlprodict.onnxrt import OnnxInference
        from onnxcustom.utils.onnx_function import function_onnx_graph

        model_onnx = function_onnx_graph(
            'update_penalty_elastic_error')
        oinf = OnnxInference(model_onnx, inplace=False)

        print("DOT-SECTION", oinf.to_dot())
    """
    from skl2onnx.algebra.onnx_ops import (
        OnnxSub, OnnxMul, OnnxSign)

    res = OnnxSub(
        OnnxMul('X', numpy.array([1 - 2 * l2], dtype=dtype),
                op_version=target_opset),
        OnnxMul(OnnxSign('X', op_version=target_opset),
                numpy.array([l1], dtype=dtype),
                op_version=target_opset),
        op_version=target_opset,
        output_names=['Y'])

    var_type = dtype_to_var_type(dtype)
    varsx = [('X', var_type())]
    onx = res.to_onnx(
        varsx, outputs=[('Y', var_type())],
        target_opset=target_opset)
    return onx


def _onnx_grad_sigmoid_neg_log_loss_error(target_opset=None,
                                          dtype=numpy.float32,
                                          eps=1e-5,
                                          weight_name=None):
    """
    The function the raw scores from a classifier, uses the
    sigmoid function to compute probabilities, then the log function
    to compute the loss. It creates the ONNX graph for this function
    and the associated gradient of the loss against the raw scores.

    Probabilites (class 1): :math:`p(s) = \\frac{1}{1 + \\exp(-s)}`.
    Loss (for two classes): :math:`L(y, s) = (1 - y)\\log(1 - p(s)) +
    y \\log(p(s))`.
    Gradient :math:`\\frac{dL(y, s)}{ds} = y - p(s)`.
    To avoid nan values, probabilies are clipped:
    :math:`p(s) = \\max(\\min(p(s), 1 - \\epsilon), \\epsilon)`.
    :math:`y \\in \\{0, 1\\}` (integer). *s* is a float.

    :param eps: to clip probabilities and avoid computing `log(0)`

    .. gdot::
        :script: DOT-SECTION

        from mlprodict.onnxrt import OnnxInference
        from onnxcustom.utils.onnx_function import function_onnx_graph

        model_onnx = function_onnx_graph('grad_sigmoid_neg_log_loss_error')
        oinf = OnnxInference(model_onnx, inplace=False)

        print("DOT-SECTION", oinf.to_dot())
    """
    from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE
    from skl2onnx.algebra.onnx_ops import (
        OnnxSub, OnnxMul, OnnxSigmoid, OnnxLog, OnnxNeg,
        OnnxReduceSum, OnnxReshape, OnnxAdd, OnnxCast, OnnxClip)

    p1c = OnnxSigmoid('X2', op_version=target_opset)
    p1 = OnnxClip(p1c, numpy.array([eps], dtype=dtype),
                  numpy.array([1 - eps], dtype=dtype),
                  op_version=target_opset)
    p0 = OnnxSub(numpy.array([1], dtype=dtype), p1,
                 op_version=target_opset)
    y1 = OnnxCast('X1', to=NP_TYPE_TO_TENSOR_TYPE[numpy.dtype(dtype)],
                  op_version=target_opset)
    y0 = OnnxSub(numpy.array([1], dtype=dtype), y1,
                 op_version=target_opset)
    loss_obs = OnnxAdd(
        OnnxMul(y0, OnnxLog(p0, op_version=target_opset),
                op_version=target_opset),
        OnnxMul(y1, OnnxLog(p1, op_version=target_opset),
                op_version=target_opset),
        op_version=target_opset)

    loss_neg = OnnxNeg(loss_obs, op_version=target_opset)
    if weight_name is None:
        loss = OnnxReduceSum(loss_neg, op_version=target_opset)
        grad = OnnxSub(p1, y1, op_version=target_opset,
                       output_names=['Z'])
    else:
        loss = OnnxReduceSum(
            OnnxMul(loss_neg,
                    OnnxReshape(
                        weight_name, numpy.array([-1, 1], dtype=numpy.int64),
                        op_version=target_opset),
                    op_version=target_opset),
            op_version=target_opset)
        grad = OnnxMul(
            OnnxSub(p1, y1, op_version=target_opset),
            OnnxReshape(weight_name, numpy.array([-1, 1], dtype=numpy.int64),
                        op_version=target_opset),
            output_names=['Z'], op_version=target_opset)

    res = OnnxReshape(loss, numpy.array([-1], numpy.int64),
                      op_version=target_opset,
                      output_names=['Y'])

    var_type_int64 = dtype_to_var_type(numpy.int64)
    var_type = dtype_to_var_type(dtype)
    varsx = [('X1', var_type_int64([None, None])),
             ('X2', var_type([None, None]))]
    if weight_name is not None:
        varsx.append((weight_name, var_type([None])))
    onx = res.to_onnx(
        varsx, outputs=[('Y', var_type()), ('Z', var_type())],
        target_opset=target_opset, other_outputs=[grad])
    if weight_name is not None:
        onx = add_initializer(
            onx, weight_name, numpy.array([1], dtype=dtype))
    return onx
