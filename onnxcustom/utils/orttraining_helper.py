# pylint: disable=C0415,E1101
"""
@file
@brief ONNX manipulations to help build ONNX gradient graphs.
"""
from collections import OrderedDict
import numpy
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
from onnx.numpy_helper import to_array, from_array
from onnx.helper import (
    make_node, make_graph, make_model, make_tensor_value_info,
    set_model_props)
from onnx import TensorProto


def _unique_name(existing_names, name):
    """
    Returns a name different from any name in *existing_names*.

    :param existing_names: set of names
    :param name: current
    :return: unique name
    """
    if name not in existing_names:
        existing_names.add(name)
        return name
    name0 = name
    i = 2
    while name in existing_names:
        name = "%s_%d" % (name0, i)
        i += 1
    existing_names.add(name)
    return name


def _loss_l1(existing_names, elem, shape,
             output_name, label_name,
             weight_name, loss_name):
    """
    Implements loss l1.
    """
    diff_name = _unique_name(existing_names, "loss_diff")
    diff2_name = _unique_name(existing_names, "loss_diff")
    nodes = [make_node('Sub', [output_name, label_name], [diff_name]),
             make_node('Abs', [diff_name], [diff2_name])]
    if weight_name is not None:
        res_name = _unique_name(existing_names, "loss_diff_weight")
        nodes.append(
            make_node('Mul', [diff2_name, weight_name], [res_name]))
    else:
        res_name = diff2_name
    nodes.append(make_node('ReduceSum', [res_name], [loss_name]))

    inputs = [make_tensor_value_info(label_name, elem, shape)]
    if weight_name is not None:
        inputs.append(
            make_tensor_value_info(weight_name, elem, [shape[0]]))
    return (
        [], inputs, nodes,
        [make_tensor_value_info(loss_name, elem, [1, 1])])


def _loss_l2(existing_names, elem, shape,
             output_name, label_name,
             weight_name, loss_name):
    """
    Implements loss l2.
    """
    diff_name = _unique_name(existing_names, "loss_diff")
    diff2_name = _unique_name(existing_names, "loss_diff")
    nodes = [make_node('Sub', [output_name, label_name], [diff_name]),
             make_node('Mul', [diff_name, diff_name], [diff2_name])]
    if weight_name is not None:
        res_name = _unique_name(existing_names, "loss_diff_weight")
        nodes.append(
            make_node('Mul', [diff2_name, weight_name], [res_name]))
    else:
        res_name = diff2_name
    nodes.append(make_node('ReduceSum', [res_name], [loss_name]))

    inputs = [make_tensor_value_info(label_name, elem, shape)]
    if weight_name is not None:
        inputs.append(
            make_tensor_value_info(weight_name, elem, [shape[0]]))
    return (
        [], inputs, nodes,
        [make_tensor_value_info(loss_name, elem, [1, 1])])


def _loss_elastic(existing_names, elem, shape,
                  output_name, label_name,
                  weight_name, loss_name,
                  l1_weight=0.5, l2_weight=0.5):
    """
    Implements mixture of losses l1 and l2.
    """
    l1_name = _unique_name(existing_names, "l1_name")
    l2_name = _unique_name(existing_names, "l2_name")
    dtype = TENSOR_TYPE_TO_NP_TYPE[elem]
    onx_l1_weight = from_array(
        numpy.array([l1_weight], dtype=dtype), name=l1_name)
    onx_l2_weight = from_array(
        numpy.array([l2_weight], dtype=dtype), name=l2_name)
    inits = [onx_l1_weight, onx_l2_weight]

    diff_name = _unique_name(existing_names, "loss_diff")
    diff1_name = _unique_name(existing_names, "loss_l1")
    diff2_name = _unique_name(existing_names, "loss_l2")
    wl1_name = _unique_name(existing_names, "loss_l1")
    wl2_name = _unique_name(existing_names, "loss_l2")
    final_loss = _unique_name(existing_names, "final_loss")
    nodes = [make_node('Sub', [output_name, label_name], [diff_name]),
             make_node('Mul', [diff_name, diff_name], [diff2_name]),
             make_node('Abs', [diff_name], [diff1_name]),
             make_node('Mul', [diff1_name, l1_name], [wl1_name]),
             make_node('Mul', [diff2_name, l2_name], [wl2_name]),
             make_node('Add', [wl1_name, wl2_name], [final_loss]),
             ]
    if weight_name is not None:
        res_name = _unique_name(existing_names, "loss_diff_weight")
        nodes.append(
            make_node('Mul', [final_loss, weight_name], [res_name]))
    else:
        res_name = final_loss
    nodes.append(make_node('ReduceSum', [res_name], [loss_name]))

    inputs = [make_tensor_value_info(label_name, elem, shape)]
    if weight_name is not None:
        inputs.append(
            make_tensor_value_info(weight_name, elem, [shape[0]]))
    return (
        inits, inputs, nodes,
        [make_tensor_value_info(loss_name, elem, [1, 1])])


def _loss_log(existing_names, elem, shape,
              output_name, label_name,
              weight_name, loss_name,
              eps=1e-6):
    """
    This only works for a binary classification.
    The log loss is `'log(yt, yp) = (1-yt)\\log(1-yp) - yt\\log(yp)`,
    this only works for a binary classification where *yp* is the
    predicted probability, *yt* is the expected probability.
    *yt* is expected to be binary, *yp* is a matrix with two
    columns, the sum on every line is 1.
    Parameter *eps* is used to avoid computing *log(0)*.
    """
    if output_name == 'output_label':
        raise RuntimeError(  # pragma: no cover
            "output_name=%r, log loss does not work on labels."
            "" % output_name)
    dtype = TENSOR_TYPE_TO_NP_TYPE[elem]
    one_name = _unique_name(existing_names, "one_name")
    eps_name = _unique_name(existing_names, "eps_name")
    eps1_name = _unique_name(existing_names, "eps1_name")
    axes_name = _unique_name(existing_names, "axes_name")

    eps_init = from_array(numpy.array([eps], dtype=dtype), name=eps_name)
    one_init = from_array(numpy.array([1], dtype=dtype), name=one_name)
    eps1_init = from_array(
        numpy.array([1 - eps], dtype=dtype), name=eps1_name)
    axes_init = from_array(
        numpy.array([1], dtype=numpy.int64), name=axes_name)

    clip_name = _unique_name(existing_names, "clip_name")
    clip_red_name = _unique_name(existing_names, "clip_red_name")
    new_output_name = _unique_name(existing_names, "new_output_name")
    cast_name = _unique_name(existing_names, "cast_name")
    log_name = _unique_name(existing_names, "log_name")
    subl_name = _unique_name(existing_names, "subl_name")
    conc_name = _unique_name(existing_names, "conc_name")
    mul_name = _unique_name(existing_names, "mul_name")
    like_name = _unique_name(existing_names, "like_name")

    nodes = [
        make_node(
            'Clip', [output_name, eps_name, eps1_name], [clip_name]),
        make_node(
            'ReduceSum', [clip_name, axes_name], [clip_red_name], keepdims=1),
        make_node('Div', [clip_name, clip_red_name], [new_output_name]),
        make_node('Log', [new_output_name], [log_name]),
        make_node('Cast', [label_name], [cast_name], to=elem),
        make_node('Sub', [one_name, cast_name], [subl_name]),
        make_node('Concat', [subl_name, cast_name], [conc_name], axis=1),
        make_node('Mul', [log_name, conc_name], [mul_name]),
        make_node(
            'ReduceSum', [mul_name, axes_name], [like_name], keepdims=1)]

    inputs = [make_tensor_value_info(label_name, TensorProto.INT64, shape)]

    if weight_name is not None:
        inputs.append(
            make_tensor_value_info(weight_name, elem, [shape[0]]))
        likew_name = _unique_name(existing_names, "likew_name")
        nodes.append(
            make_node('Mul', [like_name, weight_name], [likew_name]))
        like_name = likew_name

    shape_name = _unique_name(existing_names, "shape_name")
    onx_shape = from_array(
        numpy.array([1, 1], dtype=numpy.int64), name=shape_name)
    reduced_loss = _unique_name(existing_names, "reduced_loss")
    neg_reduced_loss = _unique_name(existing_names, "neg_reduced_loss")
    nodes.extend([
        make_node('ReduceMean', [like_name], [reduced_loss]),
        make_node('Neg', [reduced_loss], [neg_reduced_loss]),
        make_node('Reshape', [neg_reduced_loss, shape_name], [loss_name])])

    return (
        [eps_init, eps1_init, one_init, axes_init, onx_shape],
        inputs, nodes, [make_tensor_value_info(loss_name, elem, [1, 1])])


def penalty_loss_onnx(name, dtype, l1=None, l2=None, existing_names=None):
    """
    Returns onnx nodes to compute
    :math:`|w| \\alpha + w^2 \\beta`
    where :math:`\\alpha=l1` and :math:`\\beta=l2`.

    :param name: name of weights
    :param dtype: numpy dtype
    :param l1: coefficient for L1 norm
    :param l2: coefficient for L2 norm
    :param existing_names: names already taken in the ONNX graph
    :return: initializer, nodes
    """
    suffix = name
    cst_shape = _unique_name(existing_names, "shape_%s" % suffix)
    new_name = _unique_name(existing_names, "reshaped_%s" % suffix)
    inits = [from_array(
             numpy.array([-1], dtype=numpy.int64), name=cst_shape)]
    nodes = [make_node('Reshape', [name, cst_shape], [new_name])]
    name = new_name

    if l1 is None or l1 == 0:
        if l2 is None or l2 == 0:
            raise ValueError(  # pragma: no cover
                "l1 and l2 cannot be null or None at the same time, "
                "name=%r." % name)
        l2_name = _unique_name(existing_names, "l2_weight_%s" % suffix)
        inits.extend([from_array(
            numpy.array([l2], dtype=dtype), name=l2_name)])
        mul_name = _unique_name(existing_names, "reduced0_%s" % suffix)
        red_name = _unique_name(existing_names, "reduced_%s" % suffix)
        pen_name = _unique_name(existing_names, "penalty_%s" % suffix)
        nodes.extend([
            make_node('Mul', [name, name], [mul_name]),
            make_node('ReduceSum', [mul_name], [red_name]),
            make_node('Mul', [red_name, l2_name], [pen_name])])
        return inits, nodes

    if l2 is None or l2 == 0:
        l1_name = _unique_name(existing_names, "l1_weight_%s" % suffix)
        inits.extend([from_array(
            numpy.array([l1], dtype=dtype), name=l1_name)])
        red_name = _unique_name(existing_names, "reduced_%s" % suffix)
        abs_name = _unique_name(existing_names, "absolute_%s" % suffix)
        pen_name = _unique_name(existing_names, "penalty_%s" % suffix)
        nodes.extend([
            make_node('Abs', [name], [abs_name]),
            make_node('ReduceSum', [abs_name], [red_name]),
            make_node('Mul', [red_name, l1_name], [pen_name])])
        return inits, nodes

    l1_name = _unique_name(existing_names, "l1_weight_%s" % suffix)
    l2_name = _unique_name(existing_names, "l2_weight_%s" % suffix)
    inits.extend([
        from_array(numpy.array([l1], dtype=dtype), name=l1_name),
        from_array(numpy.array([l2], dtype=dtype), name=l2_name)])

    red_name1 = _unique_name(existing_names, "reduced1_%s" % suffix)
    mul_name = _unique_name(existing_names, "reducedm_%s" % suffix)
    red_name2 = _unique_name(existing_names, "reduced2_%s" % suffix)
    abs_name = _unique_name(existing_names, "absolute_%s" % suffix)
    pen_name1 = _unique_name(existing_names, "penalty1_%s" % suffix)
    pen_name2 = _unique_name(existing_names, "penalty2_%s" % suffix)
    pen_name = _unique_name(existing_names, "penalty_%s" % suffix)
    nodes.extend([
        make_node('Mul', [name, name], [mul_name]),
        make_node('ReduceSum', [mul_name], [red_name2]),
        make_node('Mul', [red_name2, l2_name], [pen_name2]),
        make_node('Abs', [name], [abs_name]),
        make_node('ReduceSum', [abs_name], [red_name1]),
        make_node('Mul', [red_name1, l1_name], [pen_name1]),
        make_node('Add', [pen_name1, pen_name2], [pen_name])])

    return inits, nodes


def get_train_initializer(onx):
    """
    Returns the list of initializers to train.

    :return: dictionary `{name: (value, tensor)}`

    The function walk through the list of initializers and
    returns all tensors with elements from types float or double.
    """
    res = OrderedDict()
    for init in onx.graph.initializer:
        if init.data_type in (
                TensorProto.FLOAT16,  # pylint: disable=E1101
                TensorProto.FLOAT,  # pylint: disable=E1101
                TensorProto.DOUBLE):  # pylint: disable=E1101
            res[init.name] = (to_array(init), init)
    return res


def _rewrite_op_no_grad(onx):
    """
    Rewrites operators with no gradient.
    """
    set_types = set(n.op_type for n in onx.graph.node)
    if "Reciprocal" in set_types:
        from skl2onnx.algebra.onnx_ops import OnnxDiv  # pylint: disable=E0611
        from skl2onnx.common.data_types import FloatTensorType
        from .onnx_rewriter import onnx_rewrite_operator

        opset = None
        for op in onx.opset_import:
            if op.domain in ('', 'ai.onnx'):
                opset = op.version
        if opset is None:  # pragma: no cover
            from .. import get_max_opset
            opset = get_max_opset()

        node = OnnxDiv(numpy.array([1], dtype=numpy.float32),
                       'X', output_names=['Y'],
                       op_version=opset)
        rewrite_onx = node.to_onnx(
            inputs={'X': FloatTensorType()},
            outputs={'Y': FloatTensorType()},
            target_opset=opset)
        onx = onnx_rewrite_operator(onx, 'Reciprocal', rewrite_onx)

    return onx


def add_loss_output(onx, score_name='squared_error',
                    loss_name='loss', label_name='label',
                    weight_name=None, penalty=None,
                    output_index=None, **kwargs):
    """
    Modifies an ONNX graph to add operators to score and allow training.

    :param onx: onx graph
    :param score_name: name of the score
    :param loss_name: name of the output loss
    :param label_name: name of the label input
    :param weight_name: None or any value to consider weight
        while computing loss
    :param penalty: dictionary similar to the
        following one `{ weight_name: {'l1': alpha, 'l2': beta} }`
        or `{ weight_name: beta}`,
        it adds a L1 and/or L2 penalty to one input or initializer,
        penalty = :math:`|w| \\alpha + w^2 \\beta`
    :param output_index: the output used to compute the loss,
        if None, the function assumes there is only one output,
        it must be specified if there are more than 1,
        it can be an integer or a string (output name)
    :param kwargs: additional arguments for losses (see below)
    :return: modified graph

    Possible values for *score_name*:

    * `'squared_error'` or `'l2`': :math:`\\sum_i{(f(x_i)-y_i)^2}` or
      :math:`\\sum_i{w_i (f(x_i)-y_i)^2}` if *weight_name*
      is not None
    * `'absolute_error'` or `'l1`': :math:`\\sum_i{|f(x_i)-y_i|}` or
      :math:`\\sum_i{w_i |f(x_i)-y_i|}` if *weight_name*
      is not None
    * `'elastic'`: mixture of losses, kwargs must define
      *l1_weight* and *l2_weight*, undefined, default value are 0.5
    * `'log(yt, yp)'`: log loss :math:`(1-yt)\\log(1-yp) - yt\\log(yp)`,
        this only works for a binary classification where *yp* is the
        predicted probability, *yt* is the expected probability.
        *yt* is expected to be binary, *yp* is a matrix with two
        columns, the sum on every line is 1.

    See example :ref:`l-orttraining-nn-gpu`.
    Next example shows the loss with L1 and L2 loss.

    .. gdot::
        :script: DOT-SECTION

        import numpy
        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from mlprodict.onnx_conv import to_onnx
        from mlprodict.onnxrt import OnnxInference
        from onnxcustom import __max_supported_opset__ as opset
        from onnxcustom.utils.orttraining_helper import add_loss_output
        from onnxcustom.training.optimizers import OrtGradientOptimizer

        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=10, bias=2, random_state=0)
        X = X.astype(numpy.float32)
        y = y.astype(numpy.float32)
        w = (numpy.random.rand(y.shape[0]) + 1).astype(X.dtype)
        X_train, _, y_train, __, w_train, ___ = train_test_split(X, y, w)
        reg = LinearRegression()
        reg.fit(X_train, y_train, sample_weight=w_train)
        reg.coef_ = reg.coef_.reshape((1, -1))
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'})

        onx_loss = add_loss_output(
            onx, weight_name='weight', score_name='elastic',
            l1_weight=0.1, l2_weight=0.9)

        print("DOT-SECTION", OnnxInference(onx_loss).to_dot())

    Next example shows how to add a L2 loss with L1 and L2 penalties
    on the coefficients.

    .. gdot::
        :script: DOT-SECTION

        import numpy
        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from mlprodict.onnx_conv import to_onnx
        from mlprodict.onnxrt import OnnxInference
        from onnxcustom import __max_supported_opset__ as opset
        from onnxcustom.utils.orttraining_helper import add_loss_output
        from onnxcustom.training.optimizers import OrtGradientOptimizer

        X, y = make_regression(  # pylint: disable=W0632
            100, n_features=10, bias=2, random_state=0)
        X = X.astype(numpy.float32)
        y = y.astype(numpy.float32)
        w = (numpy.random.rand(y.shape[0]) + 1).astype(X.dtype)
        X_train, _, y_train, __, w_train, ___ = train_test_split(X, y, w)
        reg = LinearRegression()
        reg.fit(X_train, y_train, sample_weight=w_train)
        reg.coef_ = reg.coef_.reshape((1, -1))
        onx = to_onnx(reg, X_train, target_opset=opset,
                      black_op={'LinearRegressor'})

        onx_loss = add_loss_output(
            onx, weight_name='weight', score_name='elastic',
            penalty={'coef': {'l1': 0.5, 'l2':0.5},
                     'intercept': {'l1': 0.5, 'l2':0.5}})

        print("DOT-SECTION", OnnxInference(onx_loss).to_dot())
    """
    from mlprodict.onnx_tools.optim import onnx_remove_node_unused

    # rename every intermediate output call label
    def _replace(ens):
        for i in range(len(ens)):  # pylint: disable=C0200
            if ens[i] == 'label':
                ens[i] = '_label_'

    for node in onx.graph.node:
        if "_label_" in node.input or "_label_" in node.output:
            raise RuntimeError(  # pragma: no cover
                "One intermediate result contains '_label_'. "
                "It should be removed manually.\n%r" % node)
        _replace(node.input)
        _replace(node.output)

    if output_index is None:
        if len(onx.graph.output) != 1:
            raise ValueError(  # pragma: no cover
                "Unable to guess the output to compare to the "
                "expacted labels among %r." % (
                    [o.name for o in onx.graph.output]))
        outputs = onx.graph.output
        output_index = 0
    elif isinstance(output_index, int):
        outputs = [onx.graph.output[output_index]]
    elif isinstance(output_index, str):
        outputs = [(i, o) for i, o in enumerate(onx.graph.output)
                   if o.name == output_index]
        if len(outputs) != 1:
            raise ValueError(  # pragma: no cover
                "Unable to find output %r in %r." % (
                    output_index, [o.name for o in onx.graph.output]))
        output_index = outputs[0][0]
        outputs = [outputs[0][1]]
    else:
        raise TypeError(  # pragma: no cover
            "output_index must be an integer or a str not %r."
            "" % type(output_index))

    existing_names = []
    for node in onx.graph.node:
        existing_names.extend(node.output)
        existing_names.extend(node.input)
    existing_names = set(existing_names)

    output_onx = onx.graph.output[output_index]
    output_name = output_onx.name
    elem = output_onx.type.tensor_type.elem_type
    if elem == 0:
        raise TypeError(  # pragma: no cover
            "Unable to guess input tensor type from %r."
            "" % output_onx)
    shape = []
    for d in output_onx.type.tensor_type.shape.dim:
        shape.append(d.dim_value if d.dim_value > 0 else None)

    if score_name in ('squared_error', 'l2'):
        inits, inputs, nodes, outputs = _loss_l2(
            existing_names, elem, shape, output_name, label_name,
            weight_name, loss_name)
    elif score_name in ('absolute_error', 'l1'):
        inits, inputs, nodes, outputs = _loss_l1(
            existing_names, elem, shape, output_name, label_name,
            weight_name, loss_name)
    elif score_name == 'elastic':
        inits, inputs, nodes, outputs = _loss_elastic(
            existing_names, elem, shape, output_name, label_name,
            weight_name, loss_name, **kwargs)
    elif score_name == 'log':
        shape = (None, 1)
        inits, inputs, nodes, outputs = _loss_log(
            existing_names, elem, shape, output_name, label_name,
            weight_name, loss_name, **kwargs)
    else:
        raise NotImplementedError(  # pragma: no cover
            "Unexpected %r value for score_name." % score_name)

    if penalty is not None:
        final_name = nodes[-1].output[0]
        loss_name = _unique_name(existing_names, "loss_diff")
        nodes[-1].output[0] = loss_name
        names = []
        for k, v in penalty.items():
            if isinstance(v, float):
                v = {'l2': v}
            inits_to_add, nodes_to_add = penalty_loss_onnx(
                k, dtype=TENSOR_TYPE_TO_NP_TYPE[elem],
                existing_names=existing_names, **v)
            names.append(nodes_to_add[-1].output[0])
            nodes.extend(nodes_to_add)
            inits.extend(inits_to_add)
        # Operator Sum does not have a gradient.
        if len(names) == 1:
            pen_name = names[0]
        else:
            current = names[0]
            for i in range(1, len(names)):
                new_name = _unique_name(existing_names, "sumop")
                nodes.append(
                    make_node('Add', [current, names[i]], [new_name]))
                current = new_name
            pen_name = current

        cst_shape = _unique_name(existing_names, "shapevect")
        inits.append(from_array(
            numpy.array([-1, 1], dtype=numpy.int64), name=cst_shape))
        loss_reshape = _unique_name(existing_names, "loss_reshape")
        pen_reshape = _unique_name(existing_names, "penalty_reshape")
        nodes.extend([
            make_node("Reshape", [pen_name, cst_shape], [pen_reshape]),
            make_node("Reshape", [loss_name, cst_shape], [loss_reshape])])

        nodes.append(
            make_node('Add', [pen_reshape, loss_reshape], [final_name]))

    inits = list(onx.graph.initializer) + inits
    graph = make_graph(
        list(onx.graph.node) + nodes,
        onx.graph.name,
        list(onx.graph.input) + inputs,
        outputs + [onx.graph.output[output_index]],
        inits)
    onnx_model = make_model(graph)
    onnx_model.ir_version = onx.ir_version
    onnx_model.producer_name = onx.producer_name
    onnx_model.producer_version = onx.producer_version
    onnx_model.domain = onx.domain
    onnx_model.model_version = onx.model_version
    onnx_model.doc_string = onx.doc_string
    if len(onx.metadata_props) > 0:
        values = {p.key: p.value for p in onx.metadata_props}
        set_model_props(onnx_model, values)

    # fix opset import
    del onnx_model.opset_import[:]  # pylint: disable=E1101
    for oimp in onx.opset_import:
        op_set = onnx_model.opset_import.add()  # pylint: disable=E1101
        op_set.domain = oimp.domain
        op_set.version = oimp.version
    return _rewrite_op_no_grad(onnx_remove_node_unused(onnx_model))
