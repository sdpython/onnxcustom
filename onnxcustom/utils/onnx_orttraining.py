# pylint: disable=C0415
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


def _unique_name(existing_names, name, add=True):
    """
    Returns a name different from any name in *existing_names*.

    :param existing_names: set of names
    :param name: current
    :param add: add the name of the list of existing names
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


def add_loss_output(onx, score_name='squared_error',
                    loss_name='loss', label_name='label',
                    weight_name=None, **kwargs):
    """
    Modifies an ONNX graph to add operators to score and allow training.

    :param onx: onx graph
    :param score_name: name of the score
    :param loss_name: name of the output loss
    :param label_name: name of the label input
    :param weight_name: None or any value to consider weight
        while computing loss
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

    See example :ref:`l-orttraining-nn-gpu`.
    """
    outputs = onx.graph.output
    if len(outputs) != 1:
        raise ValueError(  # pragma: no cover
            "Unable to guess the output to compare to the "
            "expacted labels among %r." % (o.name for o in outputs))

    existing_names = []
    for node in onx.graph.node:
        existing_names.extend(node.output)
        existing_names.extend(node.input)
    existing_names = set(existing_names)

    output_name = onx.graph.output[0].name
    elem = onx.graph.output[0].type.tensor_type.elem_type
    shape = []
    for d in onx.graph.output[0].type.tensor_type.shape.dim:
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
    else:
        raise NotImplementedError(  # pragma: no cover
            "Unexpected %r value for score_name." % score_name)
    inits = list(onx.graph.initializer) + inits
    graph = make_graph(
        list(onx.graph.node) + nodes,
        onx.graph.name,
        list(onx.graph.input) + inputs,
        outputs + list(onx.graph.output),
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
    return onnx_model


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
