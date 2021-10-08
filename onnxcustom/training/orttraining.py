"""
@file
@brief Helper for :epkg:`onnxruntime-training`.
"""
from onnx.helper import (
    make_node, make_graph, make_model, make_tensor_value_info,
    set_model_props)
from onnx.numpy_helper import to_array


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


def add_loss_output(onx, score_name='squared_error',
                    loss_name='loss', label_name='label'):
    """
    Modifies an ONNX graph to add operators to score and allow training.

    :param onx: onx graph
    :param score_name: name of the score
    :param loss_name: name of the output loss
    :param label_name: name of the label input
    :return: modified graph

    Possible values for *score_name*:

    * `'squared_error'`: :math:`\\sum_i{(f(x_i)-y_i)^2}`

    See example :ref:`l-orttraining-nn-gpu`.
    """
    outputs = onx.graph.output
    if len(outputs) != 1:
        raise ValueError(
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

    if score_name == 'squared_error':
        diff_name = _unique_name(existing_names, "loss_diff")
        diff2_name = _unique_name(existing_names, "loss_diff")
        nodes = [make_node('Sub', [output_name, label_name], [diff_name]),
                 make_node('Mul', [diff_name, diff_name], [diff2_name]),
                 make_node('ReduceSum', [diff2_name], [loss_name])]

        inputs = [make_tensor_value_info('label', elem, shape)]
        outputs = [make_tensor_value_info('loss', elem, [1, 1])]
    else:
        raise NotImplementedError(
            "Unexpected %r value for score_name." % score_name)

    graph = make_graph(
        list(onx.graph.node) + nodes,
        onx.graph.name,
        list(onx.graph.input) + inputs,
        outputs + list(onx.graph.output),
        onx.graph.initializer)
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
    Returns the list of initializer to train.

    :return: dictionary `{name: (value, tensor)}`
    """
    res = {}
    for init in onx.graph.initializer:
        res[init.name] = (to_array(init), init)
    return res
