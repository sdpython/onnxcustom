"""
@file
@brief Rewrites operator in ONNX graph.
"""
from onnx.helper import (
    make_graph, make_node, make_tensor_value_info, make_model)
from onnx import NodeProto
from onnx.numpy_helper import to_array, from_array


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


def _existing_names(onx):
    """
    Makes the list of existing names.
    Returns a set of unique names including
    intermediate results.
    """
    existing_names = set()
    graph = onx.graph if hasattr(onx, 'graph') else onx
    for node in graph.node:
        existing_names.update(node.input)
        existing_names.update(node.output)
    return existing_names


def _onnx_rewrite_operator_node(existing_names, node, sub_onx):
    """
    Replaces a node by a subgraph.

    :param existing_names: existing results names
    :param node: onnx node to replace
    :param sub_onx: onnx sub_graph to use as a replacement
    :return: new_initializer, new_nodes
    """
    if len(node.input) != len(sub_onx.graph.input):
        raise ValueError(  # pragma: no cover
            "Mismatch with the number of inputs for operator type %r. "
            "%d != %d." % (
                node.op_type, len(node.input), len(sub_onx.graph.nput)))
    if len(node.output) != len(sub_onx.graph.output):
        raise ValueError(  # pragma: no cover
            "Mismatch with the number of outputs for operator type %r. "
            "%d != %d." % (
                node.op_type, len(node.output), len(sub_onx.graph.output)))
    replaces = {}
    for inp, name in zip(sub_onx.graph.input, node.input):
        replaces[inp.name] = name
    for inp, name in zip(sub_onx.graph.output, node.output):
        replaces[inp.name] = name

    new_inits = []
    for init in sub_onx.graph.initializer:
        name = _unique_name(existing_names, init.name)
        replaces[init.name] = name
        tensor = from_array(to_array(init), name=name)
        new_inits.append(tensor)

    new_nodes = []
    for n in sub_onx.graph.node:
        new_node = NodeProto()
        new_node.op_type = n.op_type
        new_node.attribute.extend(n.attribute)  # pylint: disable=E1101
        new_node.input.extend(  # pylint: disable=E1101
            [replaces[i] for i in n.input])  # pylint: disable=E1101
        new_node.domain = n.domain
        new_out = []
        for o in n.output:
            if o in replaces:
                new_out.append(replaces[o])
            else:
                n = _unique_name(existing_names, o)
                new_out.append(n)
        new_node.output.extend(new_out)  # pylint: disable=E1101
        new_nodes.append(new_node)

    return new_inits, new_nodes


def onnx_rewrite_operator(onx, op_type, sub_onx, recursive=True, debug_info=None):
    """
    Replaces one operator by an onnx graph.

    :param onx: onnx graph
    :param op_type: operator type
    :param sub_onx: onnx graph
    :param recursive: looks into subgraphs
    :param debug_info: unused
    :return: modified onnx graph

    .. runpython::
        :showcode:

        import numpy
        from skl2onnx.common.data_types import FloatTensorType
        from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
            OnnxReciprocal, OnnxDiv)
        from mlprodict.plotting.text_plot import onnx_simple_text_plot
        from onnxcustom import get_max_opset
        from onnxcustom.utils.onnx_rewriter import onnx_rewrite_operator

        # first graph: it contains the node to replace
        opset = get_max_opset()
        node1 = OnnxReciprocal('X', output_names=['Y'],
                               op_version=opset)
        onx1 = node1.to_onnx(
            inputs={'X': FloatTensorType()},
            outputs={'Y': FloatTensorType()},
            target_opset=opset)

        # second graph: it contains the replacement graph
        node2 = OnnxDiv(numpy.array([1], dtype=numpy.float32),
                        'X', output_names=['Y'],
                        op_version=opset)
        onx2 = node2.to_onnx(
            inputs={'X': FloatTensorType()},
            outputs={'Y': FloatTensorType()},
            target_opset=opset)

        # third graph: the modified graph
        onx3 = onnx_rewrite_operator(onx1, 'Reciprocal', onx2)
        print(onnx_simple_text_plot(onx3))
    """
    from mlprodict.onnx_tools.optim._onnx_optimisation_common import (  # pylint: disable=C0415
        _apply_remove_node_fct_node, _apply_optimisation_on_graph)

    if hasattr(onx, 'graph'):
        fct = (lambda graph, recursive=False, debug_info=None:
               onnx_rewrite_operator(
                   graph, op_type, sub_onx, recursive=recursive,
                   debug_info=debug_info))
        return _apply_optimisation_on_graph(fct, onx, recursive=recursive)

    existing_names = _existing_names(onx)
    nodes = list(onx.node)
    new_nodes = []
    new_inits = list(onx.initializer)
    for i, node in enumerate(nodes):
        if node.op_type != op_type:
            new_nodes.append(node)
            continue
        inits, newn = _onnx_rewrite_operator_node(
            existing_names, node, sub_onx)
        new_inits.extend(inits)
        new_nodes.extend(newn)

    if recursive:
        # Handles subgraphs.
        for i in range(len(new_nodes)):  # pylint: disable=C0200
            node = nodes[i]
            if node is None or not (node.attribute):  # pylint: disable=C0325
                continue
            nodes[i] = _apply_remove_node_fct_node(
                onnx_rewrite_operator, node, recursive=True,
                debug_info=None)

    graph = make_graph(
        new_nodes, onx.name, onx.input, onx.output, new_inits)
    return graph


def unreduced_onnx_loss(onx, output_name='score'):
    """
    Every loss function reduces the results to compute a loss.
    The score function needs to get the loss for every observation,
    not the whole loss. This function looks for a reducing node
    and removes it before exposing the output as the only output.

    :param onx: onx graph
    :param output_name: new output name
    :return: new onx graph
    """
    from mlprodict.onnx_tools.onnx_manipulations import (  # pylint: disable=C0415
        select_model_inputs_outputs)

    graph = onx.graph
    found = []
    for node in graph.node:
        if node.op_type.startswith('Reduce'):
            found.append(node)
    if len(found) != 1:
        raise RuntimeError(  # pragma: no cover
            "Unable to find one unique Reducing node but found %d - %r."
            "" % (len(found), [(n.op_type, n.name) for n in found]))
    node = found[0]
    input_name = node.input[0]
    new_onx = select_model_inputs_outputs(
        onx, outputs=[input_name], infer_shapes=True)

    inits = new_onx.graph.initializer
    inputs = new_onx.graph.input  # pylint: disable=E1101
    existing_names = _existing_names(new_onx)
    new_name = _unique_name(existing_names, output_name)
    new_nodes = list(new_onx.graph.node)  # pylint: disable=E1101
    elem = graph.output[0].type.tensor_type.elem_type
    new_output = [make_tensor_value_info(new_name, elem, [None, 1])]

    if node.op_type == "ReduceSumSquare":
        new_node = make_node('Mul', [input_name, input_name], [new_name])
        new_nodes.append(new_node)
    elif node.op_type == 'ReduceSum':
        new_node = make_node('Identity', [input_name], [new_name])
        new_nodes.append(new_node)
    else:
        raise RuntimeError(  # pragma: no cover
            "Unable to unreduce node %r." % node.op_type)

    graph = make_graph(
        new_nodes, graph.name, inputs, new_output, inits)
    new_model = make_model(graph)
    new_model.ir_version = onx.ir_version
    new_model.producer_name = onx.producer_name
    new_model.producer_version = onx.producer_version
    new_model.domain = onx.domain
    new_model.model_version = onx.model_version
    new_model.doc_string = onx.doc_string
    if hasattr(onx, 'value_info'):
        graph.value_info.extend(onx.value_info)  # pylint: disable=E1101
    del new_model.opset_import[:]  # pylint: disable=E1101
    for oimp in onx.opset_import:
        op_set = new_model.opset_import.add()  # pylint: disable=E1101
        op_set.domain = oimp.domain
        op_set.version = oimp.version
    return new_model
