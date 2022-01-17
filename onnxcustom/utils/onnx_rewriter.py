"""
@file
@brief Rewrites operator in ONNX graph.
"""
from onnx.helper import make_graph
from onnx import NodeProto
from onnx.numpy_helper import to_array, from_array
from mlprodict.onnx_tools.optim._onnx_optimisation_common import (
    _apply_remove_node_fct_node, _apply_optimisation_on_graph)


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


def _onnx_rewrite_operator_node(existing_names, node, sub_onx):
    """
    Replaces a node by a subgraph.

    :param existing_names: existing results names
    :param node: onnx node to replace
    :param sub_onx: onnx sub_graph to use as a replacement
    :return: new_initializer, new_nodes
    """
    if len(node.input) != len(sub_onx.graph.input):
        raise ValueError(
            "Mismatch with the number of inputs for operator type %r. "
            "%d != %d." % (
                node.op_type, len(node.input), len(sub_onx.graph.nput)))
    if len(node.output) != len(sub_onx.graph.output):
        raise ValueError(
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
    if hasattr(onx, 'graph'):
        fct = (lambda graph, recursive=False, debug_info=None:
               onnx_rewrite_operator(
                   graph, op_type, sub_onx, recursive=recursive,
                   debug_info=debug_info))
        return _apply_optimisation_on_graph(fct, onx, recursive=recursive)

    existing_names = set()
    for node in onx.node:
        existing_names.update(node.input)
        existing_names.update(node.output)

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
                onnx_rewrite_operator, node, recursive=True)

    graph = make_graph(
        new_nodes, onx.name, onx.input, onx.output, new_inits)
    return graph
