"""
@file
@brief Helpers to split an ONNX models.
"""
from onnx import ModelProto


def split_onnx(onnx_model, n_parts):
    """
    Splits an ONNX model into *n_parts* consecutive subgraphs.
    Chained altogether, they are equivalent to the given model.

    :param onnx_model: onnx model
    :param n_parts: number of subgraphs
    :return: list of onnx model
    """
    if not isinstance(onnx_model, ModelProto):
        raise TypeError(f"onnx_model must a ModelProto not a {type(model)}.")
    backwards = {}
    outputs = {}
    node_list = [(i, node) for i, node in enumerate(onnx_model.graph.node)]
    for idn, node in node_list:
        back = set()
        for i in node.input:
            back.add(i)
            if i in outputs:
                back |= backwards[outputs[i]]
        backwards[idn] = back
        for i in node.output:
            outputs[i] = idn

    forwards = {}
    inputs = {}    
    for idn, node in node_list[::-1]:
        forw = set()
        for i in node.output:
            forw.add(i)
            if i in inputs:
                forw |= forwards[inputs[i]]
        forwards[idn] = forw
        for i in node.input:
            inputs[i] = idn

    import pprint
    pprint.pprint(backwards)
    pprint.pprint(outputs)
    pprint.pprint(forwards)
    pprint.pprint(inputs)
