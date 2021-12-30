# pylint: disable=C0415,E0611,E1101
"""
@file
@brief Onnx implementation of common functions used to train a model.
"""
import math
import numpy
from onnx import TensorProto, numpy_helper, helper


def onnx_rename_weights(onx):
    """
    Renames ONNX initialiers to make sure their name
    follows the alphabetical order. The model is
    modified inplace. This function calls
    :func:`onnx_rename_names
    <mlprodict.onnx_tools.onnx_manipulations.onnx_rename_names>`.

    :param onx: ONNX model
    :return: same model

    .. note::
        The function does not go into subgraphs.
    """
    from mlprodict.onnx_tools.onnx_manipulations import (  # pylint: disable=C0415
        onnx_rename_names)

    init = [init.name for init in onx.graph.initializer]
    ninit = max(1, int(math.log(len(init)) / math.log(10) + 1))
    fmt = "I%0{}d_%s".format(ninit)
    new_names = [fmt % (i, name) for i, name in enumerate(init)]
    repl = dict(zip(init, new_names))
    return onnx_rename_names(onx, recursive=False, replace=repl)


def get_onnx_opset(onx, domain=''):
    """
    Returns the opset associated to an opset.

    :param onx: onx graph
    :param domain: domain
    :return: value
    """
    for opset in onx.opset_import:
        if opset.domain == domain:
            return opset.version
    raise ValueError(
        "Unable to find opset for domain=%r." % domain)


def proto_type_to_dtype(proto_type):
    """
    Converts a ONNX TensorProto type into numpy type.

    :param proto_type: integer
    :return: proto type
    """
    if proto_type == TensorProto.FLOAT:
        return numpy.float32
    if proto_type == TensorProto.DOUBLE:
        return numpy.float64
    # Not efficient.
    if proto_type == 'tensor(float)':
        return numpy.float32
    if proto_type == 'tensor(double)':
        return numpy.float64
    raise ValueError(
        "Unexpected value proto_type=%r (type=%r)." % (
            proto_type, type(proto_type)))


def dtype_to_var_type(dtype):
    """
    Converts a numpy dtype into a var type.
    """
    from skl2onnx.common.data_types import FloatTensorType, DoubleTensorType
    if dtype == numpy.float32:
        return FloatTensorType
    if dtype == numpy.float64:
        return DoubleTensorType
    raise ValueError(
        "Unexpected value dtype=%r." % dtype)


def add_initializer(model, name, value):
    """
    Adds an initializer to graph.

    :param model: onnx model
    :param name: initializer name
    :param value: value
    :return: new ONNX graph
    """
    inits = set(i.name for i in model.graph.initializer)
    if name in inits:
        raise ValueError(
            "Name %r is already taken among %r." % (
                name, inits))
    list_inits = list(model.graph.initializer)
    list_inits.append(
        numpy_helper.from_array(value, name=name))
    graph_def = helper.make_graph(
        model.graph.node, model.graph.name,
        model.graph.input, model.graph.output,
        list_inits)
    onnx_model = helper.make_model(graph_def)
    onnx_model.ir_version = model.ir_version
    onnx_model.producer_name = model.producer_name
    onnx_model.producer_version = model.producer_version
    onnx_model.domain = model.domain
    onnx_model.model_version = model.model_version
    onnx_model.doc_string = model.doc_string
    if len(model.metadata_props) > 0:  # pragma: no cover
        values = {p.key: p.value for p in model.metadata_props}
        helper.set_model_props(onnx_model, values)

    del onnx_model.opset_import[:]  # pylint: disable=E1101
    for oimp in model.opset_import:
        op_set = onnx_model.opset_import.add()  # pylint: disable=E1101
        op_set.domain = oimp.domain
        op_set.version = oimp.version
    return onnx_model
