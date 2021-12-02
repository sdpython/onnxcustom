# pylint: disable=C0415,E0611,E1101
"""
@file
@brief Onnx implementation of common functions used to train a model.
"""
import math
import numpy
from onnx import TensorProto


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
    raise ValueError(
        "Unexpected value proto_type=%r." % proto_type)


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
