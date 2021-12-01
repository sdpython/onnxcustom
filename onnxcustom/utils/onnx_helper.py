"""
@file
@brief Onnx helper.
"""
import math


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
