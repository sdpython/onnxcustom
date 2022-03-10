# coding: utf-8
# flake8: noqa: F401
# pylint: disable=W0611,C0415
"""
@file
@brief Experimentation with ONNX, examples.
"""

__version__ = "0.4.322"
__author__ = "Xavier Dupré, ..."
__max_supported_opset__ = 15  # Converters are tested up to this version.
__max_supported_opsets__ = {
    '': __max_supported_opset__,
    'ai.onnx.ml': 2}


def check(verbose=1):
    """
    Runs a couple of functions to check the module is working.

    :param verbose: 0 to hide the standout output
    :return: list of dictionaries, result of each test
    """
    tests = []
    try:
        import onnx
        import onnx.helper
    except ImportError as e:  # pragma: no cover
        tests.append(dict(test='onnx', exc=e))
    try:
        import onnxruntime
        from onnxruntime import InferenceSession
    except ImportError as e:  # pragma: no cover
        tests.append(dict(test='onnxruntime', exc=e))
    try:
        import onnxruntime.training
        from onnxruntime.training import TrainingSession
    except ImportError as e:  # pragma: no cover
        tests.append(dict(test='onnxruntime_training', exc=e))
    try:
        import skl2onnx
        from skl2onnx import to_onnx
    except ImportError as e:  # pragma: no cover
        tests.append(dict(test='skl2onnx', exc=e))
    return tests


def get_max_opset():
    """
    Returns the highest available onnx opset version.
    """
    from onnx.defs import onnx_opset_version
    return min(onnx_opset_version(), __max_supported_opset__)
