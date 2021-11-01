# coding: utf-8
"""
@file
@brief Experimentation with ONNX, examples.
"""

__version__ = "0.2.122"
__author__ = "Xavier Dupré, ..."
__max_supported_opset__ = 15  # Converters are tested up to this version.


def check(verbose=1):
    """
    Runs a couple of functions to check the module is working.

    :param verbose: 0 to hide the standout output
    :return: list of dictionaries, result of each test
    """
    tests = []
    try:
        import onnx
    except ImportError as e:  # pragma: no cover
        tests.append(dict(test='onnx', exc=e))
    try:
        import skl2onnx
    except ImportError as e:  # pragma: no cover
        tests.append(dict(test='skl2onnx', exc=e))
    try:
        import onnxruntime
    except ImportError as e:  # pragma: no cover
        tests.append(dict(test='onnxruntime', exc=e))
    try:
        import onnxruntime.training
    except ImportError as e:  # pragma: no cover
        tests.append(dict(test='onnxruntime_training', exc=e))
    return test
