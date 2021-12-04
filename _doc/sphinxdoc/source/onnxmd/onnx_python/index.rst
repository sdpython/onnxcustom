
Useful functions and classes from onnx
======================================

This section gathers documentations from the most frequent
used functions or classes from :epkg:`onnx`. Next function returns
the opset of the installed version of package :epkg:`onnx`.

.. autofunction:: onnx.defs.onnx_opset_version

.. runpython::
    :showcode:

    from onnx import __version__
    from onnx.defs import onnx_opset_version
    print("onnx", __version__, "opset", onnx_opset_version())

Other functions are dispatched accress following sections.

.. toctree::
    :maxdepth: 1

    serialize
    helper
    numpy_helper
    classes
    shape_inference
    potting
    spec
    hub
