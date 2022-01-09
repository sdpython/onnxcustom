
Serialization
=============

.. contents::
    :local:

Load a model
++++++++++++

.. autofunction:: onnx.load

::

    from onnx import load

    onnx_model = load("model.onnx")

Or:

::

    from onnx import load

    with open("model.onnx", "rb") as f:
        onnx_model = load(f)

Save a model
++++++++++++

This ONNX graph needs to be serialized into one contiguous
memory buffer. Method `SerializeToString` is available
in every ONNX objects.

::

    with open("model.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())
