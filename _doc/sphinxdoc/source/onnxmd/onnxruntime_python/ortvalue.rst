
OrtValue
========

:epkg:`onnxruntime` implements tensors with class :epkg:`OrtValue`.
It has the same properties as a :class:`numpy.array`, shape and type
but only represents a contiguous array. The current implementation
is just a container and do not allow standard operator such as
addition, substraction. :epkg:`onnxruntime` has a C implementation
wrapped into a Python class with the same.

.. contents::
    :local:

Python Wrapper for OrtValue
+++++++++++++++++++++++++++

.. autoclass:: onnxruntime.OrtValue
    :members:

C Class OrtValue
++++++++++++++++

.. autoclass:: onnxruntime.capi._pybind_state.OrtValue
    :members:

C Class OrtValueVector
++++++++++++++++++++++

.. autoclass:: onnxruntime.capi._pybind_state.OrtValueVector
    :members:
