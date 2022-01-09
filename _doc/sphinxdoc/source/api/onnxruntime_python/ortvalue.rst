
OrtValue
========

:epkg:`onnxruntime` implements tensors with class :epkg:`OrtValue`.
It has the same properties as a :class:`numpy.array`, shape and type
but only represents a contiguous array. The current implementation
is just a container and does not allow standard operators such as
addition, substraction. :epkg:`onnxruntime` has a C implementation
wrapped into a Python class with the same.

.. contents::
    :local:

Python Wrapper for OrtValue
+++++++++++++++++++++++++++

.. note::
    Method `ortvalue_from_numpy` does not copy data, it borrows
    the data pointer. The numpy array must remain alive while
    the instance of OrtValue is in use.

.. autoclass:: onnxruntime.OrtValue
    :members:
    :undoc-members:

C Class OrtValue or C_OrtValue
++++++++++++++++++++++++++++++

.. autoclass:: onnxruntime.capi._pybind_state.OrtValue
    :members:
    :undoc-members:

C Class OrtValueVector
++++++++++++++++++++++

.. autoclass:: onnxruntime.capi._pybind_state.OrtValueVector
    :members:
    :undoc-members:
