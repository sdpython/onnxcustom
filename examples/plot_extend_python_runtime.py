"""

.. _l-extend-python-runtime:

Extend a python runtime
=======================

:epkg:`ONNX operators` does not contain operator
from :epkg:`numpy`. There is no operator for
`solve <https://numpy.org/doc/stable/reference/
generated/numpy.linalg.solve.html>`_ but this one
is needed to implement the prediction function
of model :epkg:`NMF`. The converter can be written
including a new ONNX operator but then it requires a
runtime for it to be tested. This example shows how
to be that with the python runtime implemented in
:epkg:`mlprodict`. It may not be :epkg:`onnxruntime`
but that speeds up the implementation of the converter.

*to be continued*
"""
