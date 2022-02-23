
Summary of onnxruntime and onnxruntime-training API
===================================================

Module :epkg:`onnxcustom` leverages :epkg:`onnxruntime-training` to train models.
Next sections exposes frequent functions uses to run inference
and training with :epkg:`onnxruntime` and :epkg:`onnxruntume-training`.

Most of the code in :epkg:`onnxruntime` is written in C++ and exposed
in Python using :epkg:`pybind11`. For inference, the main class
is :epkg:`InferenceSession`. It wraps C class :ref:`l-ort-inference-session-c`.
The python class is easier to use. Both have the same name.
It adds some short overhead but significant on small models
such as a linear regression.
It is recommended to use C classes in that case
(inference, ortvalue, device).

.. toctree::
    :maxdepth: 1

    helpers
    ortvalue
    sparse
    inference
    training
    training_partial
    training_torch
    exceptions
    grad
