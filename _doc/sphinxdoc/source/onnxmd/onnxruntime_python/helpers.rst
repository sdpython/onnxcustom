
onnxruntime helpers
===================

.. contents::
    :local:

Frequent functions
++++++++++++++++++

.. autofunction:: onnxruntime.get_device

.. autofunction:: onnxruntime.capi._pybind_state.set_seed

.. autofunction:: onnxruntime.capi._pybind_state.set_default_logger_severity

.. autofunction:: onnxruntime.capi._pybind_state.get_all_providers

Python Wrapper OrtDevice
++++++++++++++++++++++++

.. autoclass:: onnxruntime.capi._pybind_state.OrtDevice
    :members:
    :undoc-members:

C class, OrtDevice
++++++++++++++++++

.. autoclass:: onnxruntime.capi._pybind_state.OrtDevice
    :members:
    :undoc-members:

C classes, frequent types
+++++++++++++++++++++++++

.. autoclass:: onnxruntime.capi._pybind_state.ModelMetadata
    :members:
    :undoc-members:

.. autoclass:: onnxruntime.capi._pybind_state.OrtMemType
    :members:
    :undoc-members:

Rare functions
++++++++++++++

.. autofunction:: onnxruntime.capi._pybind_state.enable_telemetry_events

.. autofunction:: onnxruntime.capi._pybind_state.disable_telemetry_events

.. autofunction:: onnxruntime.capi._pybind_state.create_and_register_allocator

.. autofunction:: onnxruntime.capi._pybind_state._register_provider_lib
