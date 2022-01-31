
onnxruntime helpers
===================

.. contents::
    :local:

Frequent functions
++++++++++++++++++

.. autofunction:: onnxruntime.get_device

.. runpython::
    :showcode:

    import onnxruntime
    print(onnxruntime.get_device())

.. autofunction:: onnxruntime.get_all_providers

.. runpython::
    :showcode:

    import pprint
    import onnxruntime
    pprint.pprint(onnxruntime.get_all_providers())

.. autofunction:: onnxruntime.get_available_providers

.. runpython::
    :showcode:

    import onnxruntime
    import pprint
    pprint.pprint(onnxruntime.get_available_providers())

.. autofunction:: onnxruntime.set_default_logger_severity

.. autofunction:: onnxruntime.set_seed

Python Wrapper OrtDevice
++++++++++++++++++++++++

.. autoclass:: onnxruntime.OrtDevice
    :members:
    :undoc-members:

C class, OrtDevice or C_OrtDevice
+++++++++++++++++++++++++++++++++

.. autoclass:: onnxruntime.capi._pybind_state.OrtDevice
    :members:
    :undoc-members:

OrtMemoryInfo
+++++++++++++

.. autoclass:: onnxruntime.capi._pybind_state.OrtMemoryInfo
    :members:
    :undoc-members:

C classes, frequent types
+++++++++++++++++++++++++

.. autoclass:: onnxruntime.capi._pybind_state.ModelMetadata
    :members:
    :undoc-members:

.. autoclass:: onnxruntime.capi._pybind_state.OrtMemType
    :members:

Rare functions
++++++++++++++

.. autofunction:: onnxruntime.capi._pybind_state.clear_training_ep_instances

.. autofunction:: onnxruntime.capi._pybind_state.create_and_register_allocator

.. autofunction:: onnxruntime.capi._pybind_state.enable_telemetry_events

.. autofunction:: onnxruntime.capi._pybind_state.disable_telemetry_events
