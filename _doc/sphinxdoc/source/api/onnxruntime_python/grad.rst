
Gradient
========

.. contents::
    :local:

C++ API
+++++++

.. autoclass:: onnxruntime.capi._pybind_state.GradientGraphBuilder

.. autoclass:: onnxruntime.capi._pybind_state.GradientNodeAttributeDefinition

.. autoclass:: onnxruntime.capi._pybind_state.GradientNodeDefinition

.. autofunction:: onnxruntime.capi._pybind_state.register_gradient_definition

.. autofunction:: onnxruntime.capi._pybind_state.register_aten_op_executor

.. autofunction:: onnxruntime.capi._pybind_state.register_backward_runner

.. autofunction:: onnxruntime.capi._pybind_state.register_forward_runner

Python API
++++++++++

.. autofunction:: onnxruntime.training.experimental.gradient_graph._gradient_graph_tools.export_gradient_graph
