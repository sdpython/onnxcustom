
=====
Tools
=====

.. contents::
    :local:

Quantization
============

The main functions.

.. autofunction:: onnxruntime.quantization.quantize.quantize_dynamic

.. autofunction:: onnxruntime.quantization.quantize.quantize_static

.. autofunction:: onnxruntime.quantization.shape_inference.quant_pre_process

Calibration:

.. autoclass:: onnxruntime.quantization.calibrate.CalibrationDataReader
    :members:

The parameters.

.. autoclass:: onnxruntime.quantization.quant_utils.QuantFormat
    :members:

.. autoclass:: onnxruntime.quantization.quant_utils.QuantizationMode
    :members:

.. autoclass:: onnxruntime.quantization.quant_utils.QuantType
    :members:
