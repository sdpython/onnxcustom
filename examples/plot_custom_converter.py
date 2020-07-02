"""
Implement a new converter
=========================

By default, :epkg:`sklearn-onnx` assumes that a classifier
has two outputs (label and probabilities), a regressor
has one output (prediction), a transform has one output
(the transformed data). This example assumes the model to
convert is one of them. In that case, a new converter requires
in fact two functions:

* a shape calculator: it defines the output shape and type
  based on the model and input type,
* a converter: it actually builds an ONNX graph equivalent
  to the prediction function to be converted.

This example implements both components for a new model.

*to be continued*
"""
