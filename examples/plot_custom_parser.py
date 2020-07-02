"""
Change the number of outputs by adding a parser
===============================================

By default, :epkg:`sklearn-onnx` assumes that a classifier
has two outputs (label and probabilities), a regressor
has one output (prediction), a transform has one output
(the transformed data). What if it is not the case?
The following example creates a custom converter
and a custom parser which defines the number of outputs
expected by the converted model.

*to be continued*
"""
