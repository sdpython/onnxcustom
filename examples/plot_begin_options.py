"""
One model, many possible conversions with options
=================================================

There is not one way to convert a model. A new operator
might have been added in a newer version of :epkg:`ONNX`
and that speeds up the converted model. The rational choice
would be to use this new operator but what means the associated
runtime has an implementation for it. What if two different
users needs two different conversion for the same model?
Let's see how this may be done.

*to be continued*
"""
