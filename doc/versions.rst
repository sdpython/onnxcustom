
Changes
=======

.. contents::
    :local:

Dependencies
++++++++++++

.. runpython::
    :showcode:

    import numpy
    import onnx
    import onnxruntime
    import sklearn
    import onnxconverter_common
    import skl2onnx
    import mlprodict

    for mod in [numpy, onnx, onnxruntime, sklearn,
                onnxconverter_common, skl2onnx, mlprodict]:
        print("%s: %s" % (mod.__name__, mod.__version__))

History
+++++++

**0.1**

First version with almost all examples for the tutorial.
Missing: example with a custom operator with :epkg:`onnxruntime`.
