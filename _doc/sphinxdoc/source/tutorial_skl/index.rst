
scikit-learn to ONNX Tutorial
=============================

.. index:: tutorial

The tutorial goes from a simple example which
converts a pipeline to a more complex example
involving operator not actually implemented in
:epkg:`ONNX operators` or :epkg:`ONNX ML Operators`.

.. toctree::
    :maxdepth: 2

    tutorial_1_simple
    tutorial_1-5_external
    tutorial_2_new_converter
    tutorial_3_new_operator
    tutorial_4_complex

The tutorial was tested with following version:

.. runpython::
    :showcode:

    import sys
    import numpy
    import scipy
    import onnx
    import onnxruntime
    import lightgbm
    import xgboost
    import sklearn
    import onnxconverter_common
    import onnxmltools
    import skl2onnx
    import pyquickhelper
    import mlprodict
    import onnxcustom

    print("python {}".format(sys.version_info))
    mods = [numpy, scipy, sklearn, lightgbm, xgboost,
            onnx, onnxmltools, onnxruntime, onnxcustom,
            onnxconverter_common,
            skl2onnx, mlprodict, pyquickhelper]
    mods = [(m.__name__, m.__version__) for m in mods]
    mx = max(len(_[0]) for _ in mods) + 1
    for name, vers in sorted(mods):
        print("{}{}{}".format(name, " " * (mx - len(name)), vers))
