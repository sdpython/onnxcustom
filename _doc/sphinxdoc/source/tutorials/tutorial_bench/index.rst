
Benchmarking and profiling Tutorial
===================================

.. index:: tutorial

Next sections shows how to measure performance of an ONNX graph
when executing with :epkg:`onnxruntime`.

.. toctree::
    :maxdepth: 2

    tutorial_op
    tutorial_benchmark
    tutorial_profile
    tutorial_training

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
