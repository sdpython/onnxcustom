
Parallelization
===============

.. index:: tutorial

:epkg:`onnxruntime` is already parallelization the computation
on multiple cores if the execution runs on CPU only and obvioulsy
on GPU. Recent machines have multiple GPUs but :epkg:`onnxruntime`
usually runs on one single GPUs. These examples tries to take
advantage of that configuration. The first parallelize the execution
of the same model on each GPU. It assumes a single GPU can host the
whole model. The second model explores a way to split the model
into pieces when the whole model does not hold in one single GPUs.
This is done through function
:func:`split_onnx <onnxcustom.utils.onnx_split.split_onnx>`.

.. toctree::
    :maxdepth: 1

    ../../gyexamples/plot_parallel_execution
    ../../gyexamples/plot_parallel_execution_big_model

The tutorial was tested with following version:

.. runpython::
    :showcode:

    import sys
    import numpy
    import scipy
    import onnx
    import onnxruntime
    import onnxcustom
    import sklearn
    import torch

    print("python {}".format(sys.version_info))
    mods = [numpy, scipy, sklearn, onnx,
            onnxruntime, onnxcustom, torch]
    mods = [(m.__name__, m.__version__) for m in mods]
    mx = max(len(_[0]) for _ in mods) + 1
    for name, vers in sorted(mods):
        print("{}{}{}".format(name, " " * (mx - len(name)), vers))
