
Training
========

.. contents::
    :local:

.. faqref::
    :title: Differences between onnxruntime and onnxruntime-training

    onnxruntime-training is an extension of onnxruntime
    that supports training. Version 1.10 is obtained by compiling
    onnxruntime from the sources with different flags.
    One example:

    ::

        python ./tools/ci_build/build.py --build_dir ./build/debian \\
               --config Release --build_wheel --numpy_version= \\
               --skip_tests --build_shared_lib --enable_training \\
               --enable_training_ops --enable_training_torch_interop \\
               --parallel

.. _l-ort-training-session:

Python Wrapper for TrainingSession
++++++++++++++++++++++++++++++++++

.. autoclass:: onnxruntime.TrainingSession
    :members:
    :inherited-members:
    :undoc-members:

.. _l-ort-training-session-c:

C Class TrainingSession
+++++++++++++++++++++++

.. autoclass:: onnxruntime.capi._pybind_state.TrainingSession
    :members:
    :undoc-members:

TrainingParameters
++++++++++++++++++

.. autoclass:: onnxruntime.capi._pybind_state.TrainingParameters
    :members:
    :undoc-members:

GradientGraphBuilder
++++++++++++++++++++

.. autoclass:: onnxruntime.capi._pybind_state.GradientGraphBuilder
    :members:
    :undoc-members:
