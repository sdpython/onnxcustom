
Installation
============

The main dependency is :epkg:`onnxruntime-training`. It is only available
on Linux. It is available from pypi for CPU. GPU versions are available
`download.onnxruntime.ai <https://download.onnxruntime.ai/>`_.
Its installation replaces *onnxruntime* and includes *onnxruntime* and
*onnxruntime-training*.

Installation of onnxruntime-training for GPU
++++++++++++++++++++++++++++++++++++++++++++

onnxruntime-training is only available on Linux. The CPU
can be installed with the following instruction.

::

    pip install onnxruntime-training --extra-index-url https://download.onnxruntime.ai/onnxruntime_nightly_cpu.html

Versions using GPU with CUDA or ROCm are available. Check
`download.onnxruntime.ai <https://download.onnxruntime.ai/>`_
to find a specific version.
You can use it on Windows
inside WSL (Windows Linux Subsystem) or compile it for CPU:

::

    python tools\ci_build\build.py --skip_tests --build_dir .\build\Windows --config Release --build_shared_lib --build_wheel --numpy_version= --cmake_generator="Visual Studio 16 2019" --enable_training --enable_training_ops --enable_training_torch_interop

GPU versions work better on WSL, see `Build onnxruntime on WSL (Windows Linux Subsystem)
<http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/blog/2021/2021-12-16_wsl.html>`_.

Installation of onnxcustom
++++++++++++++++++++++++++

::

    pip install onnxcustom
