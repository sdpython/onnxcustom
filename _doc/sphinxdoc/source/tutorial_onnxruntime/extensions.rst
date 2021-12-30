
==========
Extensions
==========

.. contents::
    :local:

C API
=====

:epkg:`onnxruntime` implements a C API in three files:

* `onnxruntime_c_api.h <https://github.com/microsoft/onnxruntime/blob/master/include/onnxruntime/core/session/onnxruntime_c_api.h>`_
* `onnxruntime_cxx_api.h <https://github.com/microsoft/onnxruntime/blob/master/include/onnxruntime/core/session/onnxruntime_cxx_api.h>`_
* `onnxruntime_cxx_inline.h <https://github.com/microsoft/onnxruntime/blob/master/include/onnxruntime/core/session/onnxruntime_cxx_inline.h>`_

Other languages
===============

:epkg:`onnxruntime` is available in others languages such as C#, java, javascript,
webassembly, Objective C.

.. _l-custom-runtime-extensions:

Custom runtime
==============

Project :epkg:`onnxruntime-extensions` leverages the C API to implement
a runtime for a couple of tokenizers used by :epkg:`tensorflow` models.
`PR 148 <https://github.com/microsoft/onnxruntime-extensions/pull/148>`_
shows how to add a new operator dealing with text.

Tools
=====

`perfstats.py <https://github.com/microsoft/onnxconverter-common/
blob/master/onnxconverter_common/perfstats.py>`_
reads a file produced by a profiling. It returns the time in every
operator or type of operators in a table. It helps find where the
runtime spends most of its time.
