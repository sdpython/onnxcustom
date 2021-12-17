
onnxruntime concepts
====================

.. contents::
    :local:

Tools
+++++

`perfstats.py <https://github.com/microsoft/onnxconverter-common/
blob/master/onnxconverter_common/perfstats.py>`_
reads a file produced by a profiling. It returns the time in every
operator or type of operators in a table. It helps find where the
runtime spends most of its time.
