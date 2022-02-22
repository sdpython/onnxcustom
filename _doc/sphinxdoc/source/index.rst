
onnxcustom: deploy, train machine learned models
================================================

.. image:: https://circleci.com/gh/sdpython/onnxcustom/tree/master.svg?style=svg
    :target: https://circleci.com/gh/sdpython/onnxcustom/tree/master

.. image:: https://travis-ci.com/sdpython/onnxcustom.svg?branch=master
    :target: https://app.travis-ci.com/github/sdpython/onnxcustom
    :alt: Build status

.. image:: https://ci.appveyor.com/api/projects/status/a3sn45a2fayoxb5q?svg=true
    :target: https://ci.appveyor.com/project/sdpython/onnxcustom
    :alt: Build Status Windows

.. image:: https://codecov.io/gh/sdpython/onnxcustom/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/sdpython/onnxcustom

.. image:: https://badge.fury.io/py/onnxcustom.svg
    :target: http://badge.fury.io/py/onnxcustom

.. image:: http://img.shields.io/github/issues/sdpython/onnxcustom.png
    :alt: GitHub Issues
    :target: https://github.com/sdpython/onnxcustom/issues

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :alt: MIT License
    :target: http://opensource.org/licenses/MIT

.. image:: https://pepy.tech/badge/onnxcustom/month
    :target: https://pepy.tech/project/onnxcustom/month
    :alt: Downloads

.. image:: https://img.shields.io/github/forks/sdpython/onnxcustom.svg
    :target: https://github.com/sdpython/onnxcustom/
    :alt: Forks

.. image:: https://img.shields.io/github/stars/sdpython/onnxcustom.svg
    :target: https://github.com/sdpython/onnxcustom/
    :alt: Stars

.. image:: https://img.shields.io/github/repo-size/sdpython/onnxcustom
    :target: https://github.com/sdpython/onnxcustom/
    :alt: size

Examples, tutorial on how to convert machine learned models into ONNX,
implement your own converter or runtime, or even train with :epkg:`ONNX`,
:epkg:`onnxruntime`.
The documentation introduces :epkg:`onnx`, :epkg:`onnxruntime` for
inference and training. It implements training classes following
:epkg:`scikit-learn` based on :epkg:`onnxruntime-training` enabling training
linear models, neural networks on CPU or GPU.
It implements tools to manipulate logs produced NVidia Profiler logs
(:func:`convert_trace_to_json <onnxcustom.utils.nvprof2json.convert_trace_to_json>`),
tools to manipulate :epkg:`onnx` graphs.

Section :ref:`l-apis` summarizes APIs for :epkg:`onnx`, :epkg:`onnxruntime`,
and this package. Section :ref:`l-tutorials` explains the logic behind
:epkg:`onnx`, :epkg:`onnxruntime` and this package. It guides the user
through all the examples this documentation contains.

**Contents**

.. toctree::
    :maxdepth: 1

    tutorials/index
    api/apis
    gyexamples/index
    all_notebooks
    other_pages
    blog/blogindex

Sources are available on
`github/onnxcustom <https://github.com/sdpython/onnxcustom>`_.
Package is available on `pypi <https://pypi.python.org/pypi/onnxcustom/>`_,
:ref:`l-README`, and a blog for unclassified topics :ref:`blog <ap-main-0>`.
The tutorial related to :epkg:`scikit-learn` has been merged into
`sklearn-onnx documentation
<http://onnx.ai/sklearn-onnx/index_tutorial.html>`_.
This package supports ONNX opsets to the latest opset stored
in `onnxcustom.__max_supported_opset__` which is:

.. runpython::
    :showcode:

    import onnxcustom
    print(onnxcustom.__max_supported_opset__)

Any opset beyond that value is not supported and could fail.
That's for the main set of ONNX functions or domain.
Converters for :epkg:`scikit-learn` requires another domain,
`'ai.onnxml'` to implement tree. Latest supported options
are defined here:

.. runpython::
    :showcode:

    import pprint
    import onnxcustom
    pprint.pprint(onnxcustom.__max_supported_opsets__)

+----------------------+---------------------+---------------------+--------------------+------------------------+------------------------------------------------+
| :ref:`l-modules`     |  :ref:`l-functions` | :ref:`l-classes`    | :ref:`l-methods`   | :ref:`l-staticmethods` | :ref:`l-properties`                            |
+----------------------+---------------------+---------------------+--------------------+------------------------+------------------------------------------------+
| :ref:`modindex`      |  :ref:`l-EX2`       | :ref:`search`       | :ref:`l-license`   | :ref:`l-changes`       | :ref:`l-README`                                |
+----------------------+---------------------+---------------------+--------------------+------------------------+------------------------------------------------+
| :ref:`genindex`      |  :ref:`l-FAQ2`      | :ref:`l-notebooks`  | :ref:`l-NB2`       | :ref:`l-statcode`      | `Unit Test Coverage <coverage/index.html>`_    |
+----------------------+---------------------+---------------------+--------------------+------------------------+------------------------------------------------+
