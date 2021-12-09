
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
This package is used to write documentation in an early stage it
is moved to over documentation website.
Some of the tutorials has been merged into `sklearn-onnx documentation
<http://onnx.ai/sklearn-onnx/index_tutorial.html>`_.
Among the tools this package implements, you may find:

* a tool to convert NVidia Profilder logs into a dataframe:
  :func:`convert_trace_to_json <onnxcustom.utils.nvprof2json.convert_trace_to_json>`
* A couple of SGD optimizer similar to what scikit-learn implements
  (see `Stochastic Gradient Descent
  <https://scikit-learn.org/stable/modules/sgd.html>`_)
  but based on :epkg:`onnxruntime-training` and able to train an CPU and GPU
  (see example :ref:`l-orttraining-nn-gpu`).

**Contents**

.. toctree::
    :maxdepth: 1

    tutorial_onnx/index
    tutorial/index
    tutorial_training/index
    api/apis
    onnxmd/index
    gyexamples/index
    all_notebooks
    license
    other_pages

Sources are available on
`github/onnxcustom <https://github.com/sdpython/onnxcustom>`_.

+----------------------+---------------------+---------------------+--------------------+------------------------+------------------------------------------------+
| :ref:`l-modules`     |  :ref:`l-functions` | :ref:`l-classes`    | :ref:`l-methods`   | :ref:`l-staticmethods` | :ref:`l-properties`                            |
+----------------------+---------------------+---------------------+--------------------+------------------------+------------------------------------------------+
| :ref:`modindex`      |  :ref:`l-EX2`       | :ref:`search`       | :ref:`l-license`   | :ref:`l-changes`       | :ref:`l-README`                                |
+----------------------+---------------------+---------------------+--------------------+------------------------+------------------------------------------------+
| :ref:`genindex`      |  :ref:`l-FAQ2`      | :ref:`l-notebooks`  | :ref:`l-NB2`       | :ref:`l-statcode`      | `Unit Test Coverage <coverage/index.html>`_    |
+----------------------+---------------------+---------------------+--------------------+------------------------+------------------------------------------------+
